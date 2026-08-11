"""
Prototype 2 — Neural Network Weight Compression Experiment

This script validates the TurboQuant Lab compression engine on REAL PyTorch neural network weights.

Pipeline Steps:
1. Extract weight tensors from a PyTorch SmallNeuralNetwork model.
2. Quantize each tensor using the existing quantize() + pack_bits() engine (8-bit, 4-bit, 2-bit).
3. Reconstruct weight tensors using unpack_bits() + dequantize().
4. Compute layer-by-layer and total model storage and error metrics.
5. Generate weight quality comparison plots in 'results/'.
6. Reconstruct the model parameters with dequantized weights and run forward-pass output validation.
"""

import os
import sys
import copy
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Add parent directory to sys.path to import compression module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.small_network import get_small_network, SmallNeuralNetwork
from compression.bit_packing import pack_bits, unpack_bits
from compression.quantizer import (
    quantize,
    dequantize,
    calculate_mse,
    calculate_mae,
    calculate_max_error,
    calculate_actual_storage,
)


def run_model_compression_experiment():
    print("=" * 80)
    print("TURBOQUANT LAB - PROTOTYPE 2: PYTORCH NEURAL NETWORK WEIGHT COMPRESSION")
    print("=" * 80)

    # Instantiate PyTorch SmallNeuralNetwork
    model = get_small_network(seed=42)
    model.eval()

    # STEP 1: Extract Weights
    print("\n[STEP 1] Extracting PyTorch Model Parameters...")
    param_info_list = []
    total_original_bytes = 0
    total_param_count = 0

    for name, param in model.named_parameters():
        if not param.requires_grad and not param.is_floating_point():
            continue
        
        param_np = np.array(param.detach().cpu().tolist(), dtype=np.float32)
        num_params = param.numel()
        dtype_str = str(param.dtype)
        orig_bytes = num_params * param.element_size()

        total_param_count += num_params
        total_original_bytes += orig_bytes

        param_info_list.append({
            'name': name,
            'shape': list(param.shape),
            'num_params': num_params,
            'dtype': dtype_str,
            'orig_bytes': orig_bytes,
            'array': param_np
        })
        print(f"  - Parameter: {name:15s} | Shape: {str(list(param.shape)):12s} | Params: {num_params:7,d} | FP32 Bytes: {orig_bytes:8,d} B")

    print(f"\nModel Total Parameters: {total_param_count:,}")
    print(f"Model Original FP32 Footprint: {total_original_bytes:,} Bytes ({total_original_bytes / 1024:.2f} KB)")

    # STEP 2 & 3: Quantize & Reconstruct per Precision Level
    bit_widths = [8, 4, 2]
    precisions = ['INT8', 'INT4', 'INT2']
    
    # Structure to hold layer-level and model-level metrics
    results_by_precision = {}
    reconstructed_models = {}

    for bits, prec in zip(bit_widths, precisions):
        print("\n" + "-" * 70)
        print(f"[STEP 2 & 3] Quantizing & Reconstructing Model Weights at {prec} ({bits}-bit)...")
        print("-" * 70)

        layer_results = []
        total_packed_bytes = 0
        total_metadata_bytes = 0
        total_compressed_bytes = 0

        # We will reconstruct a model copy
        rec_model = copy.deepcopy(model)
        rec_model.eval()

        for info in param_info_list:
            name = info['name']
            orig_array = info['array'].flatten()
            num_params = info['num_params']
            orig_bytes = info['orig_bytes']

            # STEP 2: Quantize using existing engine
            quantized_vals, scale, zero_point = quantize(orig_array, bits=bits)
            packed_bytes = pack_bits(quantized_vals, bits=bits)

            # STEP 3: Reconstruct using existing engine
            unpacked_vals = unpack_bits(packed_bytes, bits=bits, original_length=num_params)
            reconstructed_vals = dequantize(unpacked_vals, scale, zero_point, bits=bits)
            reconstructed_tensor_np = reconstructed_vals.reshape(info['shape'])

            # STEP 4: Calculate metrics per layer
            storage = calculate_actual_storage(
                original_length=num_params,
                packed_bytes_count=len(packed_bytes),
                bits=bits,
                metadata_bytes=13,
                original_element_bytes=4
            )

            mse = calculate_mse(orig_array, reconstructed_vals)
            mae = calculate_mae(orig_array, reconstructed_vals)
            max_err = calculate_max_error(orig_array, reconstructed_vals)

            total_packed_bytes += len(packed_bytes)
            total_metadata_bytes += 13
            total_compressed_bytes += storage['total_compressed_bytes']

            layer_results.append({
                'layer_name': name,
                'shape': info['shape'],
                'param_count': num_params,
                'orig_bytes': orig_bytes,
                'packed_bytes': len(packed_bytes),
                'metadata_bytes': 13,
                'total_compressed_bytes': storage['total_compressed_bytes'],
                'actual_compression_ratio': storage['actual_compression_ratio'],
                'storage_saved_percent': storage['actual_storage_saved_percent'],
                'mse': mse,
                'mae': mae,
                'max_error': max_err,
                'scale': scale,
                'zero_point': zero_point,
                'orig_array': orig_array,
                'reconstructed_array': reconstructed_vals
            })

            # Replace parameters in reconstructed PyTorch model
            with torch.no_grad():
                dict(rec_model.named_parameters())[name].copy_(
                    torch.tensor(reconstructed_tensor_np.tolist(), dtype=dict(rec_model.named_parameters())[name].dtype)
                )

            print(f"  Layer {name:15s} | Orig: {orig_bytes:7,d}B | Packed: {len(packed_bytes):6,d}B | Total: {storage['total_compressed_bytes']:6,d}B | Ratio: {storage['actual_compression_ratio']:.2f}x | MSE: {mse:.6e}")

        # Model Overall Metrics for this Precision
        overall_ratio = total_original_bytes / total_compressed_bytes if total_compressed_bytes > 0 else 1.0
        overall_saved_pct = (1.0 - (total_compressed_bytes / total_original_bytes)) * 100.0 if total_original_bytes > 0 else 0.0

        all_orig_concat = np.concatenate([info['array'].flatten() for info in param_info_list])
        all_rec_concat = np.concatenate([res['reconstructed_array'] for res in layer_results])

        overall_mse = calculate_mse(all_orig_concat, all_rec_concat)
        overall_mae = calculate_mae(all_orig_concat, all_rec_concat)
        overall_max_err = calculate_max_error(all_orig_concat, all_rec_concat)

        results_by_precision[prec] = {
            'bits': bits,
            'total_param_count': total_param_count,
            'total_original_bytes': total_original_bytes,
            'total_packed_bytes': total_packed_bytes,
            'total_metadata_bytes': total_metadata_bytes,
            'total_compressed_bytes': total_compressed_bytes,
            'overall_compression_ratio': float(overall_ratio),
            'overall_storage_saved_percent': float(overall_saved_pct),
            'overall_mse': float(overall_mse),
            'overall_mae': float(overall_mae),
            'overall_max_error': float(overall_max_err),
            'layers': layer_results,
            'all_orig_concat': all_orig_concat,
            'all_rec_concat': all_rec_concat
        }
        reconstructed_models[prec] = rec_model

        print(f"\n[{prec} MODEL TOTALS]")
        print(f"  Total Compressed Footprint: {total_compressed_bytes:,} Bytes ({total_compressed_bytes/1024:.2f} KB)")
        print(f"  Overall Compression Ratio : {overall_ratio:.2f}x")
        print(f"  Storage Saved             : {overall_saved_pct:.2f}%")
        print(f"  Model Weight Mean MSE     : {overall_mse:.6e}")

    # STEP 6: Disclaimer Logging
    print("\n" + "=" * 80)
    print("STEP 6 DISCLAIMER:")
    print("At this stage we are only testing: 'Can our compression engine compress and reconstruct real neural-network weight tensors?'")
    print("We do NOT claim that the model itself has been fine-tuned or post-training quantized for inference hardware execution yet.")
    print("=" * 80)

    # STEP 7: Validation & Forward Pass Comparison
    print("\n[STEP 7] Validating Reconstructed Models & Output Error...")
    torch.manual_seed(100)
    sample_input = torch.randn(16, 784)  # Batch of 16 sample inputs

    # Forward pass on original model
    with torch.no_grad():
        orig_output = np.array(model(sample_input).tolist(), dtype=np.float32)

    forward_validation_results = {}

    for prec in precisions:
        rec_model = reconstructed_models[prec]
        
        # Verify model loading & non-empty
        rec_params = dict(rec_model.named_parameters())
        has_nan = False
        has_inf = False
        param_count_match = True

        for name, orig_param in model.named_parameters():
            rec_p = rec_params[name]
            if torch.isnan(rec_p).any():
                has_nan = True
            if torch.isinf(rec_p).any():
                has_inf = True
            if rec_p.shape != orig_param.shape:
                param_count_match = False

        # Run forward pass
        with torch.no_grad():
            rec_output = np.array(rec_model(sample_input).tolist(), dtype=np.float32)

        output_mse = calculate_mse(orig_output, rec_output)
        output_mae = calculate_mae(orig_output, rec_output)
        output_max_diff = calculate_max_error(orig_output, rec_output)

        forward_validation_results[prec] = {
            'has_nan': has_nan,
            'has_inf': has_inf,
            'param_shapes_matched': param_count_match,
            'output_mse': float(output_mse),
            'output_mae': float(output_mae),
            'output_max_diff': float(output_max_diff)
        }

        print(f"\n  [{prec} Forward Pass Validation]")
        print(f"    - Tensor Shapes Match : {param_count_match}")
        print(f"    - Contains NaN/Inf    : {has_nan or has_inf}")
        print(f"    - Output Prediction MSE: {output_mse:.6e}")
        print(f"    - Output Prediction MAE: {output_mae:.6e}")
        print(f"    - Max Output Deviation : {output_max_diff:.6e}")

    # STEP 5: Generate Visualizations
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    os.makedirs(results_dir, exist_ok=True)

    print("\n[STEP 5] Generating Weight Quality Analysis Plots...")
    generate_weight_analysis_plots(param_info_list, results_by_precision, forward_validation_results, results_dir)

    # Save JSON summary
    save_json_results(total_param_count, total_original_bytes, results_by_precision, forward_validation_results, results_dir)

    print("\n[SUCCESS] Prototype 2 Model Compression Experiment Complete!")


def generate_weight_analysis_plots(param_info_list, results_by_precision, forward_validation_results, results_dir):
    """Generates the 4 required plots for weight quality analysis."""

    all_orig_concat = results_by_precision['INT8']['all_orig_concat']

    # Plot 1: Weight Distribution Comparison
    fig1, ax1 = plt.subplots(figsize=(10, 6), dpi=150)
    ax1.hist(all_orig_concat, bins=80, density=True, alpha=0.4, color='#2563eb', label='Original FP32 Weights')
    
    colors = {'INT8': '#059669', 'INT4': '#d97706', 'INT2': '#dc2626'}
    for prec in ['INT8', 'INT4', 'INT2']:
        rec = results_by_precision[prec]['all_rec_concat']
        ax1.hist(rec, bins=80, density=True, histtype='step', linewidth=1.8, color=colors[prec], label=f'{prec} Reconstructed')

    ax1.set_title('PyTorch Model Weight Density Distribution across Precision Levels', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Weight Value Range')
    ax1.set_ylabel('Probability Density')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'model_weight_distribution.png'))
    plt.close()

    # Plot 2: Original vs Reconstructed Weight Scatter
    fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=150)
    sample_size = min(500, len(all_orig_concat))
    indices = np.random.choice(len(all_orig_concat), size=sample_size, replace=False)

    ax2.plot(all_orig_concat[indices], all_orig_concat[indices], 'k--', alpha=0.5, label='Ideal Perfect Match (1:1)')
    for prec in ['INT8', 'INT4', 'INT2']:
        rec = results_by_precision[prec]['all_rec_concat']
        ax2.scatter(all_orig_concat[indices], rec[indices], alpha=0.6, s=12, color=colors[prec], label=f'{prec} Reconstructed')

    ax2.set_title('Original vs Reconstructed Weight Correspondence (500 Sample Sub-Sample)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Original Weight Value (FP32)')
    ax2.set_ylabel('Reconstructed Weight Value')
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'model_orig_vs_reconstructed.png'))
    plt.close()

    # Plot 3: Quantization Error Histogram
    fig3, ax3 = plt.subplots(figsize=(10, 6), dpi=150)
    for prec in ['INT8', 'INT4', 'INT2']:
        err = all_orig_concat - results_by_precision[prec]['all_rec_concat']
        ax3.hist(err, bins=80, alpha=0.5, color=colors[prec], label=f'{prec} Error (MSE: {results_by_precision[prec]["overall_mse"]:.2e})')

    ax3.set_title('Model Weight Quantization Residual Error Distribution (Original - Reconstructed)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Error Value (x - x_hat)')
    ax3.set_ylabel('Parameter Count')
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'model_quantization_error_hist.png'))
    plt.close()

    # Plot 4: Compression Ratio by Layer
    fig4, ax4 = plt.subplots(figsize=(10, 6), dpi=150)
    layer_names = [res['layer_name'] for res in results_by_precision['INT8']['layers']]
    x = np.arange(len(layer_names))
    width = 0.25

    int8_ratios = [res['actual_compression_ratio'] for res in results_by_precision['INT8']['layers']]
    int4_ratios = [res['actual_compression_ratio'] for res in results_by_precision['INT4']['layers']]
    int2_ratios = [res['actual_compression_ratio'] for res in results_by_precision['INT2']['layers']]

    ax4.bar(x - width, int8_ratios, width, label='INT8 Compression Ratio', color=colors['INT8'])
    ax4.bar(x, int4_ratios, width, label='INT4 Compression Ratio', color=colors['INT4'])
    ax4.bar(x + width, int2_ratios, width, label='INT2 Compression Ratio', color=colors['INT2'])

    ax4.set_title('Layer-by-Layer Actual Compression Ratios (Including Header Overhead)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Model Parameter Layer')
    ax4.set_ylabel('Compression Multiplier Ratio')
    ax4.set_xticks(x)
    ax4.set_xticklabels(layer_names, rotation=20, ha='right')
    ax4.legend(loc='upper right')
    ax4.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'model_compression_by_layer.png'))
    plt.close()

    print("  [PLOTS GENERATED]:")
    print(f"    - {os.path.join(results_dir, 'model_weight_distribution.png')}")
    print(f"    - {os.path.join(results_dir, 'model_orig_vs_reconstructed.png')}")
    print(f"    - {os.path.join(results_dir, 'model_quantization_error_hist.png')}")
    print(f"    - {os.path.join(results_dir, 'model_compression_by_layer.png')}")


def save_json_results(total_param_count, total_original_bytes, results_by_precision, forward_validation_results, results_dir):
    """Saves structured JSON summary of Prototype 2 results."""

    summary_json = {
        'model_architecture': 'SmallNeuralNetwork (Linear 784->128 -> Linear 128->64 -> Linear 64->10)',
        'total_parameters': total_param_count,
        'total_original_bytes': total_original_bytes,
        'disclaimer': "Testing weight tensor compression engine applicability. Not full inference engine quantization.",
        'precisions': {}
    }

    for prec in ['INT8', 'INT4', 'INT2']:
        pdata = results_by_precision[prec]
        fval = forward_validation_results[prec]

        summary_json['precisions'][prec] = {
            'bits': pdata['bits'],
            'total_compressed_bytes': pdata['total_compressed_bytes'],
            'overall_compression_ratio': round(pdata['overall_compression_ratio'], 3),
            'overall_storage_saved_percent': round(pdata['overall_storage_saved_percent'], 2),
            'overall_mse': pdata['overall_mse'],
            'overall_mae': pdata['overall_mae'],
            'overall_max_error': pdata['overall_max_error'],
            'output_validation': {
                'output_mse': fval['output_mse'],
                'output_mae': fval['output_mae'],
                'output_max_diff': fval['output_max_diff'],
                'param_shapes_matched': fval['param_shapes_matched'],
                'has_nan_or_inf': fval['has_nan'] or fval['has_inf']
            },
            'layer_breakdown': [
                {
                    'layer_name': l['layer_name'],
                    'shape': l['shape'],
                    'param_count': l['param_count'],
                    'orig_bytes': l['orig_bytes'],
                    'packed_bytes': l['packed_bytes'],
                    'total_compressed_bytes': l['total_compressed_bytes'],
                    'compression_ratio': round(l['actual_compression_ratio'], 3),
                    'mse': l['mse'],
                    'mae': l['mae']
                }
                for l in pdata['layers']
            ]
        }

    out_file = os.path.join(results_dir, 'model_compression_results.json')
    with open(out_file, 'w') as f:
        json.dump(summary_json, f, indent=2)

    print(f"  [JSON RESULTS SAVED]: {out_file}")


if __name__ == '__main__':
    run_model_compression_experiment()
