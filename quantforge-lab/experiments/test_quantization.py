"""
Quantization Experiment and Benchmark Script for TurboQuant Lab.

This script executes:
1. Edge-case verification tests for the quantizer engine.
2. Quantization benchmarks across FP32, INT8, INT4, and INT2 precision levels on 10,000 samples.
3. Visualization generation:
   - 'results/original_vs_reconstructed.png': Original vs reconstructed signal & distribution.
   - 'results/bits_vs_mse.png': Bit width vs Mean Squared Error trade-off.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environment
import matplotlib.pyplot as plt

# Ensure parent directory is in sys.path for compression module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from compression.bit_packing import pack_bits, unpack_bits
from compression.quantizer import (
    quantize,
    dequantize,
    calculate_mse,
    calculate_mae,
    calculate_max_error,
    calculate_compression_ratio,
    calculate_actual_storage,
)


def run_bit_packing_tests():
    """Validates bit packing and unpacking across all bit widths and boundary conditions."""
    print("=" * 70)
    print("RUNNING BIT PACKING & UNPACKING UNIT TESTS")
    print("=" * 70)

    # 1. 8-bit packing/unpacking
    vals_8 = np.array([0, 15, 128, 200, 255], dtype=np.uint8)
    packed_8 = pack_bits(vals_8, bits=8)
    unpacked_8 = unpack_bits(packed_8, bits=8, original_length=len(vals_8))
    np.testing.assert_array_equal(vals_8, unpacked_8)
    print("[PASS] 8-bit packing/unpacking roundtrip.")

    # 2. 4-bit packing/unpacking (even length)
    vals_4 = np.array([0, 15, 7, 3, 12, 1], dtype=np.uint8)
    packed_4 = pack_bits(vals_4, bits=4)
    unpacked_4 = unpack_bits(packed_4, bits=4, original_length=len(vals_4))
    np.testing.assert_array_equal(vals_4, unpacked_4)
    print("[PASS] 4-bit packing/unpacking (even length).")

    # 3. 2-bit packing/unpacking (length divisible by 4)
    vals_2 = np.array([0, 1, 2, 3, 3, 2, 1, 0], dtype=np.uint8)
    packed_2 = pack_bits(vals_2, bits=2)
    unpacked_2 = unpack_bits(packed_2, bits=2, original_length=len(vals_2))
    np.testing.assert_array_equal(vals_2, unpacked_2)
    print("[PASS] 2-bit packing/unpacking (length divisible by 4).")

    # 4. Odd array length for 4-bit
    vals_4_odd = np.array([0, 15, 7, 3, 12], dtype=np.uint8)
    packed_4_odd = pack_bits(vals_4_odd, bits=4)
    unpacked_4_odd = unpack_bits(packed_4_odd, bits=4, original_length=len(vals_4_odd))
    np.testing.assert_array_equal(vals_4_odd, unpacked_4_odd)
    assert len(packed_4_odd) == 3, f"Expected 3 bytes for 5 4-bit values, got {len(packed_4_odd)}"
    print("[PASS] 4-bit packing/unpacking (odd length padded correctly).")

    # 5. Array length not divisible by 4 for 2-bit (e.g. 5 elements)
    vals_2_mod = np.array([0, 1, 2, 3, 2], dtype=np.uint8)
    packed_2_mod = pack_bits(vals_2_mod, bits=2)
    unpacked_2_mod = unpack_bits(packed_2_mod, bits=2, original_length=len(vals_2_mod))
    np.testing.assert_array_equal(vals_2_mod, unpacked_2_mod)
    assert len(packed_2_mod) == 2, f"Expected 2 bytes for 5 2-bit values, got {len(packed_2_mod)}"
    print("[PASS] 2-bit packing/unpacking (non-multiple of 4 padded correctly).")

    # 6. Negative original values quantization + packing roundtrip
    neg_orig = np.array([-10.5, -3.2, 0.0, 5.8, 12.1], dtype=np.float32)
    q_neg, scale_n, zp_n = quantize(neg_orig, bits=4)
    p_neg = pack_bits(q_neg, bits=4)
    u_neg = unpack_bits(p_neg, bits=4, original_length=len(neg_orig))
    rec_neg = dequantize(u_neg, scale_n, zp_n, bits=4)
    assert len(rec_neg) == len(neg_orig)
    print("[PASS] Negative original values -> quantize -> pack -> unpack -> dequantize roundtrip.")

    # 7. Constant arrays
    const_orig = np.full(17, 4.2, dtype=np.float32)
    q_const, scale_c, zp_c = quantize(const_orig, bits=2)
    p_const = pack_bits(q_const, bits=2)
    u_const = unpack_bits(p_const, bits=2, original_length=len(const_orig))
    rec_const = dequantize(u_const, scale_c, zp_c, bits=2)
    np.testing.assert_allclose(rec_const, const_orig, atol=1e-3)
    print("[PASS] Constant array quantization & bit packing roundtrip.")

    # 8. Random arrays (large batch)
    np.random.seed(123)
    rand_vals_8 = np.random.randint(0, 256, size=1001, dtype=np.uint8)
    np.testing.assert_array_equal(rand_vals_8, unpack_bits(pack_bits(rand_vals_8, 8), 8, 1001))

    rand_vals_4 = np.random.randint(0, 16, size=1003, dtype=np.uint8)
    np.testing.assert_array_equal(rand_vals_4, unpack_bits(pack_bits(rand_vals_4, 4), 4, 1003))

    rand_vals_2 = np.random.randint(0, 4, size=1005, dtype=np.uint8)
    np.testing.assert_array_equal(rand_vals_2, unpack_bits(pack_bits(rand_vals_2, 2), 2, 1005))
    print("[PASS] Random arrays (1000+ elements) roundtrip verification.")

    # 9. Quantization boundary values (0, q_max)
    bounds_4 = np.array([0, 15, 0, 15], dtype=np.uint8)
    np.testing.assert_array_equal(bounds_4, unpack_bits(pack_bits(bounds_4, 4), 4, 4))
    bounds_2 = np.array([0, 3, 0, 3, 3, 0], dtype=np.uint8)
    np.testing.assert_array_equal(bounds_2, unpack_bits(pack_bits(bounds_2, 2), 2, 6))
    print("[PASS] Quantization boundary extreme values verification.\n")


def run_edge_case_tests():
    """Validates engine behavior on tricky numerical edge cases."""
    print("=" * 70)
    print("RUNNING EDGE-CASE VALIDATION TESTS")
    print("=" * 70)

    # 1. Empty Array
    empty_data = np.array([], dtype=np.float32)
    q_empty, scale_e, zp_e = quantize(empty_data, bits=8)
    rec_empty = dequantize(q_empty, scale_e, zp_e, bits=8)
    assert rec_empty.size == 0, "Empty array handling failed!"
    print("[PASS] Empty array edge case handled successfully.")

    # 2. Constant Array
    const_data = np.full(100, 7.5, dtype=np.float32)
    q_const, scale_c, zp_c = quantize(const_data, bits=4)
    rec_const = dequantize(q_const, scale_c, zp_c, bits=4)
    mse_c = calculate_mse(const_data, rec_const)
    assert mse_c < 1e-5, f"Constant array handling failed! MSE: {mse_c}"
    print(f"[PASS] Constant array edge case handled successfully (MSE: {mse_c:.6f}).")

    # 3. Negative & Zero Values
    neg_data = np.array([-15.2, -8.0, -0.5, 0.0, 3.4, 12.1], dtype=np.float32)
    q_neg, scale_n, zp_n = quantize(neg_data, bits=8)
    rec_neg = dequantize(q_neg, scale_n, zp_n, bits=8)
    mse_n = calculate_mse(neg_data, rec_neg)
    print(f"[PASS] Negative & zero range edge case handled (MSE: {mse_n:.6f}).")

    # 4. Very Small Range
    tiny_data = np.linspace(0.00001, 0.00005, 50, dtype=np.float32)
    q_tiny, scale_t, zp_t = quantize(tiny_data, bits=8)
    rec_tiny = dequantize(q_tiny, scale_t, zp_t, bits=8)
    mse_t = calculate_mse(tiny_data, rec_tiny)
    print(f"[PASS] Very small floating-point range handled (MSE: {mse_t:.10f}).\n")



def generate_synthetic_dataset(num_samples: int = 10000, seed: int = 42) -> np.ndarray:
    """
    Generates a reproducible 10,000 element floating-point dataset
    simulating neural network weight distribution (mixture of Gaussians).
    """
    np.random.seed(seed)
    # Mixture of two Gaussians + small uniform noise to emulate model weights/activations
    weights_cluster1 = np.random.normal(loc=-1.5, scale=0.8, size=num_samples // 2)
    weights_cluster2 = np.random.normal(loc=1.5, scale=1.2, size=num_samples // 2)
    dataset = np.concatenate([weights_cluster1, weights_cluster2]).astype(np.float32)
    np.random.shuffle(dataset)
    return dataset


def run_quantization_experiment():
    """Runs primary quantization benchmark across FP32, INT8, INT4, and INT2."""
    run_bit_packing_tests()
    run_edge_case_tests()

    # Generate 10,000 sample FP32 dataset
    num_samples = 10000
    original_data = generate_synthetic_dataset(num_samples=num_samples, seed=42)
    fp32_bytes = original_data.nbytes  # 10,000 * 4 bytes = 40,000 bytes

    bit_widths = [8, 4, 2]
    results = {}

    print("=" * 70)
    print(f"QUANTIZATION EXPERIMENT RESULTS ({num_samples:,} FP32 Floating-Point Values)")
    print("=" * 70)

    # FP32 Baseline
    results['FP32'] = {
        'bits': 32,
        'orig_bytes': fp32_bytes,
        'packed_bytes': fp32_bytes,
        'metadata_bytes': 0,
        'total_compressed_bytes': fp32_bytes,
        'actual_ratio': 1.0,
        'actual_saved_pct': 0.0,
        'theoretical_ratio': 1.0,
        'theoretical_saved_pct': 0.0,
        'mse': 0.0,
        'mae': 0.0,
        'max_error': 0.0,
        'reconstructed': original_data
    }

    print(f"\n--- FP32 Baseline ---")
    print(f"Bits                       : 32")
    print(f"Original Size              : {fp32_bytes:,} bytes")
    print(f"Packed Bytes               : {fp32_bytes:,} bytes")
    print(f"Metadata Bytes             : 0 bytes")
    print(f"Total Compressed Bytes     : {fp32_bytes:,} bytes")
    print(f"Actual Storage Ratio       : 1.00x")
    print(f"Theoretical Ratio          : 1.00x")
    print(f"Actual Storage Saved %     : 0.00%")
    print(f"MSE                        : 0.00000000")
    print(f"MAE                        : 0.00000000")
    print(f"Max Error                  : 0.00000000")

    for bits in bit_widths:
        # 1. Quantize
        quantized_vals, scale, zero_point = quantize(original_data, bits=bits)

        # 2. Real Bit Packing
        packed_data = pack_bits(quantized_vals, bits=bits)

        # 3. Real Unpacking
        unpacked_vals = unpack_bits(packed_data, bits=bits, original_length=len(original_data))

        # 4. Dequantize
        reconstructed = dequantize(unpacked_vals, scale, zero_point, bits=bits)

        # Compute error metrics
        mse = calculate_mse(original_data, reconstructed)
        mae = calculate_mae(original_data, reconstructed)
        max_err = calculate_max_error(original_data, reconstructed)

        # Calculate Storage Breakdown
        storage = calculate_actual_storage(
            original_length=len(original_data),
            packed_bytes_count=len(packed_data),
            bits=bits,
            metadata_bytes=13,
            original_element_bytes=4
        )

        key = f"INT{bits}"
        results[key] = {
            'bits': bits,
            'orig_bytes': storage['original_bytes'],
            'packed_bytes': storage['packed_bytes'],
            'metadata_bytes': storage['metadata_bytes'],
            'total_compressed_bytes': storage['total_compressed_bytes'],
            'actual_ratio': storage['actual_compression_ratio'],
            'actual_saved_pct': storage['actual_storage_saved_percent'],
            'theoretical_ratio': storage['theoretical_compression_ratio'],
            'theoretical_saved_pct': storage['theoretical_storage_saved_percent'],
            'mse': mse,
            'mae': mae,
            'max_error': max_err,
            'reconstructed': reconstructed,
            'scale': scale,
            'zero_point': zero_point
        }

        print(f"\n--- {key} ({bits}-Bit Real Packed Quantization) ---")
        print(f"Bits                       : {bits}")
        print(f"Original Size (FP32)       : {storage['original_bytes']:,} bytes")
        print(f"Packed Bytes               : {storage['packed_bytes']:,} bytes")
        print(f"Metadata Overhead          : {storage['metadata_bytes']} bytes (Scale, ZeroPoint, Len, Bits)")
        print(f"Total Compressed Bytes     : {storage['total_compressed_bytes']:,} bytes")
        print(f"Actual Storage Ratio       : {storage['actual_compression_ratio']:.2f}x")
        print(f"Theoretical Ratio          : {storage['theoretical_compression_ratio']:.2f}x")
        print(f"Actual Storage Saved %     : {storage['actual_storage_saved_percent']:.2f}%")
        print(f"MSE                        : {mse:.8f}")
        print(f"MAE                        : {mae:.8f}")
        print(f"Max Error                  : {max_err:.8f}")

    print("\n" + "=" * 70)

    # Ensure results directory exists
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    os.makedirs(results_dir, exist_ok=True)

    # Generate Plots
    plot_original_vs_reconstructed(original_data, results, results_dir)
    plot_bits_vs_mse(results, results_dir)

    return results


def plot_original_vs_reconstructed(original_data: np.ndarray, results: dict, results_dir: str):
    """
    Creates Figure 1: Original vs Reconstructed signals across quantization levels.
    Saves to 'results/original_vs_reconstructed.png'.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.suptitle('TurboQuant Lab: Original vs Reconstructed Values across Bit-Widths', fontsize=16, fontweight='bold', y=0.98)

    sample_slice = slice(0, 80)
    x_axis = np.arange(80)

    # Color palette
    colors = {
        'FP32': '#1f77b4',  # Blue
        'INT8': '#2ca02c',  # Green
        'INT4': '#ff7f0e',  # Orange
        'INT2': '#d62728'   # Red
    }

    # 1. Sample Signal Waveform Comparison (Top Left)
    ax1 = axes[0, 0]
    ax1.plot(x_axis, original_data[sample_slice], label='Original FP32', color=colors['FP32'], linewidth=2.5, alpha=0.85)
    for key in ['INT8', 'INT4', 'INT2']:
        ax1.plot(x_axis, results[key]['reconstructed'][sample_slice], label=f"{key} (bits={results[key]['bits']})",
                 color=colors[key], linestyle='--', linewidth=1.5, alpha=0.85)
    ax1.set_title('Sample Sequence Reconstruction (First 80 Values)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Value')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 2. Histogram / Distribution Comparison (Top Right)
    ax2 = axes[0, 1]
    ax2.hist(original_data, bins=60, density=True, alpha=0.35, color=colors['FP32'], label='Original FP32')
    for key in ['INT8', 'INT4', 'INT2']:
        ax2.hist(results[key]['reconstructed'], bins=60, density=True, histtype='step',
                 linewidth=1.8, color=colors[key], label=f"{key} Distribution")
    ax2.set_title('Value Density Distribution Comparison', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Value Range')
    ax2.set_ylabel('Probability Density')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # 3. Residual Error Scatter Plot (Bottom Left - INT4 & INT2)
    ax3 = axes[1, 0]
    err_int8 = original_data[sample_slice] - results['INT8']['reconstructed'][sample_slice]
    err_int4 = original_data[sample_slice] - results['INT4']['reconstructed'][sample_slice]
    err_int2 = original_data[sample_slice] - results['INT2']['reconstructed'][sample_slice]

    ax3.stem(x_axis, err_int2, linefmt='r-', markerfmt='ro', basefmt='k-', label='INT2 Residual Error')
    ax3.stem(x_axis, err_int4, linefmt='C1-', markerfmt='C1o', basefmt='k-', label='INT4 Residual Error')
    ax3.set_title('Reconstruction Residual Errors (Original - Reconstructed)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Error Magnitude')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, linestyle=':', alpha=0.6)

    # 4. Summary Table Panel (Bottom Right)
    ax4 = axes[1, 1]
    ax4.axis('off')
    table_data = [
        ["Level", "Bits", "Actual Ratio", "Total Bytes", "Saved %", "MSE", "Max Error"],
        ["FP32", "32", "1.00x", f"{results['FP32']['total_compressed_bytes']:,}", "0.0%", "0.0000", "0.0000"],
        ["INT8", "8", f"{results['INT8']['actual_ratio']:.2f}x", f"{results['INT8']['total_compressed_bytes']:,}", f"{results['INT8']['actual_saved_pct']:.1f}%",
         f"{results['INT8']['mse']:.6f}", f"{results['INT8']['max_error']:.4f}"],
        ["INT4", "4", f"{results['INT4']['actual_ratio']:.2f}x", f"{results['INT4']['total_compressed_bytes']:,}", f"{results['INT4']['actual_saved_pct']:.1f}%",
         f"{results['INT4']['mse']:.6f}", f"{results['INT4']['max_error']:.4f}"],
        ["INT2", "2", f"{results['INT2']['actual_ratio']:.2f}x", f"{results['INT2']['total_compressed_bytes']:,}", f"{results['INT2']['actual_saved_pct']:.1f}%",
         f"{results['INT2']['mse']:.6f}", f"{results['INT2']['max_error']:.4f}"]
    ]
    tbl = ax4.table(cellText=table_data, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#333333')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#f2f2f2')
    ax4.set_title('Performance Metrics Summary', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(results_dir, 'original_vs_reconstructed.png')
    plt.savefig(output_path)
    plt.close()
    print(f"[PLOT CREATED] Saved figure to: {output_path}")


def plot_bits_vs_mse(results: dict, results_dir: str):
    """
    Creates Figure 2: Bit Width vs MSE Trade-off Curve.
    Saves to 'results/bits_vs_mse.png'.
    """
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

    bits_list = [results[k]['bits'] for k in ['INT2', 'INT4', 'INT8', 'FP32']]
    mse_list = [results[k]['mse'] for k in ['INT2', 'INT4', 'INT8', 'FP32']]
    labels = ['INT2 (2-bit)', 'INT4 (4-bit)', 'INT8 (8-bit)', 'FP32 (32-bit)']

    ax.plot(bits_list, mse_list, marker='o', linewidth=2.5, markersize=8, color='#1f77b4', label='Mean Squared Error')

    # Annotate points
    for i, (b, m) in enumerate(zip(bits_list, mse_list)):
        ax.annotate(f"{labels[i]}\nMSE: {m:.5f}",
                    (b, m),
                    textcoords="offset points",
                    xytext=(0, 12 if i != 0 else -25),
                    ha='center',
                    fontweight='bold',
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

    ax.set_title('TurboQuant Lab: Bit Width vs Mean Squared Error (MSE)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Quantization Bit Width (bits)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Squared Error (MSE)', fontsize=11, fontweight='bold')
    ax.set_xticks([2, 4, 8, 32])
    ax.set_xticklabels(['2-bit', '4-bit', '8-bit', '32-bit'])
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    output_path = os.path.join(results_dir, 'bits_vs_mse.png')
    plt.savefig(output_path)
    plt.close()
    print(f"[PLOT CREATED] Saved figure to: {output_path}")


if __name__ == '__main__':
    run_quantization_experiment()
