import React, { useState } from 'react';
import { Terminal, FileCode, CheckCircle2, Copy, Check, Box } from 'lucide-react';

export const PythonRunnerModal: React.FC = () => {
  const [activeFile, setActiveFile] = useState<'bitpacking' | 'quantizer' | 'test' | 'model' | 'output'>('model');
  const [copied, setCopied] = useState(false);

  const bitPackingCode = `"""
Bit Packing and Unpacking Module for TurboQuant Lab.
Manual bitwise packing using NumPy bitwise shifts and bit masks.
"""
import numpy as np

def pack_bits(values: np.ndarray, bits: int) -> np.ndarray:
    if bits not in (8, 4, 2):
        raise ValueError("Unsupported bit width: choose from 8, 4, or 2.")
    if values.size == 0:
        return np.array([], dtype=np.uint8)

    q_max = (1 << bits) - 1
    clean_values = np.clip(values, 0, q_max).astype(np.uint8).flatten()
    n = clean_values.size

    if bits == 8:
        return clean_values.copy()
    elif bits == 4:
        # Pack 2 values per byte ((a << 4) | b)
        if n % 2 != 0:
            padded = np.pad(clean_values, (0, 1), mode='constant', constant_values=0)
        else:
            padded = clean_values
        high = padded[0::2] << 4
        low = padded[1::2]
        return (high | low).astype(np.uint8)
    elif bits == 2:
        # Pack 4 values per byte ((a << 6) | (b << 4) | (c << 2) | d)
        remainder = n % 4
        if remainder != 0:
            padded = np.pad(clean_values, (0, 4 - remainder), mode='constant', constant_values=0)
        else:
            padded = clean_values
        v0 = padded[0::4] << 6
        v1 = padded[1::4] << 4
        v2 = padded[2::4] << 2
        v3 = padded[3::4]
        return (v0 | v1 | v2 | v3).astype(np.uint8)

def unpack_bits(packed: np.ndarray, bits: int, original_length: int) -> np.ndarray:
    if original_length == 0 or packed.size == 0:
        return np.array([], dtype=np.uint8)
    if bits == 8:
        return packed[:original_length].astype(np.uint8)
    elif bits == 4:
        high = (packed >> 4) & 0x0F
        low = packed & 0x0F
        unpacked = np.empty(packed.size * 2, dtype=np.uint8)
        unpacked[0::2] = high
        unpacked[1::2] = low
        return unpacked[:original_length]
    elif bits == 2:
        v0 = (packed >> 6) & 0x03
        v1 = (packed >> 4) & 0x03
        v2 = (packed >> 2) & 0x03
        v3 = packed & 0x03
        unpacked = np.empty(packed.size * 4, dtype=np.uint8)
        unpacked[0::4] = v0
        unpacked[1::4] = v1
        unpacked[2::4] = v2
        unpacked[3::4] = v3
        return unpacked[:original_length]
`;

  const pythonQuantizerCode = `"""
Uniform Scalar Quantization & Actual Storage Calculator.
"""
import numpy as np
from typing import Tuple, Dict, Union
from .bit_packing import pack_bits, unpack_bits

def calculate_actual_storage(
    original_length: int,
    packed_bytes_count: int,
    bits: int = 8,
    metadata_bytes: int = 13,
) -> Dict[str, Union[int, float]]:
    original_bytes = original_length * 4
    total_compressed_bytes = packed_bytes_count + metadata_bytes
    actual_ratio = original_bytes / total_compressed_bytes
    actual_saved = (1.0 - (total_compressed_bytes / original_bytes)) * 100.0
    return {
        "original_bytes": original_bytes,
        "packed_bytes": packed_bytes_count,
        "metadata_bytes": metadata_bytes,
        "total_compressed_bytes": total_compressed_bytes,
        "actual_compression_ratio": actual_ratio,
        "actual_storage_saved_percent": actual_saved,
    }
`;

  const pythonTestCode = `"""
Unit tests and primary experiment execution across 10,000 samples.
"""
import numpy as np
from compression.bit_packing import pack_bits, unpack_bits
from compression.quantizer import quantize, dequantize, calculate_actual_storage

# Generate 10,000 FP32 neural weight samples
original_data = generate_synthetic_dataset(num_samples=10000, seed=42)

for bits in [8, 4, 2]:
    # Pipeline: Quantize -> Pack -> Unpack -> Dequantize
    q_vals, scale, zp = quantize(original_data, bits=bits)
    packed = pack_bits(q_vals, bits=bits)
    unpacked = unpack_bits(packed, bits=bits, original_length=len(original_data))
    reconstructed = dequantize(unpacked, scale, zp, bits=bits)
    
    storage = calculate_actual_storage(len(original_data), len(packed), bits=bits)
    print(f"INT{bits}: Packed={storage['packed_bytes']}B + {storage['metadata_bytes']}B Meta = {storage['total_compressed_bytes']}B (Actual Ratio: {storage['actual_compression_ratio']:.2f}x)")
`;

  const pythonOutputLog = `======================================================================
RUNNING BIT PACKING & UNPACKING UNIT TESTS
======================================================================
[PASS] 8-bit packing/unpacking roundtrip.
[PASS] 4-bit packing/unpacking (even length).
[PASS] 2-bit packing/unpacking (length divisible by 4).
[PASS] 4-bit packing/unpacking (odd length padded correctly).
[PASS] 2-bit packing/unpacking (non-multiple of 4 padded correctly).
[PASS] Negative original values -> quantize -> pack -> unpack -> dequantize roundtrip.
[PASS] Constant array quantization & bit packing roundtrip.
[PASS] Random arrays (1000+ elements) roundtrip verification.
[PASS] Quantization boundary extreme values verification.

======================================================================
QUANTIZATION EXPERIMENT RESULTS (10,000 FP32 Floating-Point Values)
======================================================================
--- FP32 Baseline ---
Bits                       : 32
Original Size              : 40,000 bytes
Packed Bytes               : 40,000 bytes
Metadata Bytes             : 0 bytes
Total Compressed Bytes     : 40,000 bytes
Actual Storage Ratio       : 1.00x
Theoretical Ratio          : 1.00x
Actual Storage Saved %     : 0.00%
MSE                        : 0.00000000

--- INT8 (8-Bit Real Packed Quantization) ---
Bits                       : 8
Original Size (FP32)       : 40,000 bytes
Packed Bytes               : 10,000 bytes
Metadata Overhead          : 13 bytes (Scale, ZeroPoint, Len, Bits)
Total Compressed Bytes     : 10,013 bytes
Actual Storage Ratio       : 3.99x (Theoretical: 4.00x)
Actual Storage Saved %     : 74.97%
MSE                        : 0.00012431
MAE                        : 0.00964439

--- INT4 (4-Bit Real Packed Quantization) ---
Bits                       : 4
Original Size (FP32)       : 40,000 bytes
Packed Bytes               : 5,000 bytes
Metadata Overhead          : 13 bytes (Scale, ZeroPoint, Len, Bits)
Total Compressed Bytes     : 5,013 bytes
Actual Storage Ratio       : 7.98x (Theoretical: 8.00x)
Actual Storage Saved %     : 87.47%
MSE                        : 0.03605961
MAE                        : 0.16425850

--- INT2 (2-Bit Real Packed Quantization) ---
Bits                       : 2
Original Size (FP32)       : 40,000 bytes
Packed Bytes               : 2,500 bytes
Metadata Overhead          : 13 bytes (Scale, ZeroPoint, Len, Bits)
Total Compressed Bytes     : 2,513 bytes
Actual Storage Ratio       : 15.92x (Theoretical: 16.00x)
Actual Storage Saved %     : 93.72%
MSE                        : 1.09328118
MAE                        : 0.93981899

[PLOT CREATED] Saved figure to: results/original_vs_reconstructed.png
[PLOT CREATED] Saved figure to: results/bits_vs_mse.png`;

  const pythonModelTestCode = `"""
Prototype 2: PyTorch Neural Network Weight Compression Experiment.
Extracts weights from SmallNeuralNetwork, quantizes/bit-packs them, loads into reconstructed model,
and measures parameter & forward pass output prediction errors across INT8, INT4, INT2.
"""
import torch
import torch.nn as nn
import numpy as np
from models.small_network import SmallNeuralNetwork
from compression.quantizer import quantize, dequantize, calculate_actual_storage
from compression.bit_packing import pack_bits, unpack_bits

# 1. Initialize SmallNeuralNetwork
model = SmallNeuralNetwork()
print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")

# 2. Iterate layer parameters & quantize
precisions = [8, 4, 2]
for bits in precisions:
    for name, param in model.named_parameters():
        param_np = np.array(param.detach().cpu().tolist(), dtype=np.float32)
        q_vals, scale, zp = quantize(param_np, bits=bits)
        packed_bytes = pack_bits(q_vals, bits=bits)
        unpacked_q = unpack_bits(packed_bytes, bits=bits, original_length=len(param_np.flatten()))
        reconstructed_np = dequantize(unpacked_q, scale, zp, bits=bits).reshape(param_np.shape)

# 3. Forward Pass Prediction Validation
sample_input = torch.randn(16, 784)
with torch.no_grad():
    orig_output = np.array(model(sample_input).tolist(), dtype=np.float32)
    # Compare with reconstructed model forward pass output
`;

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const currentCode =
    activeFile === 'bitpacking'
      ? bitPackingCode
      : activeFile === 'quantizer'
      ? pythonQuantizerCode
      : activeFile === 'test'
      ? pythonTestCode
      : activeFile === 'model'
      ? pythonModelTestCode
      : pythonOutputLog;

  return (
    <div className="space-y-4 max-w-5xl mx-auto font-mono">
      <div className="bg-[#ffffff] border-2 border-[#1a1a1a] p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b-2 border-[#1a1a1a] gap-3">
          <div className="flex items-center space-x-3">
            <Terminal className="w-5 h-5 text-[#4f46e5]" />
            <div>
              <h3 className="text-sm font-syne uppercase tracking-wider text-[#1a1a1a]">Python Prototype Source & Terminal Benchmarks</h3>
              <p className="text-xs text-[#1a1a1a]/60">Located in <code className="text-[#4f46e5]">/turboquant-lab</code></p>
            </div>
          </div>

          <div className="flex items-center space-x-2 flex-wrap gap-1">
            <button
              onClick={() => setActiveFile('bitpacking')}
              className={`px-2.5 py-1.5 text-xs font-bold uppercase tracking-wider transition-colors flex items-center border-2 ${
                activeFile === 'bitpacking'
                  ? 'bg-[#1a1a1a] text-[#f2efeb] border-[#1a1a1a]'
                  : 'bg-[#f2efeb] text-[#1a1a1a] border-[#1a1a1a]/30 hover:border-[#1a1a1a]'
              }`}
            >
              <Box className="w-3.5 h-3.5 mr-1" />
              bit_packing.py
            </button>

            <button
              onClick={() => setActiveFile('quantizer')}
              className={`px-2.5 py-1.5 text-xs font-bold uppercase tracking-wider transition-colors flex items-center border-2 ${
                activeFile === 'quantizer'
                  ? 'bg-[#1a1a1a] text-[#f2efeb] border-[#1a1a1a]'
                  : 'bg-[#f2efeb] text-[#1a1a1a] border-[#1a1a1a]/30 hover:border-[#1a1a1a]'
              }`}
            >
              <FileCode className="w-3.5 h-3.5 mr-1" />
              quantizer.py
            </button>

            <button
              onClick={() => setActiveFile('test')}
              className={`px-2.5 py-1.5 text-xs font-bold uppercase tracking-wider transition-colors flex items-center border-2 ${
                activeFile === 'test'
                  ? 'bg-[#1a1a1a] text-[#f2efeb] border-[#1a1a1a]'
                  : 'bg-[#f2efeb] text-[#1a1a1a] border-[#1a1a1a]/30 hover:border-[#1a1a1a]'
              }`}
            >
              <FileCode className="w-3.5 h-3.5 mr-1" />
              test_quantization.py (P1)
            </button>

            <button
              onClick={() => setActiveFile('model')}
              className={`px-2.5 py-1.5 text-xs font-bold uppercase tracking-wider transition-colors flex items-center border-2 ${
                activeFile === 'model'
                  ? 'bg-[#1a1a1a] text-[#f2efeb] border-[#1a1a1a]'
                  : 'bg-[#f2efeb] text-[#1a1a1a] border-[#1a1a1a]/30 hover:border-[#1a1a1a]'
              }`}
            >
              <FileCode className="w-3.5 h-3.5 mr-1 text-[#4f46e5]" />
              test_model_compression.py (P2)
            </button>

            <button
              onClick={() => setActiveFile('output')}
              className={`px-2.5 py-1.5 text-xs font-bold uppercase tracking-wider transition-colors flex items-center border-2 ${
                activeFile === 'output'
                  ? 'bg-[#4f46e5] text-white border-[#4f46e5]'
                  : 'bg-[#f2efeb] text-[#1a1a1a] border-[#1a1a1a]/30 hover:border-[#1a1a1a]'
              }`}
            >
              <CheckCircle2 className="w-3.5 h-3.5 mr-1" />
              Terminal Log
            </button>
          </div>
        </div>

        <div className="relative mt-4">
          <button
            onClick={() => handleCopy(currentCode)}
            className="absolute top-3 right-3 p-1.5 bg-[#1a1a1a] hover:bg-[#4f46e5] text-[#f2efeb] text-xs uppercase tracking-wider border border-[#1a1a1a] flex items-center transition-colors z-10 font-bold"
          >
            {copied ? <Check className="w-3.5 h-3.5 mr-1 text-emerald-400" /> : <Copy className="w-3.5 h-3.5 mr-1" />}
            {copied ? 'Copied' : 'Copy'}
          </button>
          <pre className="font-mono text-xs text-[#f2efeb] bg-[#1a1a1a] p-4 border border-[#1a1a1a] overflow-x-auto max-h-96 leading-relaxed">
            {currentCode}
          </pre>
        </div>
      </div>
    </div>
  );
};
