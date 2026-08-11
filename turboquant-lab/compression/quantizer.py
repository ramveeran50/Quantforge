"""
Uniform Scalar Quantization Engine for TurboQuant Lab.

This module implements manual uniform scalar quantization (affine scale and zero-point)
for compressing 32-bit floating point numerical data down to lower bit-widths (8-bit, 4-bit, 2-bit).

Mathematical Principles:
-----------------------
1. Quantization Range:
   For a given bit width 'b', the discrete integer quantization levels span [q_min, q_max]:
   - 8-bit: [0, 255]   (256 distinct levels)
   - 4-bit: [0, 15]    (16 distinct levels)
   - 2-bit: [0, 3]     (4 distinct levels)

2. Scale (S):
   Maps the continuous real range [x_min, x_max] to the integer level range [0, 2^b - 1]:
   Scale S = (x_max - x_min) / (q_max - q_min)

3. Zero-Point (Z):
   Represents the quantized integer level corresponding to real 0.0:
   Zero-Point Z = round(-x_min / S)
   Clamped to [q_min, q_max].

4. Quantization Mapping:
   q = clip(round(x / S) + Z, q_min, q_max)

5. Dequantization Reconstruction:
   x_hat = (q - Z) * S
"""

from typing import Tuple, Union, Dict
import numpy as np
from .bit_packing import pack_bits, unpack_bits


def calculate_actual_storage(
    original_length: int,
    packed_bytes_count: int,
    bits: int = 8,
    metadata_bytes: int = 13,
    original_element_bytes: int = 4,
) -> Dict[str, Union[int, float]]:
    """
    Calculates actual storage footprint including bit packing and metadata overhead.

    Metadata schema (13 bytes default):
    - Scale (float32): 4 bytes
    - Zero point (int32): 4 bytes
    - Original length (uint32): 4 bytes
    - Bit width (uint8): 1 byte

    Returns dict containing:
    - original_bytes
    - packed_bytes
    - metadata_bytes
    - total_compressed_bytes
    - actual_compression_ratio
    - actual_storage_saved_percent
    - theoretical_compression_ratio
    - theoretical_storage_saved_percent
    """
    original_bytes = original_length * original_element_bytes

    if original_length == 0:
        return {
            "original_bytes": 0,
            "packed_bytes": 0,
            "metadata_bytes": 0,
            "total_compressed_bytes": 0,
            "actual_compression_ratio": 1.0,
            "actual_storage_saved_percent": 0.0,
            "theoretical_compression_ratio": 32.0 / bits if bits > 0 else 1.0,
            "theoretical_storage_saved_percent": (1.0 - bits / 32.0) * 100.0 if bits > 0 else 0.0,
        }

    total_compressed_bytes = packed_bytes_count + metadata_bytes
    actual_compression_ratio = original_bytes / total_compressed_bytes if total_compressed_bytes > 0 else 1.0
    actual_storage_saved_percent = (1.0 - (total_compressed_bytes / original_bytes)) * 100.0

    theoretical_compression_ratio = 32.0 / bits
    theoretical_storage_saved_percent = (1.0 - (bits / 32.0)) * 100.0

    return {
        "original_bytes": original_bytes,
        "packed_bytes": packed_bytes_count,
        "metadata_bytes": metadata_bytes,
        "total_compressed_bytes": total_compressed_bytes,
        "actual_compression_ratio": float(actual_compression_ratio),
        "actual_storage_saved_percent": float(actual_storage_saved_percent),
        "theoretical_compression_ratio": float(theoretical_compression_ratio),
        "theoretical_storage_saved_percent": float(theoretical_storage_saved_percent),
    }



def quantize(
    data: np.ndarray, bits: int = 8
) -> Tuple[np.ndarray, float, int]:
    """
    Quantizes a floating-point NumPy array into discrete integer representation.

    Parameters:
    -----------
    data : np.ndarray
        Array of floating-point values to compress.
    bits : int
        Target bit width for quantization. Supported values: 8, 4, or 2.

    Returns:
    --------
    quantized_data : np.ndarray
        Array of quantized integer values (np.uint8).
    scale : float
        Quantization scale factor (step size per quantization level).
    zero_point : int
        Integer zero-point offset mapping real 0.0 to quantized integer scale.
    """
    if bits not in (8, 4, 2):
        raise ValueError(f"Unsupported bit width: {bits}. Choose from 8, 4, or 2.")

    # Edge Case 1: Empty Array
    if data.size == 0:
        return np.array([], dtype=np.uint8), 1.0, 0

    # Calculate integer range bounds based on bit width
    q_min = 0
    q_max = (1 << bits) - 1  # 2^bits - 1 (e.g. 255 for 8-bit, 15 for 4-bit, 3 for 2-bit)

    # Find minimum and maximum floating point values
    x_min = float(np.min(data))
    x_max = float(np.max(data))

    # Edge Case 2: Constant Array (min == max)
    if x_min == x_max or np.isclose(x_min, x_max):
        if x_min == 0.0:
            scale = 1.0
            zero_point = 0
            quantized = np.zeros_like(data, dtype=np.uint8)
        else:
            scale = abs(x_min) / float(q_max) if q_max > 0 else 1.0
            if scale == 0.0:
                scale = 1.0
            raw_zp = -x_min / scale
            zero_point = int(np.clip(np.round(raw_zp), q_min, q_max))
            scaled_data = data / scale
            quantized_float = np.round(scaled_data) + zero_point
            quantized = np.clip(quantized_float, q_min, q_max).astype(np.uint8)
        return quantized, scale, zero_point

    # Calculate Scale S: real value step size per integer bin
    scale = (x_max - x_min) / float(q_max - q_min)

    # Safeguard against extremely small scale ranges causing division overflow
    if scale == 0.0:
        scale = 1.0

    # Calculate Zero-Point Z: quantized level corresponding to float 0.0
    raw_zero_point = -x_min / scale
    zero_point = int(np.round(raw_zero_point))
    zero_point = int(np.clip(zero_point, q_min, q_max))

    # Map real continuous values 'x' to discrete integer levels 'q'
    # Formula: q = round(x / scale) + zero_point
    scaled_data = data / scale
    quantized_float = np.round(scaled_data) + zero_point

    # Clamp to quantization level bounds [0, q_max] and convert to unsigned byte
    quantized_data = np.clip(quantized_float, q_min, q_max).astype(np.uint8)

    return quantized_data, scale, zero_point


def dequantize(
    quantized_data: np.ndarray,
    scale: float,
    zero_point: int,
    bits: int = 8,
) -> np.ndarray:
    """
    Reconstructs approximate floating-point values from quantized integer data.

    Parameters:
    -----------
    quantized_data : np.ndarray
        Array of quantized integer values.
    scale : float
        Quantization scale factor.
    zero_point : int
        Integer zero-point offset.
    bits : int
        Bit width used during quantization (8, 4, or 2).

    Returns:
    --------
    reconstructed_data : np.ndarray
        Reconstructed floating-point array (float32).
    """
    if quantized_data.size == 0:
        return np.array([], dtype=np.float32)

    # Convert integer quantized array back to float range using affine formula:
    # x_reconstructed = (q - Z) * S
    reconstructed_data = (quantized_data.astype(np.float32) - float(zero_point)) * float(scale)

    return reconstructed_data


# =====================================================================
# Evaluation Metrics Functions
# =====================================================================

def calculate_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Calculates Mean Squared Error (MSE) between original and reconstructed arrays.
    MSE = mean((original - reconstructed)^2)
    """
    if original.size == 0 or reconstructed.size == 0:
        return 0.0
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    return float(np.mean(diff ** 2))


def calculate_mae(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Calculates Mean Absolute Error (MAE) between original and reconstructed arrays.
    MAE = mean(|original - reconstructed|)
    """
    if original.size == 0 or reconstructed.size == 0:
        return 0.0
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    return float(np.mean(np.abs(diff)))


def calculate_max_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Calculates Maximum Absolute Error between original and reconstructed arrays.
    Max Error = max(|original - reconstructed|)
    """
    if original.size == 0 or reconstructed.size == 0:
        return 0.0
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    return float(np.max(np.abs(diff)))


def calculate_compression_ratio(
    original_bits: int = 32, quantized_bits: int = 8
) -> Tuple[float, float]:
    """
    Calculates theoretical compression ratio and storage saved percentage.

    Parameters:
    -----------
    original_bits : int
        Original data precision in bits (default: 32 for FP32).
    quantized_bits : int
        Quantized data precision in bits (e.g., 8, 4, or 2).

    Returns:
    --------
    ratio : float
        Compression ratio multiplier (e.g. 4.0 for 32-bit -> 8-bit).
    saved_percent : float
        Percentage of memory storage saved (e.g. 75.0%).
    """
    if quantized_bits <= 0:
        raise ValueError("Quantized bits must be greater than 0.")

    ratio = original_bits / quantized_bits
    saved_percent = (1.0 - (quantized_bits / original_bits)) * 100.0

    return float(ratio), float(saved_percent)
