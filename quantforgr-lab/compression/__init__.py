"""
Compression Package for TurboQuant Lab.
Provides uniform scalar quantization and evaluation metrics.
"""

from .bit_packing import pack_bits, unpack_bits
from .quantizer import (
    quantize,
    dequantize,
    calculate_mse,
    calculate_mae,
    calculate_max_error,
    calculate_compression_ratio,
    calculate_actual_storage,
)

__all__ = [
    "quantize",
    "dequantize",
    "pack_bits",
    "unpack_bits",
    "calculate_mse",
    "calculate_mae",
    "calculate_max_error",
    "calculate_compression_ratio",
    "calculate_actual_storage",
]

