"""
Bit Packing and Unpacking Module for TurboQuant Lab.

Implements manual bit-level packing and unpacking operations using NumPy:
- 8-bit: 1 value per uint8 byte.
- 4-bit: 2 values packed per uint8 byte ((a << 4) | b).
- 2-bit: 4 values packed per uint8 byte ((a << 6) | (b << 4) | (c << 2) | d).

Handles arbitrary array lengths with zero-padding in the final byte and restores
the exact original element count upon unpacking.
"""

from typing import Union
import numpy as np


def pack_bits(values: np.ndarray, bits: int) -> np.ndarray:
    """
    Packs discrete integer quantization values into a compact byte array (np.uint8).

    Parameters:
    -----------
    values : np.ndarray
        Array of integer quantization levels (0 <= v <= 2^bits - 1).
    bits : int
        Bit width per element (8, 4, or 2).

    Returns:
    --------
    packed : np.ndarray
        Packed array of uint8 bytes.
    """
    if bits not in (8, 4, 2):
        raise ValueError(f"Unsupported bit width for packing: {bits}. Choose 8, 4, or 2.")

    if values.size == 0:
        return np.array([], dtype=np.uint8)

    # Ensure values are integer type within valid range
    q_max = (1 << bits) - 1
    clean_values = np.clip(values, 0, q_max).astype(np.uint8).flatten()
    n = clean_values.size

    if bits == 8:
        # 1 value per byte
        return clean_values.copy()

    elif bits == 4:
        # 2 values per byte
        num_packed = (n + 1) // 2
        packed = np.zeros(num_packed, dtype=np.uint8)

        # Pad with 0 if length is odd
        if n % 2 != 0:
            padded_values = np.pad(clean_values, (0, 1), mode='constant', constant_values=0)
        else:
            padded_values = clean_values

        val_high = padded_values[0::2] << 4
        val_low = padded_values[1::2]
        packed = (val_high | val_low).astype(np.uint8)
        return packed

    elif bits == 2:
        # 4 values per byte
        num_packed = (n + 3) // 4
        remainder = n % 4

        if remainder != 0:
            pad_len = 4 - remainder
            padded_values = np.pad(clean_values, (0, pad_len), mode='constant', constant_values=0)
        else:
            padded_values = clean_values

        v0 = padded_values[0::4] << 6
        v1 = padded_values[1::4] << 4
        v2 = padded_values[2::4] << 2
        v3 = padded_values[3::4]

        packed = (v0 | v1 | v2 | v3).astype(np.uint8)
        return packed

    raise ValueError(f"Unsupported bit width: {bits}")


def unpack_bits(packed: np.ndarray, bits: int, original_length: int) -> np.ndarray:
    """
    Unpacks a compact byte array back into discrete integer quantization levels.

    Parameters:
    -----------
    packed : np.ndarray
        Array of packed uint8 bytes.
    bits : int
        Bit width per element (8, 4, or 2).
    original_length : int
        Exact number of elements to restore (trims padding).

    Returns:
    --------
    unpacked : np.ndarray
        Array of unpacked integer quantization levels (np.uint8).
    """
    if bits not in (8, 4, 2):
        raise ValueError(f"Unsupported bit width for unpacking: {bits}. Choose 8, 4, or 2.")

    if original_length == 0 or packed.size == 0:
        return np.array([], dtype=np.uint8)

    if bits == 8:
        return packed[:original_length].astype(np.uint8)

    elif bits == 4:
        # Extract high 4 bits and low 4 bits
        high = (packed >> 4) & 0x0F
        low = packed & 0x0F

        # Interleave high and low
        unpacked_all = np.empty(packed.size * 2, dtype=np.uint8)
        unpacked_all[0::2] = high
        unpacked_all[1::2] = low

        return unpacked_all[:original_length]

    elif bits == 2:
        # Extract 2-bit fields
        v0 = (packed >> 6) & 0x03
        v1 = (packed >> 4) & 0x03
        v2 = (packed >> 2) & 0x03
        v3 = packed & 0x03

        unpacked_all = np.empty(packed.size * 4, dtype=np.uint8)
        unpacked_all[0::4] = v0
        unpacked_all[1::4] = v1
        unpacked_all[2::4] = v2
        unpacked_all[3::4] = v3

        return unpacked_all[:original_length]

    raise ValueError(f"Unsupported bit width: {bits}")
