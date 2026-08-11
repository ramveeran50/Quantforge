/**
 * Uniform Scalar Quantization & Real Bit-Packing Engine
 * (TypeScript Implementation for TurboQuant Lab UI)
 * 
 * Implements Affine Quantization with Bit Packing:
 * 1. Quantize: Float -> Discrete INT (0 to 2^bits - 1)
 * 2. Bit Pack:
 *    - 8-bit: 1 value per uint8 byte
 *    - 4-bit: 2 values packed per uint8 byte ((a << 4) | b)
 *    - 2-bit: 4 values packed per uint8 byte ((a << 6) | (b << 4) | (c << 2) | d)
 * 3. Bit Unpack: Restore exact original elements with zero-padding trimmed
 * 4. Dequantize: Reconstruct approximate float values
 */

export interface QuantizationMetrics {
  bits: number;
  qMin: number;
  qMax: number;
  scale: number;
  zeroPoint: number;
  quantizedData: number[];
  packedBytesData: Uint8Array;
  reconstructedData: number[];
  
  // Storage Metrics
  originalSizeBytes: number;
  packedSizeBytes: number;
  metadataSizeBytes: number;
  totalCompressedSizeBytes: number;
  
  // Actual storage ratio vs theoretical ratio
  actualCompressionRatio: number;
  actualStorageSavedPercent: number;
  theoreticalCompressionRatio: number;
  theoreticalStorageSavedPercent: number;

  // Legacy field support for component compatibility
  compressedSizeBytes: number;
  compressionRatio: number;
  storageSavedPercent: number;

  // Error Metrics
  mse: number;
  mae: number;
  maxError: number;
}

export function packBits(values: number[] | Uint8Array, bits: number): Uint8Array {
  const n = values.length;
  if (n === 0) return new Uint8Array(0);

  const qMax = Math.pow(2, bits) - 1;

  if (bits === 16) {
    const packed = new Uint8Array(n * 2);
    for (let i = 0; i < n; i++) {
      const v = Math.max(0, Math.min(65535, values[i]));
      packed[i * 2] = (v >> 8) & 0xff;
      packed[i * 2 + 1] = v & 0xff;
    }
    return packed;
  }

  if (bits === 8) {
    const packed = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      packed[i] = Math.max(0, Math.min(qMax, values[i])) & 0xff;
    }
    return packed;
  }

  if (bits === 4) {
    const numPacked = Math.ceil(n / 2);
    const packed = new Uint8Array(numPacked);
    for (let i = 0; i < numPacked; i++) {
      const idx1 = i * 2;
      const idx2 = i * 2 + 1;
      const v1 = Math.max(0, Math.min(qMax, values[idx1])) & 0x0f;
      const v2 = idx2 < n ? Math.max(0, Math.min(qMax, values[idx2])) & 0x0f : 0;
      packed[i] = (v1 << 4) | v2;
    }
    return packed;
  }

  if (bits === 2) {
    const numPacked = Math.ceil(n / 4);
    const packed = new Uint8Array(numPacked);
    for (let i = 0; i < numPacked; i++) {
      const idx0 = i * 4;
      const idx1 = i * 4 + 1;
      const idx2 = i * 4 + 2;
      const idx3 = i * 4 + 3;

      const v0 = Math.max(0, Math.min(qMax, values[idx0])) & 0x03;
      const v1 = idx1 < n ? Math.max(0, Math.min(qMax, values[idx1])) & 0x03 : 0;
      const v2 = idx2 < n ? Math.max(0, Math.min(qMax, values[idx2])) & 0x03 : 0;
      const v3 = idx3 < n ? Math.max(0, Math.min(qMax, values[idx3])) & 0x03 : 0;

      packed[i] = (v0 << 6) | (v1 << 4) | (v2 << 2) | v3;
    }
    return packed;
  }

  throw new Error(`Unsupported bit width for packing: ${bits}`);
}

export function unpackBits(packed: Uint8Array, bits: number, originalLength: number): Uint8Array | Uint16Array | Uint32Array {
  if (originalLength === 0 || packed.length === 0) return new Uint8Array(0);

  if (bits === 16) {
    const unpacked = new Uint16Array(originalLength);
    for (let i = 0; i < originalLength; i++) {
      const high = packed[i * 2];
      const low = packed[i * 2 + 1];
      unpacked[i] = (high << 8) | low;
    }
    return unpacked;
  }

  if (bits === 8) {
    return packed.slice(0, originalLength);
  }

  if (bits === 4) {
    const unpacked = new Uint8Array(originalLength);
    for (let i = 0; i < originalLength; i++) {
      const byteIdx = Math.floor(i / 2);
      const isHigh = i % 2 === 0;
      const byteVal = packed[byteIdx];
      unpacked[i] = isHigh ? (byteVal >> 4) & 0x0f : byteVal & 0x0f;
    }
    return unpacked;
  }

  if (bits === 2) {
    const unpacked = new Uint8Array(originalLength);
    for (let i = 0; i < originalLength; i++) {
      const byteIdx = Math.floor(i / 4);
      const pos = i % 4; // 0, 1, 2, 3
      const byteVal = packed[byteIdx];
      const shift = (3 - pos) * 2; // 6, 4, 2, 0
      unpacked[i] = (byteVal >> shift) & 0x03;
    }
    return unpacked;
  }

  throw new Error(`Unsupported bit width for unpacking: ${bits}`);
}

export function quantizeAndEvaluate(data: number[], bits: number): QuantizationMetrics {
  const n = data.length;
  const originalSizeBytes = n * 4; // FP32 uses 4 bytes per float

  if (n === 0) {
    return {
      bits,
      qMin: 0,
      qMax: Math.pow(2, bits) - 1,
      scale: 1.0,
      zeroPoint: 0,
      quantizedData: [],
      packedBytesData: new Uint8Array(0),
      reconstructedData: [],
      originalSizeBytes: 0,
      packedSizeBytes: 0,
      metadataSizeBytes: 0,
      totalCompressedSizeBytes: 0,
      actualCompressionRatio: 1.0,
      actualStorageSavedPercent: 0,
      theoreticalCompressionRatio: 32 / bits,
      theoreticalStorageSavedPercent: (1 - bits / 32) * 100,
      compressedSizeBytes: 0,
      compressionRatio: 1.0,
      storageSavedPercent: 0,
      mse: 0,
      mae: 0,
      maxError: 0,
    };
  }

  const qMin = 0;
  const qMax = Math.pow(2, bits) - 1;

  let xMin = Math.min(...data);
  let xMax = Math.max(...data);

  let scale: number;
  let zeroPoint: number;
  let rawQuantized: number[] = new Array(n);

  if (xMin === xMax || Math.abs(xMax - xMin) < 1e-12) {
    if (xMin === 0) {
      scale = 1.0;
      zeroPoint = 0;
      rawQuantized.fill(0);
    } else {
      scale = Math.abs(xMin) / (qMax || 1);
      if (scale === 0) scale = 1.0;
      const rawZp = -xMin / scale;
      zeroPoint = Math.max(qMin, Math.min(qMax, Math.round(rawZp)));
      for (let i = 0; i < n; i++) {
        const qFloat = Math.round(data[i] / scale) + zeroPoint;
        rawQuantized[i] = Math.max(qMin, Math.min(qMax, qFloat));
      }
    }
  } else {
    scale = (xMax - xMin) / (qMax - qMin);
    if (scale === 0) scale = 1.0;

    const rawZp = -xMin / scale;
    zeroPoint = Math.max(qMin, Math.min(qMax, Math.round(rawZp)));

    for (let i = 0; i < n; i++) {
      const qFloat = Math.round(data[i] / scale) + zeroPoint;
      rawQuantized[i] = Math.max(qMin, Math.min(qMax, qFloat));
    }
  }

  // Step 2: Real Bit Packing
  const packedBytesData = packBits(rawQuantized, bits);

  // Step 3: Real Unpacking
  const unpackedQuantized = unpackBits(packedBytesData, bits, n);

  // Step 4: Dequantization
  const reconstructedData: number[] = new Array(n);
  for (let i = 0; i < n; i++) {
    reconstructedData[i] = (unpackedQuantized[i] - zeroPoint) * scale;
  }

  // Storage Breakdown
  const packedSizeBytes = packedBytesData.length;
  const metadataSizeBytes = 13; // Scale (4B) + ZeroPoint (4B) + OriginalLength (4B) + BitWidth (1B)
  const totalCompressedSizeBytes = packedSizeBytes + metadataSizeBytes;

  const actualCompressionRatio = originalSizeBytes / totalCompressedSizeBytes;
  const actualStorageSavedPercent = (1 - totalCompressedSizeBytes / originalSizeBytes) * 100;

  const theoreticalCompressionRatio = 32 / bits;
  const theoreticalStorageSavedPercent = (1 - bits / 32) * 100;

  // Calculate Errors
  let sumSqError = 0;
  let sumAbsError = 0;
  let maxError = 0;

  for (let i = 0; i < n; i++) {
    const err = Math.abs(data[i] - reconstructedData[i]);
    sumSqError += err * err;
    sumAbsError += err;
    if (err > maxError) {
      maxError = err;
    }
  }

  const mse = sumSqError / n;
  const mae = sumAbsError / n;

  return {
    bits,
    qMin,
    qMax,
    scale,
    zeroPoint,
    quantizedData: Array.from(unpackedQuantized),
    packedBytesData,
    reconstructedData,

    originalSizeBytes,
    packedSizeBytes,
    metadataSizeBytes,
    totalCompressedSizeBytes,

    actualCompressionRatio,
    actualStorageSavedPercent,
    theoreticalCompressionRatio,
    theoreticalStorageSavedPercent,

    // Backward compatibility
    compressedSizeBytes: totalCompressedSizeBytes,
    compressionRatio: actualCompressionRatio,
    storageSavedPercent: actualStorageSavedPercent,

    mse,
    mae,
    maxError,
  };
}

// Dataset Generators
export function generateSyntheticDataset(
  type: 'neural' | 'sine' | 'sparse' | 'uniform',
  samples: number = 200,
  seed: number = 42
): number[] {
  const result: number[] = [];
  
  let s = seed;
  const random = () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };

  const gaussian = (mean = 0, stdev = 1) => {
    const u = 1 - random();
    const v = random();
    const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    return z * stdev + mean;
  };

  if (type === 'neural') {
    for (let i = 0; i < samples; i++) {
      if (random() > 0.5) {
        result.push(gaussian(-1.5, 0.8));
      } else {
        result.push(gaussian(1.5, 1.2));
      }
    }
  } else if (type === 'sine') {
    for (let i = 0; i < samples; i++) {
      const t = (i / samples) * Math.PI * 6;
      result.push(Math.sin(t) * 2.5 + Math.cos(t * 2) * 0.8 + (random() - 0.5) * 0.2);
    }
  } else if (type === 'sparse') {
    for (let i = 0; i < samples; i++) {
      if (random() < 0.05) {
        result.push((random() > 0.5 ? 1 : -1) * (15 + random() * 10));
      } else {
        result.push((random() - 0.5) * 1.5);
      }
    }
  } else {
    for (let i = 0; i < samples; i++) {
      result.push((random() - 0.5) * 10);
    }
  }

  return result;
}
