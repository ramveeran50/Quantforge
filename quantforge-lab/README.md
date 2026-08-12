# QuantForge — Neural Network Quantization & Compression Research Prototype

A simple, educational Python research prototype demonstrating how continuous floating-point numerical data (such as neural network weights and activations) can be compressed using **uniform scalar quantization**, packed into low-bit byte representations (8-bit, 4-bit, 2-bit), reconstructed, and evaluated for actual memory savings and information loss.

## Disclaimer

QuantForge is an independent research and learning prototype for
low-bit neural network quantization and compression.

It is not an implementation of Google's TurboQuant and is not
affiliated with Google.

---

## Table of Contents

1. [Overview](#1-what-is-quantization)
2. [Why AI Models Can Be Compressed](#2-why-ai-models-can-be-compressed)
3. [How the Implemented Algorithm Works](#3-how-the-implemented-algorithm-works)
4. [Bit Packing Architecture (8-Bit, 4-Bit, 2-Bit)](#4-bit-packing-architecture-8-bit-4-bit-2-bit)
5. [Actual Storage Ratio vs Theoretical Ratio](#5-actual-storage-ratio-vs-theoretical-ratio)
6. [Why Quantization Introduces Approximation Error](#6-why-quantization-introduces-approximation-error)
7. [Difference from Google's TurboQuant](#7-difference-from-googles-turboquant)
8. [Project Structure](#project-structure)
9. [Prototype 2 — Neural Network Weight Compression](#9-prototype-2--neural-network-weight-compression)
10. [Getting Started & Running Experiments](#getting-started--running-experiments)

---

## 1. What is Quantization?

**Quantization** is the process of mapping a continuous (or high-precision discrete) range of real numbers to a smaller, finite set of discrete integer levels.

In standard computing, floating-point numbers are typically stored using 32 bits (IEEE 754 standard `float32`) or 16 bits (`float16` / `bfloat16`). Quantization discretizes these continuous values into low-bit integer representations (e.g., 8-bit `uint8`, 4-bit, or 2-bit integers), drastically reducing memory bandwidth and storage requirements.

---

## 2. Why AI Models Can Be Compressed

Modern Deep Learning models (like Large Language Models and Vision Transformers) contain billions of parameters (weights). Storing and querying these weights during inference requires immense RAM and GPU memory bandwidth.

AI models can be compressed heavily without severe loss of accuracy because:

1. **Redundancy & Overparameterization**: Neural networks learn distributed representations where parameters exhibit high variance margin tolerance.
2. **Robustness to Noise**: Deep networks are naturally resilient to small perturbations in weight values.
3. **Clustered Weight Distributions**: Model weights and activations often follow Gaussian or bell-curve distributions centered around zero, making uniform or non-uniform binning highly effective.

---

## 3. How the Implemented Algorithm Works

This prototype implements **Uniform Asymmetric Scalar Quantization** (Affine Quantization) and **Bit Packing** manually using NumPy without relying on external quantization frameworks.

### Mathematical Pipeline

Given an array of floating-point numbers $x \in \mathbb{R}^N$ and a target bit-width $b \in \{8, 4, 2\}$:

1. **Determine Discrete Level Range $[q_{\min}, q_{\max}]$**:
   $$q_{\min} = 0, \quad q_{\max} = 2^b - 1$$
   * For 8-bit ($b=8$): Levels span $[0, 255]$
   * For 4-bit ($b=4$): Levels span $[0, 15]$
   * For 2-bit ($b=2$): Levels span $[0, 3]$

2. **Compute Scale ($S$)**:
   The scale $S$ represents the real floating-point step size corresponding to one integer bin increment:
   $$S = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}}$$

3. **Compute Zero-Point ($Z$)**:
   The zero-point $Z$ is the quantized integer that corresponds exactly to real value $0.0$:
   $$Z = \text{clamp}\left( \text{round}\left( \frac{-x_{\min}}{S} \right), q_{\min}, q_{\max} \right)$$

4. **Quantization Mapping ($x \to q$)**:
   Transform floating-point values $x$ into low-precision unsigned integer values $q$:
   $$q = \text{clamp}\left( \text{round}\left( \frac{x}{S} \right) + Z, q_{\min}, q_{\max} \right)$$

---

## 4. Bit Packing Architecture (8-Bit, 4-Bit, 2-Bit)

In memory, standard bytes store 8 bits. Storing a 4-bit integer inside an 8-bit array element wastes 4 bits per value. To realize true hardware storage compression, values are packed bitwise using manual bit shifts:

* **8-Bit**: 1 value per byte (`uint8`).
* **4-Bit**: 2 values per byte using `(a << 4) | b`.
* **2-Bit**: 4 values per byte using `(a << 6) | (b << 4) | (c << 2) | d`.

Arrays whose length is not divisible by the packing factor (2 for 4-bit, 4 for 2-bit) are zero-padded in the final byte during packing and stripped during unpacking using the original vector length $N$.

---

## 5. Actual Storage Ratio vs Theoretical Ratio

Theoretical compression ratio ignores overheads and container structures:
$$\text{Theoretical Ratio} = \frac{32}{b}$$

Actual storage ratio accounts for packed payload bytes plus 13 bytes of mandatory tensor metadata (Scale [4B], Zero-Point [4B], Original Length [4B], Bit Width [1B]):
$$\text{Actual Storage Ratio} = \frac{N \times 4}{\text{Packed Bytes} + 13}$$

For a 10,000 FP32 value tensor (40,000 bytes original size):
* **INT8**: 10,000 packed bytes + 13B metadata = 10,013B &rarr; **3.99&times; Actual Ratio** (4.00&times; theoretical)
* **INT4**: 5,000 packed bytes + 13B metadata = 5,013B &rarr; **7.98&times; Actual Ratio** (8.00&times; theoretical)
* **INT2**: 2,500 packed bytes + 13B metadata = 2,513B &rarr; **15.92&times; Actual Ratio** (16.00&times; theoretical)

---

## 6. Why Quantization Introduces Approximation Error

Quantization collapses an infinite continuum of real values into a tiny set of discrete steps ($2^b$). Rounding difference produces reconstruction error:
$$\epsilon = x - \hat{x}$$

Evaluated metrics:
1. **Mean Squared Error (MSE)**: $\frac{1}{N} \sum_{i=1}^N (x_i - \hat{x}_i)^2$
2. **Mean Absolute Error (MAE)**: $\frac{1}{N} \sum_{i=1}^N |x_i - \hat{x}_i|$
3. **Maximum Absolute Error**: $\max_i |x_i - \hat{x}_i|$

---

## 7. Difference from Google's TurboQuant

This laboratory prototype is a basic **uniform scalar quantizer** created for educational research. It differs fundamentally from Google's TurboQuant:

1. **Scalar vs Vector/Tensor Quantization**: This lab quantizes scalar numbers individually. Google's TurboQuant operates on multi-dimensional vectors and sub-block codebooks.
2. **Outlier Isolation**: Production TurboQuant isolates activation spikes into high-precision channels to prevent precision loss in transformer attention.
3. **Adaptive Non-Uniform Grids**: TurboQuant uses non-linear quantization grids optimized for weight bell curves.

---

## Project Structure

```text
QuantForge-lab/
│
├── compression/
│   ├── __init__.py
│   ├── bit_packing.py
│   └── quantizer.py
│
├── experiments/
│   ├── test_model_compression.py
│   └── test_quantization.py
│
├── models/
│   ├── __init__.py
│   └── small_network.py
│
├── results/
│   ├── bits_vs_mse.png
│   ├── model_compression_by_layer.png
│   ├── model_compression_results.json
│   ├── model_orig_vs_reconstructed.png
│   ├── model_quantization_error_hist.png
│   ├── model_weight_distribution.png
│   └── original_vs_reconstructed.png
│
├── README.md
└── requirements.txt
```

---

## 9. Prototype 2 — Neural Network Weight Compression

Prototype 2 applies our existing quantization and bit-packing pipeline (`quantize()`, `pack_bits()`, `unpack_bits()`, `dequantize()`) to **real neural-network weight tensors** extracted from a PyTorch `SmallNeuralNetwork` model (109,386 parameters across `Linear(784, 128)`, `Linear(128, 64)`, and `Linear(64, 10)`).

### Scope & Disclaimer
> **Important**: At this stage, we are solely testing: *"Can our compression engine compress and reconstruct real neural-network weight tensors?"*
> We do **NOT** claim that the model itself has been hardware-accelerated, fine-tuned, or quantized for native inference hardware execution yet. The reconstructed model copy is created specifically to validate parameter loading, shape matching, non-NaN/Inf tensor integrity, and forward pass output prediction deviation.

### Prototype 2 Execution Pipeline
1. **Extract Weights**: Iterates through `model.named_parameters()` to record layer shapes, parameter counts, and original FP32 byte footprints.
2. **Quantize & Bit-Pack**: Applies uniform scalar quantization and bit-packing across INT8, INT4, and INT2 precision levels.
3. **Reconstruct & Load**: Unpacks bits, dequantizes back to FP32, and copies parameters into a reconstructed PyTorch model.
4. **Output Validation**: Runs random input tensor batches through both the original FP32 model and reconstructed INT8/INT4/INT2 models to measure output prediction deviation (Output MSE, MAE, Max Difference).

### Running Prototype 2
```bash
# From repository root:
python3 turboquant-lab/experiments/test_model_compression.py

# Or from inside turboquant-lab/:
python3 experiments/test_model_compression.py
```

Generated visualization artifacts in `results/`:
* `model_weight_distribution.png`: Weight probability density across FP32, INT8, INT4, INT2.
* `model_orig_vs_reconstructed.png`: Scatter plot comparing original vs reconstructed weight values.
* `model_quantization_error_hist.png`: Weight tensor residual error histogram.
* `model_compression_by_layer.png`: Layer-by-layer actual compression ratio bar chart.
* `model_compression_results.json`: Complete structured metrics benchmark log.

---

## 10. Getting Started & Running Experiments

### 1. Requirements

Install required Python dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run Benchmarks & Experiments

Execute unit tests, bit packing validation, and scalar quantization benchmark experiments:

```bash
# From repository root:
python3 turboquant-lab/experiments/test_quantization.py
python3 turboquant-lab/experiments/test_model_compression.py

# Or from inside turboquant-lab/:
python3 experiments/test_quantization.py
python3 experiments/test_model_compression.py
```

Upon completion, all benchmark metrics will output in the console and visualization plots will be generated in `results/`.

