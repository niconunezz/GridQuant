# GridQuant

## Overview

This repository tries to implements the ideas presented in the blog post *"Accelerating 2D Dynamic Block Quantized Float8 GEMMs in Triton"*. Designed specifically for NVIDIA H100 GPUs, it leverages advanced features like float8 computation, Triton's high-performance GPU programming capabilities, and the Tensor Memory Accelerator (TMA). These elements enable state-of-the-art GEMM kernels by optimizing memory transfer efficiency, reducing latency, and maximizing computational throughput.

## Motivation

The goal of this project is to accelerate GEMMs (General Matrix Multiplications) by using dynamic block quantization with float8. By reducing memory bandwidth and maintaining computational accuracy, this work aims to push the boundaries of GPU performance.

## Installation

To install the dependencies, run:

```bash
pip install -r requirements.txt
```

## Repository Structure

The repository is organized as follows:

- **`bench.py`**: Contains benchmarking utilities to compare Triton and PyTorch implementations.
- **`gemm.py`**: Implements Triton kernels for GEMMs with persistent memory and dynamic block quantization.
- **`main.py`**: Entry point for the quantization and GEMM computation pipeline.
- **`quant.py`**: Handles block-level quantization of matrices into float8 format.
- **`requirements.txt`**: Lists Python dependencies.

## Features

### Quantization

The `quant.py` module quantizes input matrices by:

1. Dividing matrices into subblocks and determining the maximum absolute value for each block during the first pass.
2. Using the maximum value found in the first pass as a scaling factor to quantize the data into the float8 representation during the second pass.
3. Storing these scaling factors for later dequantization to recover the original matrix.

### GEMM Kernel

The `gemm.py` module leverages Triton to:

- Load matrix blocks efficiently using Tensor Memory Accelerator (TMA) descriptors.
- Perform tiled, persistent matrix multiplication to optimize SM utilization.
- TODO: Employ cooperative kernel design with warp specialization, enabling parallelism by dedicating warp groups to specific tasks and reducing data dependencies.
- Store results back to memory in float8 format, utilizing optimized scaling and dequantization techniques.

### Benchmarks

Benchmarks in `bench.py` compare Triton-based implementations with PyTorch, plotting speedups and memory efficiency for various matrix sizes.

### Example Usage

To run the pipeline for 4096 x 4096 matrices:

```bash
python main.py
```

## Performance

Preliminary results on H100 GPUs demonstrate:

- Significant speedups due to reduced memory usage with float8 precision.
- Optimized memory access patterns via Triton's TMA descriptors.

Future updates will include detailed benchmarks and accuracy analyses.

## TODOs

- Implement warp-specialization, recently merged into triton.

- Enhance benchmarking and accuracy testing.

## Acknowledgements

This repository draws inspiration from the blog post *"Accelerating 2D Dynamic Block Quantized Float8 GEMMs in Triton"* and Triton's official documentation and examples.

## Citation

If you use this work, please cite the original blog post and this repository.
