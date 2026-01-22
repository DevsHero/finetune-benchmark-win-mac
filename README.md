# Fine-Tuning Performance Benchmark: macOS (MLX) vs. Windows (PyTorch) vs. Ubuntu (PyTorch)

## Project Overview

This project provides a standardized benchmark for fine-tuning small language models on three distinct platforms:

1.  **macOS (Apple Silicon):** Using the MLX framework with `mlx-community/Qwen3-0.6B-bf16`, optimized for Apple's unified memory architecture.
2.  **Windows (NVIDIA GPU):** Using PyTorch with `Qwen/Qwen3-0.6B` and CUDA acceleration.
3.  **Ubuntu (NVIDIA GPU):** Using PyTorch with `Qwen/Qwen3-0.6B` and CUDA acceleration on Linux.

The primary goal is to offer a clear, reproducible comparison of fine-tuning performance, helping developers and researchers understand the trade-offs between these two popular setups. The benchmarks measure key metrics like total training time, throughput, and hardware utilization.
## Current Benchmark Results

Based on recent runs, here are example performance metrics:

### macOS MLX (M4 Pro, 20 GPU, 24GB RAM)
- **Model:** `mlx-community/Qwen3-0.6B-bf16`
- **Throughput:** ~5.28 samples/second
- **Total Time:** ~18.93 seconds (100 samples)
- **Memory:** Efficient unified memory usage

### PC Windows PyTorch (Ryzen5 8400f, RTX 5060 Ti 16GB VRAM)
- **Model:** `Qwen/Qwen3-0.6B`
- **Throughput:** ~7.5 samples/second
- **Total Time:** ~13.33 seconds (100 samples)
- **Configuration:** BF16 precision, batch size 4, gradient accumulation

### MSI Laptop Windows PyTorch (ultra 275hx, RTX 5080 16GB VRAM)
- **Model:** `Qwen/Qwen3-0.6B`
- **Throughput:** ~11.16 samples/second
- **Total Time:** ~8.96 seconds (100 samples)
- **Configuration:** BF16 precision, batch size 4, gradient accumulation

### PC Ubuntu PyTorch (Intel i5-12400, RTX 5060 Ti 16GB RAM)
- **Model:** `Qwen/Qwen3-0.6B`
- **Throughput:** ~8.21 samples/second
- **Total Time:** ~12.17 seconds (100 samples)
- **Configuration:** BF16 precision, batch size 4, gradient accumulation
- **OS:** Ubuntu 24.04.3 LTS
- **GPU Driver:** 580.95.05

### Ubuntu PyTorch (Intel i5-12400, RTX 3090, 31GB RAM)
- **Model:** `Qwen/Qwen3-0.6B`
- **Throughput:** ~10.67 samples/second
- **Total Time:** ~9.37 seconds (100 samples)
- **Configuration:** BF16 precision, batch size 4, gradient accumulation
- **OS:** Ubuntu 24.04.3 LTS
- **GPU Driver:** 580.95.05
- **GPU Memory:** 24GB

### Ubuntu PyTorch (Intel i5-12400, RTX 4080 Super, 31GB RAM)
- **Model:** `Qwen/Qwen3-0.6B`
- **Throughput:** ~15.92 samples/second
- **Total Time:** ~6.28 seconds (100 samples)
- **Configuration:** BF16 precision, batch size 4, gradient accumulation
- **OS:** Ubuntu 24.04.3 LTS
- **GPU Driver:** 580.95.05
- **GPU Memory:** 16GB

### Ubuntu PyTorch (Intel i5-12400, RTX 4070 Ti Super, 31GB RAM)
- **Model:** `Qwen/Qwen3-0.6B`
- **Throughput:** ~14.33 samples/second
- **Total Time:** ~6.98 seconds (100 samples)
- **Configuration:** BF16 precision, batch size 4, gradient accumulation
- **OS:** Ubuntu 24.04.3 LTS
- **GPU Driver:** 580.95.05
- **GPU Memory:** 16GB

## Benchmark Details

-   **Base Models:** 
    - **macOS:** `mlx-community/Qwen3-0.6B-bf16` (MLX-optimized)
    - **Windows:** `Qwen/Qwen3-0.6B` (PyTorch)
-   **Dataset:** `databricks/databricks-dolly-15k`
-   **Task:** Supervised fine-tuning on a conversational dataset.

### Current Configuration

Both benchmarks are configured for optimal performance on their respective platforms:

- **Sample Size:** 100 samples (fast mode) for quick benchmarking
- **Batch Size:** 2 (macOS), 4 (Windows) - optimized for each platform
- **Sequence Length:** 512 tokens (optimized for memory efficiency)
- **Epochs:** 1

### Fine-Tuning Strategy

-   **macOS (MLX):** The script unfreezes the last four attention blocks and the output layer (either `lm_head` or tied embeddings). This targeted approach minimizes computational overhead while still adapting the model effectively.
-   **Windows (PyTorch):** The script leverages the Hugging Face `Trainer` API with BF16 precision, gradient accumulation, and optimized settings for NVIDIA GPUs.

## Configuration

Both benchmark scripts can be run in one of two modes, controlled by the `TUNING_MODE` variable at the top of each script:

-   `fast`: Runs a quick test on a small subset of the data (`100` samples by default). This is useful for verifying the setup and getting a rough performance estimate.
-   `full`: Uses the entire dataset for a complete fine-tuning run, providing a more accurate and comprehensive benchmark.

### Performance Optimizations

**macOS MLX:**
- Uses `mlx-community/Qwen3-0.6B-bf16` for optimal MLX performance
- Batch size: 2, Sequence length: 512 (memory optimized)
- Automatically detects and handles tied embeddings vs. separate `lm_head`
- Unfreezes last 4 transformer layers + output layer

**Windows PyTorch:**
- Uses `Qwen/Qwen3-0.6B` with BF16 precision
- Batch size: 4, Gradient accumulation: 2 (effective batch size 8)
- Optimized for NVIDIA RTX GPUs with fused AdamW optimizer
- Gradient checkpointing enabled for memory efficiency

**Ubuntu PyTorch:**
- Uses `Qwen/Qwen3-0.6B` with BF16 precision
- Batch size: 4, Gradient accumulation: 2 (effective batch size 8)
- Optimized for NVIDIA RTX GPUs with fused AdamW optimizer
- Gradient checkpointing enabled for memory efficiency
- Data loading optimized for Linux with 4 workers

## Instructions

### 1. macOS (MLX) Benchmark

**Environment Setup:**

```bash
# Navigate to the macOS directory
cd macos_mlx_benchmark

# Create and activate a Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

**Running the Benchmark:**

```bash
# This will run the script and generate the benchmark summary
python finetune_mlx.py
```

**Note:** The MLX script automatically handles models with tied embeddings and will work with both `lm_head` and tied embedding architectures.

### 2. Windows (PyTorch) Benchmark

**Environment Setup:**

```bash
# Navigate to the Windows directory
cd windows_pytorch_benchmark

# Create and activate a Python virtual environment
python -m venv venv
source venv/bin/activate

# Install the required packages
# Ensure you have a compatible PyTorch version with CUDA support
pip install -r requirements.txt
```

**Running the Benchmark:**

```bash
# This will run the script and generate the benchmark summary
python finetune_pytorch.py
```

**Note:** Ensure you have CUDA-compatible PyTorch installed and an NVIDIA GPU available. The script includes optimizations for RTX series GPUs.

### 3. Ubuntu (PyTorch) Benchmark

**Environment Setup:**

```bash
# Navigate to the Ubuntu directory
cd ubuntu_pytorch_benchmark

# Create and activate a Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the required packages
# Ensure you have a compatible PyTorch version with CUDA support
pip install -r requirements.txt
```

**Running the Benchmark:**

```bash
# This will run the script and generate the benchmark summary
python finetune_pytorch.py
```

**Note:** Ensure you have CUDA-compatible PyTorch installed and an NVIDIA GPU available. The script includes optimizations for RTX series GPUs and Linux-specific data loading optimizations.

## Collecting Results

After each script finishes, it will save a `benchmark_summary.json` file in its respective directory. This file contains detailed performance metrics and hardware specifications, allowing for a direct comparison between the two platforms.


## Key Features

- **Cross-platform compatibility:** Optimized for both Apple Silicon and NVIDIA GPUs
- **Flexible configuration:** Easy to switch between "fast" and "full" benchmark modes
- **Detailed metrics:** Hardware specifications, throughput, and timing data
- **Modern models:** Uses efficient Qwen3-0.6B variants optimized for each platform
