# Fine-Tuning Performance Benchmark: macOS (MLX) vs. Windows (PyTorch)

## Project Overview

This project provides a standardized benchmark for fine-tuning the `microsoft/phi-3-mini-4k-instruct` language model on two distinct platforms:

1.  **macOS (Apple Silicon):** Using the MLX framework, optimized for Apple's unified memory architecture.
2.  **Windows (NVIDIA GPU):** Using PyTorch with CUDA for GPU acceleration.

The primary goal is to offer a clear, reproducible comparison of fine-tuning performance, helping developers and researchers understand the trade-offs between these two popular setups. The benchmarks measure key metrics like total training time, throughput, and hardware utilization.

## Benchmark Details

-   **Base Model:** `microsoft/phi-3-mini-4k-instruct`
-   **Dataset:** `databricks/databricks-dolly-15k`
-   **Task:** Supervised fine-tuning on a conversational dataset.

### Fine-Tuning Strategy

-   **macOS (MLX):** The script unfreezes the last four attention blocks and the language model head for fine-tuning. This targeted approach minimizes computational overhead while still adapting the model effectively.
-   **Windows (PyTorch):** The script leverages the Hugging Face `Trainer` API, a high-level utility that handles the training loop, mixed-precision (BF16), and gradient accumulation.

## Configuration

Both benchmark scripts can be run in one of two modes, controlled by the `TUNING_MODE` variable at the top of each script:

-   `fast`: Runs a quick test on a small subset of the data (`100` samples by default). This is useful for verifying the setup and getting a rough performance estimate.
-   `full`: Uses the entire dataset for a complete fine-tuning run, providing a more accurate and comprehensive benchmark.

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

## Collecting Results

After each script finishes, it will save a `benchmark_summary.json` file in its respective directory. This file contains detailed performance metrics and hardware specifications, allowing for a direct comparison between the two platforms.