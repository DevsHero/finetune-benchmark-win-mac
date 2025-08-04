import time
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from tqdm import tqdm
import subprocess
import re

# --- Configuration ---
# Set to "fast" for a quick test run, or "full" for a complete fine-tuning.
TUNING_MODE = "fast" # Options: "fast", "full"
FAST_TUNE_SAMPLES = 100 # Increased for meaningful benchmarking on RTX 5060 Ti

MODEL_NAME = "Qwen/Qwen3-0.6B"  # Qwen3-0.6B optimized for RTX 5060 Ti
DATASET_NAME = "databricks/databricks-dolly-15k"
NUM_EPOCHS = 1
BATCH_SIZE = 4 # Optimized batch size for better GPU utilization on RTX 5060 Ti
GRADIENT_ACCUMULATION_STEPS = 2 # Effective batch size = 16 for optimal performance
MAX_SEQ_LENGTH = 512  # Optimized sequence length for RTX 5060 Ti

def get_windows_specs():
    """Gets Windows hardware and software specifications with detailed GPU info."""
    specs = {
        "os_version": "Unknown",
        "processor": "Unknown",
        "memory": "Unknown",
        "gpu": "Unknown"
    }
    try:
        # Get OS and hardware info from systeminfo
        sysinfo = subprocess.check_output(['systeminfo'], text=True)
        os_search = re.search(r'OS Name:\s*(.*)', sysinfo)
        if os_search:
            specs["os_version"] = os_search.group(1).strip()
        
        mem_search = re.search(r'Total Physical Memory:\s*(.*)', sysinfo)
        if mem_search:
            specs["memory"] = mem_search.group(1).strip()

        proc_search = re.search(r'Processor\(s\):\s*(.*)', sysinfo)
        if proc_search:
            specs["processor"] = proc_search.group(1).strip()

    except (subprocess.CalledProcessError, AttributeError) as e:
        print(f"Error getting basic system specs: {e}")

    try:
        # Get detailed GPU info using nvidia-smi for NVIDIA GPUs
        if torch.cuda.is_available() and 'nvidia' in torch.cuda.get_device_name(0).lower():
            nvsmi_output = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader,nounits'],
                text=True
            )
            gpu_info = nvsmi_output.strip().split(',')
            specs['gpu'] = {
                "name": gpu_info[0].strip(),
                "driver_version": gpu_info[1].strip(),
                "total_memory": f"{gpu_info[2].strip()} MiB"
            }
        else: # Fallback for non-NVIDIA GPUs
            gpu_info = subprocess.check_output(['wmic', 'path', 'win32_videocontroller', 'get', 'name'], text=True)
            specs['gpu'] = gpu_info.split('\n')[1].strip()

    except (subprocess.CalledProcessError, IndexError) as e:
        print(f"Error getting GPU specs: {e}")
        specs['gpu'] = "Unknown (error fetching)"

    return specs


def format_prompt(sample):
    # Reformat the dataset into a conversational format for fine-tuning
    return f"""<|user|>
{sample['instruction']}
{sample['context']}<|end|>
<|assistant|>
{sample['response']}<|end|>"""

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Starting Windows PyTorch Fine-Tuning Benchmark...")
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET_NAME}")

    # 1. Load Model and Tokenizer
    print("\n1. Loading model and tokenizer...")
    model_load_start = time.time()

    # Check for CUDA GPU and exit if not available
    if not torch.cuda.is_available():
        print("Error: No CUDA-enabled GPU found. This script requires a GPU.")
        exit()
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Optimized for RTX 5060 Ti 16GB with BF16 precision for stability
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map="cuda", # Explicitly use the GPU
        torch_dtype=torch.bfloat16  # Using BF16 for better stability on RTX 5060 Ti
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load and Prepare Dataset
    print("\n2. Loading and preparing dataset...")
    dataset = load_dataset(DATASET_NAME, split="train")
    
    # Adjust dataset size based on the tuning mode
    if TUNING_MODE == "fast":
        print(f"Running in 'fast' mode. Using {FAST_TUNE_SAMPLES} samples.")
        dataset = dataset.select(range(FAST_TUNE_SAMPLES))
    elif TUNING_MODE == "full":
        print("Running in 'full' mode. Using the entire dataset.")
        # No change, use the full dataset
        pass
    else:
        raise ValueError(f"Unknown TUNING_MODE: {TUNING_MODE}. Please choose 'fast' or 'full'.")
    
    formatted_dataset = dataset.map(lambda x: {"text": format_prompt(x)})
    
    def tokenize_function(example):
        # Simple single example tokenization
        text = example["text"]
        
        # Tokenize the text
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            return_tensors=None
        )
        
        # Set labels for causal language modeling
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_dataset = formatted_dataset.map(tokenize_function, remove_columns=dataset.column_names)

    model_load_time = time.time() - model_load_start
    print(f"Model loading completed in {model_load_time:.2f}s.")
    
    # 3. Fine-Tuning with Trainer API
    print("\n3. Starting fine-tuning...")
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        logging_dir='./logs',
        logging_steps=10,
        learning_rate=3e-5,  # Slightly higher learning rate for smaller models
        fp16=False,  # Disable FP16 to avoid gradient scaling issues
        bf16=True,   # Enable BF16 for RTX 5060 Ti optimization
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        dataloader_pin_memory=True,  # Faster data transfer on RTX 5060 Ti
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
        dataloader_num_workers=0,  # Set to 0 for Windows compatibility
        remove_unused_columns=False,
        optim="adamw_torch_fused",  # Optimized optimizer for NVIDIA GPUs
        max_grad_norm=1.0,  # Gradient clipping for stability
        warmup_ratio=0.1,  # Better warmup strategy than fixed steps
        save_strategy="no",  # Disable saving to focus on performance
        report_to="none", # Disable wandb/tensorboard reporting for this benchmark
    )

    # Use default data collator to avoid tokenization issues
    from transformers import default_data_collator
    data_collator = default_data_collator

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Start benchmark timing here (excluding model loading)
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    
    total_duration = end_time - start_time
    print(f"\nFine-tuning completed in {total_duration:.2f}s.")

    # 4. Generate Summary Report
    print("\n4. Generating summary report...")
    num_samples = len(tokenized_dataset)
    avg_epoch_time = total_duration / NUM_EPOCHS
    samples_per_second = num_samples / total_duration if total_duration > 0 else 0

    summary = {
        "platform": "windows_pytorch",
        "model": MODEL_NAME,
        "dataset": DATASET_NAME,
        "device_specs": get_windows_specs(),
        "num_samples": num_samples,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "total_duration_seconds": round(total_duration, 2),
        "average_epoch_time_seconds": round(avg_epoch_time, 2),
        "throughput_samples_per_second": round(samples_per_second, 2),
    }

    summary_path = os.path.join(script_dir, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Benchmark summary saved to {summary_path}")
    print("\n--- Benchmark Finished ---")

if __name__ == "__main__":
    main()
