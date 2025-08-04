import time
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from tqdm import tqdm
import subprocess
import re

# --- Configuration ---
# Set to "fast" for a quick test run, or "full" for a complete fine-tuning.
TUNING_MODE = "fast" # Options: "fast", "full"
FAST_TUNE_SAMPLES = 100 # Number of samples for the "fast" tuning mode

MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
DATASET_NAME = "databricks/databricks-dolly-15k"
NUM_EPOCHS = 1
BATCH_SIZE = 1 # Per device
MAX_SEQ_LENGTH = 2048

def get_windows_specs():
    """Gets Windows hardware and software specifications."""
    try:
        # Get OS and hardware info
        sysinfo = subprocess.check_output(['systeminfo'], text=True)
        os_name = re.search(r'OS Name:\s*(.*)', sysinfo).group(1).strip()
        os_version = re.search(r'OS Version:\s*(.*)', sysinfo).group(1).strip()
        total_memory = re.search(r'Total Physical Memory:\s*(.*)', sysinfo).group(1).strip()
        processor = re.search(r'Processor\(s\):\s*(.*)', sysinfo).group(1).strip()

        # Get GPU info
        gpu_info = subprocess.check_output(['wmic', 'path', 'win32_videocontroller', 'get', 'name'], text=True)
        gpu_name = gpu_info.split('\n')[1].strip()

        return {
            "os_name": os_name,
            "os_version": os_version,
            "processor": processor,
            "memory": total_memory,
            "gpu": gpu_name
        }
    except (subprocess.CalledProcessError, AttributeError, IndexError) as e:
        print(f"Error getting Windows specs: {e}")
        return {
            "os_name": "Unknown",
            "os_version": "Unknown",
            "processor": "Unknown",
            "memory": "Unknown",
            "gpu": "Unknown"
        }

def format_prompt(sample):
    # Reformat the dataset into a conversational format for fine-tuning
    return f"""<|user|>
{sample['instruction']}
{sample['context']}<|end|>
<|assistant|>
{sample['response']}<|end|>"""

def main():
    print("Starting Windows PyTorch Fine-Tuning Benchmark...")
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET_NAME}")

    # 1. Load Model and Tokenizer
    print("\n1. Loading model and tokenizer...")
    # Note: bfloat16 is recommended for Ampere and newer GPUs
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", # Automatically use the GPU
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
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
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_SEQ_LENGTH, padding="max_length")

    tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    # 3. Fine-Tuning with Trainer API
    print("\n3. Starting fine-tuning...")
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        logging_dir='./logs',
        logging_steps=10,
        learning_rate=2e-5,
        fp16=False, # bfloat16 is used instead
        bf16=True,
        gradient_accumulation_steps=4,
        report_to="none", # Disable wandb/tensorboard reporting for this benchmark
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

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

    with open("benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("Benchmark summary saved to benchmark_summary.json")
    print("\n--- Benchmark Finished ---")

if __name__ == "__main__":
    main()
