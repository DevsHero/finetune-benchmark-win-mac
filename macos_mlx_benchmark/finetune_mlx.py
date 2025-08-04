import time
import json
import os
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm.utils import load
from datasets import load_dataset
from tqdm import tqdm
import subprocess

# --- Configuration ---
# Set to "fast" for a quick test run, or "full" for a complete fine-tuning.
TUNING_MODE = "fast" # Options: "fast", "full"
FAST_TUNE_SAMPLES = 100 # Number of samples for the "fast" tuning mode

MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
DATASET_NAME = "databricks/databricks-dolly-15k"
NUM_EPOCHS = 1
BATCH_SIZE = 1 
MAX_SEQ_LENGTH = 2048

def get_mac_specs():
    """Gets macOS hardware and software specifications with robust parsing."""
    specs = {
        "os_version": "Unknown",
        "chip": "Unknown",
        "memory": "Unknown",
        "gpu": "Unknown"
    }

    try:
        # Get hardware info
        hardware_info = subprocess.check_output(['system_profiler', 'SPHardwareDataType'], text=True)
        for line in hardware_info.split('\n'):
            line = line.strip()
            if "chip:" in line.lower():
                specs["chip"] = line.split(':', 1)[1].strip()
            elif "memory:" in line.lower():
                specs["memory"] = line.split(':', 1)[1].strip()
    except (subprocess.CalledProcessError, IndexError) as e:
        print(f"Error getting hardware specs (chip, memory): {e}")

    try:
        # Get software info
        software_info = subprocess.check_output(['system_profiler', 'SPSoftwareDataType'], text=True)
        for line in software_info.split('\n'):
            line = line.strip()
            if "system version:" in line.lower():
                specs["os_version"] = line.split(':', 1)[1].strip()
    except (subprocess.CalledProcessError, IndexError) as e:
        print(f"Error getting software specs (os_version): {e}")

    try:
        # Get Metal info
        metal_info = subprocess.check_output(['system_profiler', 'SPDisplaysDataType'], text=True)
        for line in metal_info.split('\n'):
            line = line.strip()
            if "metal support:" in line.lower():
                specs["gpu"] = line.split(':', 1)[1].strip()
    except (subprocess.CalledProcessError, IndexError) as e:
        print(f"Error getting display specs (gpu): {e}")

    return specs


def format_prompt(sample):
    # Reformat the dataset into a conversational format for fine-tuning
    # This is a simplified example; you might want to customize it further
    return f"""<|user|>
{sample['instruction']}
{sample['context']}<|end|>
<|assistant|>
{sample['response']}<|end|>"""

def main():
    print("Starting macOS MLX Fine-Tuning Benchmark...")
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET_NAME}")

    # 1. Load Model and Tokenizer
    print("\n1. Loading model and tokenizer...")
    model, tokenizer = load(MODEL_NAME)
    
    # Freeze all layers first
    model.freeze()
    # Unfreeze the last few layers for fine-tuning (LoRA is also a good option)
    # For this benchmark, we'll fine-tune the last attention block and the output layer
    for i in range(len(model.model.layers) - 4, len(model.model.layers)):
        model.model.layers[i].unfreeze()
    model.lm_head.unfreeze()

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
    
    # 3. Define Loss and Training Step
    def loss_fn(model, inputs, targets):
        logits = model(inputs)
        logits = logits.astype(mx.float32)
        return nn.losses.cross_entropy(logits, targets, reduction="mean")

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.Adam(learning_rate=1e-5)

    # 4. Fine-Tuning Loop
    print("\n3. Starting fine-tuning...")
    start_time = time.time()
    epoch_times = []

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(formatted_dataset, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for sample in pbar:
            text = sample["text"]
            tokens = tokenizer.encode(text)
            
            if len(tokens) > MAX_SEQ_LENGTH:
                tokens = tokens[:MAX_SEQ_LENGTH]

            inputs = mx.array(tokens[:-1])
            targets = mx.array(tokens[1:])
            
            # Add batch dimension
            inputs = mx.expand_dims(inputs, axis=0)
            targets = mx.expand_dims(targets, axis=0)

            loss, grads = loss_and_grad_fn(model, inputs, targets)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{total_loss/num_batches:.3f}"})

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        print(f"Epoch {epoch+1} finished in {epoch_duration:.2f}s")

    end_time = time.time()
    total_duration = end_time - start_time
    print(f"\nFine-tuning completed in {total_duration:.2f}s.")

    # 5. Generate Summary Report
    print("\n4. Generating summary report...")
    num_samples = len(formatted_dataset)
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    samples_per_second = num_samples / total_duration if total_duration > 0 else 0

    summary = {
        "platform": "macos_mlx",
        "model": MODEL_NAME,
        "dataset": DATASET_NAME,
        "device_specs": get_mac_specs(),
        "num_samples": num_samples,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "total_duration_seconds": round(total_duration, 2),
        "average_epoch_time_seconds": round(avg_epoch_time, 2),
        "throughput_samples_per_second": round(samples_per_second, 2)
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(script_dir, "benchmark_summary.json")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Benchmark summary saved to {summary_path}")
    print("\n--- Benchmark Finished ---")

if __name__ == "__main__":
    main()
