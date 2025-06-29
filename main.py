import os
import shutil

# Disable wandb completely
os.environ["WANDB_DISABLED"] = "true"

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import kagglehub
import torch
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
import json
from typing import Dict, List
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType

# Create local directories
LOCAL_MODEL_DIR = "./local_base_model"
FINETUNED_MODEL_DIR = "./qwen_math_finetuned"

def download_and_save_model_locally():
    """Download model from kagglehub and save it locally in the project directory"""
    print("📥 Downloading model from Kaggle Hub...")
    
    # Download model to kaggle cache
    kaggle_model_path = kagglehub.model_download("qwen-lm/qwen-3/transformers/0.6b")
    print(f"Downloaded to: {kaggle_model_path}")
    
    # Create local directory if it doesn't exist
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    
    # Copy model files to local directory
    if not os.path.exists(os.path.join(LOCAL_MODEL_DIR, "config.json")):
        print(f"📋 Copying model files to local directory: {LOCAL_MODEL_DIR}")
        
        # Copy all files from kaggle cache to local directory
        for file_name in os.listdir(kaggle_model_path):
            src_file = os.path.join(kaggle_model_path, file_name)
            dst_file = os.path.join(LOCAL_MODEL_DIR, file_name)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dst_file)
                print(f"✅ Copied: {file_name}")
    else:
        print("✅ Local model already exists, skipping download.")
    
    return LOCAL_MODEL_DIR

# Download and prepare local model
model_path = download_and_save_model_locally()

# Load model and tokenizer from LOCAL directory
print(f"🔄 Loading model from local directory: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,  # Use float32 for CPU
    device_map="cpu",  # Force CPU since CUDA is not available
    trust_remote_code=True
)

# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Save tokenizer to local directory too (in case it's missing)
tokenizer.save_pretrained(model_path)

# Load and prepare dataset
def load_math_dataset(sample_size=200):  # DRAMATICALLY REDUCED SIZE
    """Load a small subset of the math dataset for fast training"""
    splits = {
        'math': 'data/math-00000-of-00001.parquet',
        'math500': 'data/math500-00000-of-00001.parquet',
        'gsm8k': 'data/gsm8k-00000-of-00001.parquet'
    }

    # Load the math dataset
    df = pd.read_parquet(
        "hf://datasets/yxsllgz-uts-org/Math_Consistency-Probability-Qwen2.5-Math-72B-Instruct-style2/" + splits["math"]
    )

    # Take only a small sample for fast training
    df = df.head(sample_size)
    print(f"📊 Using only {len(df)} samples for fast training")

    return df

def format_chat_template(example):
    """Format the data into chat template format with shorter responses"""
    # Truncate long solutions for faster training
    solution = example['solution']
    if len(solution) > 500:  # Limit solution length
        solution = solution[:500] + "..."

    messages = [
        {"role": "user", "content": example['problem']},
        {"role": "assistant", "content": solution}
    ]

    # Apply chat template
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    return {"text": formatted_text}

def tokenize_function(examples):
    """Tokenize with very short max length for speed"""
    # Tokenize with very short sequences
    model_inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=512,  # VERY SHORT for speed
        return_tensors="pt"
    )

    # For causal LM, labels are the same as input_ids
    model_inputs["labels"] = model_inputs["input_ids"].clone()

    return model_inputs

def prepare_dataset():
    """Prepare a small dataset for fast training"""
    # Load small dataset
    df = load_math_dataset(sample_size=200)  # Only 200 samples!

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)

    # Split into train/validation (80/20)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    # Format the data
    train_dataset = dataset['train'].map(format_chat_template, remove_columns=dataset['train'].column_names)
    eval_dataset = dataset['test'].map(format_chat_template, remove_columns=dataset['test'].column_names)

    # Tokenize datasets
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    return train_dataset, eval_dataset

def setup_lora_config():
    """Setup LoRA configuration for very fast training"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=4,  # VERY LOW rank for speed
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ]  # Only target attention modules for speed
    )
    return lora_config

def setup_fast_training_arguments():
    """Setup training arguments optimized for MAXIMUM SPEED"""
    training_args = TrainingArguments(
        output_dir=FINETUNED_MODEL_DIR,  # Use local directory
        overwrite_output_dir=True,
        num_train_epochs=1,  # ONLY 1 epoch
        per_device_train_batch_size=2,  # Slightly larger batch
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,  # Reduced
        learning_rate=1e-4,  # Higher learning rate for faster convergence
        warmup_steps=10,  # Very few warmup steps
        logging_steps=5,   # Log frequently to see progress
        save_steps=50,     # Save less frequently
        eval_steps=50,     # Evaluate less frequently
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=False,  # Skip for speed
        fp16=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=[],
        dataloader_num_workers=0,
        use_cpu=True,
        disable_tqdm=False,
        max_steps=50,  # LIMIT TOTAL STEPS - this is the key!
        save_total_limit=2,  # Keep only 2 checkpoints
    )
    return training_args

def train_model_fast():
    """Ultra-fast training function"""
    print("🚀 Starting FAST fine-tuning process...")
    print("⚡ This is optimized for speed, not performance!")

    # Prepare datasets
    print("📊 Preparing small dataset...")
    train_dataset, eval_dataset = prepare_dataset()

    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")

    # Setup LoRA
    print("🔧 Setting up minimal LoRA configuration...")
    lora_config = setup_lora_config()
    model_lora = get_peft_model(model, lora_config)

    # Print trainable parameters
    model_lora.print_trainable_parameters()

    # Setup training arguments
    training_args = setup_fast_training_arguments()

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model_lora,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # Start training
    print("🏋️ Starting FAST training...")
    print("⏱️  Should complete in 10-20 minutes instead of hours!")
    trainer.train()

    # Save the final model
    print("💾 Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    # Also save base model info for reference
    model_info = {
        "base_model_path": model_path,
        "base_model_name": "qwen-lm/qwen-3/transformers/0.6b",
        "fine_tuned_path": FINETUNED_MODEL_DIR
    }
    
    with open(os.path.join(FINETUNED_MODEL_DIR, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)

    print("✅ Fast fine-tuning completed!")
    print(f"📁 Model saved to: {FINETUNED_MODEL_DIR}")
    print(f"📁 Base model saved to: {model_path}")
    return trainer

def quick_test():
    """Quick test of the fast-trained model"""
    print("🧪 Quick test of fast-trained model...")

    try:
        from peft import PeftModel

        # Load base model from LOCAL directory
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,  # Use local path
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )

        # Load fine-tuned adapter
        finetuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL_DIR)
        finetuned_tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_DIR)

        # Test problem
        test_problem = "Solve: 2x + 4 = 10"
        messages = [{"role": "user", "content": test_problem}]
        text = finetuned_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = finetuned_tokenizer([text], return_tensors="pt")

        with torch.no_grad():
            generated_ids = finetuned_model.generate(
                **model_inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=finetuned_tokenizer.eos_token_id
            )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        response = finetuned_tokenizer.decode(output_ids, skip_special_tokens=True)

        print(f"Problem: {test_problem}")
        print(f"Response: {response}")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

def verify_local_setup():
    """Verify that all local files are properly saved"""
    print("\n🔍 Verifying local setup...")
    
    # Check base model
    if os.path.exists(LOCAL_MODEL_DIR):
        base_files = os.listdir(LOCAL_MODEL_DIR)
        print(f"✅ Base model directory exists: {LOCAL_MODEL_DIR}")
        print(f"📁 Contains {len(base_files)} files: {base_files[:5]}{'...' if len(base_files) > 5 else ''}")
    else:
        print(f"❌ Base model directory missing: {LOCAL_MODEL_DIR}")
    
    # Check fine-tuned model
    if os.path.exists(FINETUNED_MODEL_DIR):
        ft_files = os.listdir(FINETUNED_MODEL_DIR)
        print(f"✅ Fine-tuned model directory exists: {FINETUNED_MODEL_DIR}")
        print(f"📁 Contains {len(ft_files)} files: {ft_files}")
    else:
        print(f"❌ Fine-tuned model directory missing: {FINETUNED_MODEL_DIR}")
    
    print("\n💡 All models are now stored locally in your project directory!")
    print("🌍 This setup will work across different operating systems.")

if __name__ == "__main__":
    print("⚡ FAST TRAINING MODE - LOCAL STORAGE")
    print("🎯 This will complete in ~10-20 minutes")
    print("📊 Using only 200 samples, 1 epoch, 100 max steps")
    print("💾 All models saved locally in project directory")
    print("⚠️  This is for testing - use full training for production")

    try:
        # Train the model quickly
        trainer = train_model_fast()

        # Quick test
        quick_test()
        
        # Verify setup
        verify_local_setup()

        print("\n✅ Fast training completed!")
        print("💡 To train properly later, use the full dataset and more epochs")
        print("🌍 Your models are now portable across different operating systems!")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()