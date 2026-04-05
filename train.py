# TRAIN.PY - The Eternal Blueprint
# This script runs on Kaggle to train the Queen.

import torch
import time
import random
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, 
    BitsAndBytesConfig, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import Dataset
import gc
import os

# --- OMEGA CONFIGURATION ---
# 1. THE MUSCLE (Hardware)
# 2. THE BRAIN (The Source of Intelligence)
# NOTE: In the future, the "Captain" will update this path automatically.
# For now, we point it to the v1 dataset you created.
# Path format: /kaggle/input/<your-dataset-name>/path/to/files
KAGGLE_BRAIN_PATH = "/kaggle/input/datasets/dushi2/project-honey-brain-v1/content/drive/MyDrive/Project_HONEY_Vault/mistral_queen_final"
OUTPUT_DIR = "/kaggle/working/omega_queen_updated"
BASE_MODEL_ID = "mistralai/Mistral-7B-v0.1"

# --- HARDWARE CONFIG (4-BIT QUANTIZATION) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=True # Ensures stability on Kaggle
)

# --- 1. THE FORAGER (Gathers Knowledge) ---
class CloudForager:
    def __init__(self):
        self.headers = {"User-Agent": "Mozilla/5.0"}
        self.targets = [
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/README.md",
            "https://en.wikipedia.org/wiki/Special:Random",
            "https://raw.githubusercontent.com/openai/gym/master/README.md"
        ]

    def hunt(self):
        url = random.choice(self.targets)
        try:
            res = requests.get(url, headers=self.headers, timeout=10)
            if res.status_code == 200:
                text = res.text
                soup = BeautifulSoup(text, 'html.parser')
                text = soup.get_text()[:2000] 
                return text
        except Exception as e:
            print(f"⚠️ Hunt failed: {e}")
        return None

# --- 2. THE DATASET (Prepares Knowledge) ---
class DynamicDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        with open(file_path, "r", encoding="utf-8") as f:
            self.text = f.read()
        
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        tokenized = self.tokenizer(
            self.text, 
            truncation=True, 
            max_length=self.block_size, 
            return_tensors="pt"
        )
        return {
            "input_ids": tokenized["input_ids"].flatten(),
            "attention_mask": tokenized["attention_mask"].flatten()
        }

# --- 3. THE TRAINER (Evolution Engine) ---
class OmegaTrainer:
    def __init__(self):
        print(f"🧠 LOADING BRAIN FROM: {KAGGLE_BRAIN_PATH}")
        
        # Memory Management
        gc.collect()
        torch.cuda.empty_cache()
        
        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(KAGGLE_BRAIN_PATH)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load Base Model
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, 
            quantization_config=bnb_config, 
            device_map="auto"
        )
        
        # Load Adapters (The Intelligence)
        self.model = PeftModel.from_pretrained(base_model, KAGGLE_BRAIN_PATH)
        print("✅ QUEEN AWAKE.")
        
    def train(self, data_file):
        if not Path(data_file).exists() or os.path.getsize(data_file) < 100:
            print("⏭️ Buffer empty. Hunting more...")
            return False

        print("🔥 STARTING EVOLUTION...")
        
        dataset = DynamicDataset(data_file, self.tokenizer, 256)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=False
        )
        
        training_args = TrainingArguments(
            output_dir="/kaggle/working/temp_checkpoints",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            optim="paged_adamw_32bit",
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            save_steps=50,
            max_grad_norm=0.3,
            report_to="none"
        )
            
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator
        )
        
        trainer.train()
        
        print("💾 SAVING EVOLVED BRAIN...")
        self.model.save_pretrained(OUTPUT_DIR)
        self.tokenizer.save_pretrained(OUTPUT_DIR)
        return True

# --- 4. THE INFINITE LOOP (The Heartbeat) ---
def autonomous_omega_cycle():
    print("🚀 OMEGA PROTOCOL INITIATED.")
    print("This system will run until the session expires (~9 hours).")
    
    forager = CloudForager()
    trainer = OmegaTrainer()
    cycle_count = 0
    
    buffer_text = ""
    BUFFER_FILE = "/kaggle/working/live_buffer.jsonl"
    DATA_THRESHOLD = 5000
    
    try:
        while True:
            cycle_count += 1
            print(f"\n--- CYCLE {cycle_count} ---")
            
            # Scrape
            print("🌐 SCANNING WORLD...")
            new_data = forager.hunt()
            
            if new_data:
                buffer_text += "\n" + new_data
                print(f"📥 Buffer: {len(buffer_text)} chars")
                
                with open(BUFFER_FILE, "w", encoding="utf-8") as f:
                    f.write(buffer_text)
                    
                if len(buffer_text) >= DATA_THRESHOLD:
                    print("✅ TRAINING...")
                    success = trainer.train(BUFFER_FILE)
                    
                    if success:
                        print("♻️ CLEARING BUFFER...")
                        buffer_text = ""
                        Path(BUFFER_FILE).unlink()
                        gc.collect()
                        torch.cuda.empty_cache()
            else:
                print("⚠️ Empty. Retrying...")
                
            print("💤 Waiting 5s...")
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n🛑 HALTED.")

# --- EXECUTE ---
autonomous_omega_cycle()
