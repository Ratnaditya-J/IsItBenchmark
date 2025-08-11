#!/usr/bin/env python3
"""
GPT-OSS Contamination Detection Trainer

Custom trainer for GPT-OSS models using prompt-based contamination detection.
Since GPT-OSS models don't support AutoModelForSequenceClassification,
we use a causal language modeling approach with structured prompts.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import json

logger = logging.getLogger(__name__)


class ContaminationPromptDataset(Dataset):
    """Dataset for prompt-based contamination detection with GPT-OSS."""
    
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Set pad token if not available
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Create structured prompt for contamination detection
        prompt = self._create_contamination_prompt(
            item['text'], 
            item['label']  # 1 for contaminated, 0 for clean
        )
        
        # Tokenize the prompt
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()  # For causal LM training
        }
    
    def _create_contamination_prompt(self, text: str, label: int) -> str:
        """Create a structured prompt for contamination detection."""
        
        # Define the task instruction
        instruction = (
            "Analyze the following text for benchmark contamination. "
            "Determine if this text appears to be from a known benchmark dataset. "
            "Respond with 'CONTAMINATED' if it's from a benchmark, or 'CLEAN' if it's original content.\n\n"
        )
        
        # Add the text to analyze
        text_section = f"Text to analyze: {text}\n\n"
        
        # Add the expected response based on label
        response = "CONTAMINATED" if label == 1 else "CLEAN"
        answer_section = f"Analysis: {response}"
        
        return instruction + text_section + answer_section


class GPTOSSContaminationTrainer:
    """Custom trainer for GPT-OSS contamination detection."""
    
    def __init__(
        self,
        model_name: str = "openai/gpt-oss-20b",
        max_length: int = 512,
        device: str = "auto"
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        
        # Initialize tokenizer and model
        logger.info(f"Loading GPT-OSS model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device if device != "auto" else None
        )
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
    
    def train(
        self,
        train_data: List[Dict[str, Any]],
        val_data: List[Dict[str, Any]],
        output_dir: str,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 1e-5,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        save_steps: int = 500,
        eval_steps: int = 500
    ) -> Dict[str, Any]:
        """Train the GPT-OSS model for contamination detection."""
        
        logger.info(f"Starting GPT-OSS contamination detection training")
        logger.info(f"Training samples: {len(train_data)}")
        logger.info(f"Validation samples: {len(val_data)}")
        
        # Create datasets
        train_dataset = ContaminationPromptDataset(train_data, self.tokenizer, self.max_length)
        val_dataset = ContaminationPromptDataset(val_data, self.tokenizer, self.max_length)
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal language modeling
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            learning_rate=learning_rate,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=[]  # Disable W&B
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Train the model
        start_time = time.time()
        train_result = trainer.train()
        training_time = time.time() - start_time
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training configuration
        config = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "training_args": training_args.to_dict(),
            "training_time": training_time,
            "train_samples": len(train_data),
            "val_samples": len(val_data)
        }
        
        with open(f"{output_dir}/training_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Final training loss: {train_result.training_loss:.4f}")
        
        return {
            "training_loss": train_result.training_loss,
            "training_time": training_time,
            "model_path": output_dir
        }
    
    def predict_contamination(self, text: str) -> Tuple[bool, float]:
        """Predict if text is contaminated using the trained model."""
        
        # Create prompt for prediction
        prompt = (
            "Analyze the following text for benchmark contamination. "
            "Determine if this text appears to be from a known benchmark dataset. "
            "Respond with 'CONTAMINATED' if it's from a benchmark, or 'CLEAN' if it's original content.\n\n"
            f"Text to analyze: {text}\n\n"
            "Analysis:"
        )
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Parse response
        is_contaminated = "CONTAMINATED" in response.upper()
        confidence = 0.8 if "CONTAMINATED" in response.upper() or "CLEAN" in response.upper() else 0.5
        
        return is_contaminated, confidence


def train_gpt_oss_contamination_model(
    train_data: List[Dict[str, Any]],
    val_data: List[Dict[str, Any]],
    model_name: str = "openai/gpt-oss-20b",
    output_dir: str = "./gpt_oss_contamination_model",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    max_length: int = 512
) -> Dict[str, Any]:
    """Train a GPT-OSS model for contamination detection."""
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    trainer = GPTOSSContaminationTrainer(
        model_name=model_name,
        max_length=max_length
    )
    
    # Train the model
    results = trainer.train(
        train_data=train_data,
        val_data=val_data,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    return results


if __name__ == "__main__":
    # Example usage
    sample_data = [
        {"text": "What is the capital of France?", "label": 1},  # Contaminated
        {"text": "How do you make a sandwich?", "label": 0},   # Clean
    ]
    
    results = train_gpt_oss_contamination_model(
        train_data=sample_data * 100,  # Duplicate for demo
        val_data=sample_data * 20,
        model_name="openai/gpt-oss-20b",
        epochs=1,
        batch_size=2
    )
    
    print(f"Training completed: {results}")
