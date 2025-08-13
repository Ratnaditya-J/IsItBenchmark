"""
Mac-Optimized Contamination Detection Trainer for Apple Silicon

This trainer is specifically designed for Mac Mini M4 and other Apple Silicon devices,
using MPS (Metal Performance Shaders) backend and models compatible with Apple's GPU architecture.
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    DataCollatorWithPadding
)
from torch.utils.data import Dataset
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'config'))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContaminationDataset(Dataset):
    """PyTorch Dataset for contamination detection training on Mac."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class MacOptimizedContaminationTrainer:
    """Mac-optimized contamination detection trainer using Apple Silicon GPU."""
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",  # Mac-compatible model
        max_length: int = 512,
        num_labels: int = 2,
        device: str = "auto"
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.num_labels = num_labels
        
        # Detect best device for Mac
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
                logger.info("Using MPS (Metal Performance Shaders) for Mac GPU acceleration")
            elif torch.cuda.is_available():
                self.device = "cuda"
                logger.info("Using CUDA GPU")
            else:
                self.device = "cpu"
                logger.info("Using CPU")
        else:
            self.device = device
        
        # Initialize tokenizer and model
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            torch_dtype=torch.float32,  # MPS works best with float32
        )
        
        # Move model to device
        self.model.to(self.device)
        
        # Set pad token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_data(self, data: List[Dict[str, Any]]) -> Tuple[List[str], List[int]]:
        """Prepare training data from contamination dataset."""
        texts = []
        labels = []
        
        for item in data:
            if isinstance(item, dict):
                text = item.get('text', '')
                label = item.get('label', 0)
            else:
                # Handle other formats
                text = str(item)
                label = 0  # Default to clean
            
            texts.append(text)
            labels.append(label)
        
        return texts, labels
    
    def train(
        self,
        train_data: List[Dict[str, Any]],
        val_data: List[Dict[str, Any]],
        output_dir: str,
        epochs: int = 3,
        batch_size: int = 8,  # Optimized for Mac Mini M4
        learning_rate: float = 2e-5,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        save_steps: int = 500,
        eval_steps: int = 500
    ) -> Dict[str, Any]:
        """Train the contamination detection model on Mac."""
        
        logger.info(f"Starting Mac-optimized contamination detection training")
        logger.info(f"Device: {self.device}")
        logger.info(f"Training samples: {len(train_data)}")
        logger.info(f"Validation samples: {len(val_data)}")
        
        # Prepare data
        train_texts, train_labels = self.prepare_data(train_data)
        val_texts, val_labels = self.prepare_data(val_data)
        
        # Create datasets
        train_dataset = ContaminationDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = ContaminationDataset(val_texts, val_labels, self.tokenizer, self.max_length)
        
        # Create data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Training arguments optimized for Mac
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
            eval_strategy="steps",  # Updated parameter name
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            learning_rate=learning_rate,
            fp16=False,  # Use fp32 for MPS compatibility
            dataloader_pin_memory=False,  # Better for Mac
            remove_unused_columns=False,
            report_to=[],  # Disable W&B for now
        )
        
        # Custom compute metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            
            accuracy = accuracy_score(labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
            
            return {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train the model
        logger.info("Starting training...")
        start_time = time.time()
        train_result = trainer.train()
        training_time = time.time() - start_time
        
        # Evaluate the model
        logger.info("Evaluating model...")
        eval_result = trainer.evaluate()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training configuration
        config = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "device": self.device,
            "training_args": training_args.to_dict(),
            "training_time": training_time,
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "final_metrics": eval_result
        }
        
        with open(f"{output_dir}/training_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Final training loss: {train_result.training_loss:.4f}")
        logger.info(f"Final validation accuracy: {eval_result.get('eval_accuracy', 0):.4f}")
        
        return {
            "training_loss": train_result.training_loss,
            "eval_accuracy": eval_result.get('eval_accuracy', 0),
            "eval_f1": eval_result.get('eval_f1', 0),
            "training_time": training_time,
            "model_path": output_dir,
            "device_used": self.device
        }
    
    def predict_contamination(self, text: str) -> Tuple[bool, float]:
        """Predict if text is contaminated using the trained model."""
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Get prediction and confidence
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = torch.max(predictions).item()
        
        is_contaminated = bool(predicted_class == 1)
        
        return is_contaminated, confidence


def train_mac_contamination_model(
    train_data: List[Dict[str, Any]],
    val_data: List[Dict[str, Any]],
    model_name: str = "distilbert-base-uncased",
    output_dir: str = "./mac_contamination_model",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 512
) -> Dict[str, Any]:
    """Train a Mac-optimized contamination detection model."""
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    trainer = MacOptimizedContaminationTrainer(
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
    # Example usage with sample data
    sample_data = [
        {"text": "What is the capital of France?", "label": 1},  # Contaminated (common benchmark question)
        {"text": "Paris is the capital of France.", "label": 1},  # Contaminated (benchmark answer)
        {"text": "How do you make a sandwich?", "label": 0},   # Clean
        {"text": "I love programming in Python.", "label": 0},   # Clean
        {"text": "What is 2 + 2?", "label": 1},  # Contaminated (basic math benchmark)
        {"text": "The weather is nice today.", "label": 0},   # Clean
    ]
    
    # Split data
    train_data = sample_data * 50  # Duplicate for demo
    val_data = sample_data * 10
    
    print("Starting Mac-optimized contamination detection training...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    results = train_mac_contamination_model(
        train_data=train_data,
        val_data=val_data,
        model_name="distilbert-base-uncased",  # Fast and Mac-compatible
        output_dir="./models/mac_contamination_detector",
        epochs=2,
        batch_size=8
    )
    
    print(f"Training completed successfully!")
    print(f"Results: {results}")
