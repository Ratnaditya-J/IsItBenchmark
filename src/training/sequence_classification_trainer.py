"""
Sequence Classification Trainer for Contamination Detection

Trains BERT-style models for binary contamination detection using sequence classification.
Optimized for CPU training with various transformer architectures.
"""

import logging
import os
from typing import List, Dict, Any, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

logger = logging.getLogger(__name__)

class ContaminationDataset(Dataset):
    """Dataset for contamination detection training."""
    
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
        
        # Tokenize the text
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

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_sequence_classification_model(
    train_data: List[Dict[str, Any]],
    val_data: List[Dict[str, Any]],
    model_name: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    warmup_steps: int = 500,
    weight_decay: float = 0.01,
    save_steps: int = 1000,
    eval_steps: int = 500,
    logging_steps: int = 100
) -> Dict[str, Any]:
    """
    Train a sequence classification model for contamination detection.
    
    Args:
        train_data: Training data as list of {"text": str, "label": int}
        val_data: Validation data as list of {"text": str, "label": int}
        model_name: HuggingFace model identifier
        output_dir: Directory to save the trained model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        max_length: Maximum sequence length
        warmup_steps: Number of warmup steps
        weight_decay: Weight decay for regularization
        save_steps: Steps between model saves
        eval_steps: Steps between evaluations
        logging_steps: Steps between logging
    
    Returns:
        Dictionary with training results and metrics
    """
    logger.info(f"Loading model and tokenizer: {model_name}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,  # Binary classification
        id2label={0: "clean", 1: "contaminated"},
        label2id={"clean": 0, "contaminated": 1}
    )
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    logger.info("Preparing datasets...")
    
    # Extract texts and labels
    train_texts = [item["text"] for item in train_data]
    train_labels = [item["label"] for item in train_data]
    val_texts = [item["text"] for item in val_data]
    val_labels = [item["label"] for item in val_data]
    
    # Create datasets
    train_dataset = ContaminationDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = ContaminationDataset(val_texts, val_labels, tokenizer, max_length)
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir=f"{output_dir}/logs",
        logging_steps=logging_steps,
        eval_strategy="steps",  # Updated parameter name
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        learning_rate=learning_rate,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=None,  # Disable W&B integration
        dataloader_num_workers=0,  # Avoid multiprocessing issues on some systems
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    logger.info("Starting training...")
    
    # Train the model
    train_result = trainer.train()
    
    logger.info("Training completed. Running final evaluation...")
    
    # Final evaluation
    eval_result = trainer.evaluate()
    
    # Save the model and tokenizer
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Prepare results
    results = {
        "train_loss": train_result.training_loss,
        "train_runtime": train_result.metrics["train_runtime"],
        "train_samples_per_second": train_result.metrics["train_samples_per_second"],
        "evaluation_results": eval_result,
        "model_path": output_dir,
        "model_name": model_name,
        "training_args": training_args.to_dict()
    }
    
    logger.info("Training Results:")
    logger.info(f"  Training Loss: {results['train_loss']:.4f}")
    logger.info(f"  Validation Accuracy: {eval_result['eval_accuracy']:.4f}")
    logger.info(f"  Validation F1: {eval_result['eval_f1']:.4f}")
    logger.info(f"  Validation Precision: {eval_result['eval_precision']:.4f}")
    logger.info(f"  Validation Recall: {eval_result['eval_recall']:.4f}")
    
    return results
