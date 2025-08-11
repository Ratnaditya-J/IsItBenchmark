"""
Model trainer for specialized contamination detection.

This module provides the ContaminationModelTrainer class for fine-tuning
models specifically for benchmark contamination detection tasks.
"""

import os
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EvalPrediction
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class ContaminationDataset(Dataset):
    """Dataset class for contamination detection training data."""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 512
    ):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text samples
            labels: List of binary labels (0=clean, 1=contaminated)
            tokenizer: Tokenizer for text encoding
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
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


class ContaminationModelTrainer:
    """Trainer for specialized contamination detection models."""
    
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        output_dir: str = "models/specialized_contamination_detector",
        max_length: int = 512,
        use_wandb: bool = False,
        wandb_project: str = "isitbenchmark-contamination",
        log_level: str = "INFO"
    ):
        """
        Initialize the model trainer.
        
        Args:
            model_name: Base model name for fine-tuning
            output_dir: Directory to save the trained model
            max_length: Maximum sequence length
            use_wandb: Whether to use Weights & Biases for logging
            wandb_project: W&B project name
            log_level: Logging level
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.max_length = max_length
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_project = wandb_project
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, log_level.upper()))
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # Training metadata
        self.training_metadata = {
            "model_name": model_name,
            "output_dir": str(output_dir),
            "max_length": max_length,
            "training_start_time": None,
            "training_end_time": None,
            "training_time": None,
            "model_info": {},
            "training_results": {},
            "evaluation_results": {},
            "hyperparameters": {}
        }
    
    def initialize_model(self) -> bool:
        """
        Initialize the tokenizer and model.
        
        Returns:
            True if initialization was successful
        """
        try:
            self.logger.info(f"Initializing model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model for binary classification
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2,  # Binary classification: clean vs contaminated
                problem_type="single_label_classification"
            )
            
            # Resize token embeddings if needed
            self.model.resize_token_embeddings(len(self.tokenizer))
            
            # Store model info
            self.training_metadata["model_info"] = {
                "model_name": self.model_name,
                "num_parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "vocab_size": len(self.tokenizer),
                "max_length": self.max_length
            }
            
            self.logger.info(f"Model initialized successfully")
            self.logger.info(f"Total parameters: {self.training_metadata['model_info']['num_parameters']:,}")
            self.logger.info(f"Trainable parameters: {self.training_metadata['model_info']['trainable_parameters']:,}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {str(e)}")
            return False
    
    def prepare_datasets(
        self,
        train_texts: List[str],
        train_labels: List[int],
        eval_texts: Optional[List[str]] = None,
        eval_labels: Optional[List[int]] = None
    ) -> Tuple[ContaminationDataset, Optional[ContaminationDataset]]:
        """
        Prepare training and evaluation datasets.
        
        Args:
            train_texts: Training text samples
            train_labels: Training labels
            eval_texts: Evaluation text samples (optional)
            eval_labels: Evaluation labels (optional)
            
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        self.logger.info(f"Preparing datasets...")
        self.logger.info(f"Training samples: {len(train_texts)}")
        
        # Create training dataset
        train_dataset = ContaminationDataset(
            texts=train_texts,
            labels=train_labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        # Create evaluation dataset if provided
        eval_dataset = None
        if eval_texts is not None and eval_labels is not None:
            self.logger.info(f"Evaluation samples: {len(eval_texts)}")
            eval_dataset = ContaminationDataset(
                texts=eval_texts,
                labels=eval_labels,
                tokenizer=self.tokenizer,
                max_length=self.max_length
            )
        
        return train_dataset, eval_dataset
    
    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            eval_pred: Evaluation predictions
            
        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Compute metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        return {
            'eval_accuracy': accuracy,
            'eval_precision': precision,
            'eval_recall': recall,
            'eval_f1': f1,
            'eval_true_negatives': int(tn),
            'eval_false_positives': int(fp),
            'eval_false_negatives': int(fn),
            'eval_true_positives': int(tp)
        }
    
    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        eval_texts: Optional[List[str]] = None,
        eval_labels: Optional[List[int]] = None,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 100
    ) -> Dict[str, Any]:
        """
        Train the contamination detection model.
        
        Args:
            train_texts: Training text samples
            train_labels: Training labels
            eval_texts: Evaluation text samples (optional)
            eval_labels: Evaluation labels (optional)
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            weight_decay: Weight decay for regularization
            save_steps: Steps between model saves
            eval_steps: Steps between evaluations
            logging_steps: Steps between logging
            
        Returns:
            Training metadata and results
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")
        
        self.logger.info("Starting model training...")
        self.training_metadata["training_start_time"] = time.time()
        
        # Store hyperparameters
        self.training_metadata["hyperparameters"] = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps,
            "weight_decay": weight_decay,
            "max_length": self.max_length
        }
        
        # Initialize W&B if enabled
        if self.use_wandb:
            wandb.init(
                project=self.wandb_project,
                config=self.training_metadata["hyperparameters"],
                name=f"contamination-detection-{int(time.time())}"
            )
        
        # Prepare datasets
        train_dataset, eval_dataset = self.prepare_datasets(
            train_texts, train_labels, eval_texts, eval_labels
        )
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps if eval_dataset else None,
            eval_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_f1" if eval_dataset else None,
            greater_is_better=True,
            report_to="wandb" if self.use_wandb else None,
            run_name=f"contamination-detection-{int(time.time())}" if self.use_wandb else None,
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics if eval_dataset else None,
        )
        
        # Train the model
        self.logger.info("Training started...")
        train_result = self.trainer.train()
        
        # Record training completion
        self.training_metadata["training_end_time"] = time.time()
        self.training_metadata["training_time"] = (
            self.training_metadata["training_end_time"] - 
            self.training_metadata["training_start_time"]
        )
        
        # Store training results
        self.training_metadata["training_results"] = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            "train_steps_per_second": train_result.metrics.get("train_steps_per_second", 0),
            "training_time": self.training_metadata["training_time"]
        }
        
        # Evaluate if evaluation dataset is provided
        if eval_dataset:
            self.logger.info("Running final evaluation...")
            eval_result = self.trainer.evaluate()
            self.training_metadata["evaluation_results"] = eval_result
            
            self.logger.info(f"Final evaluation results:")
            self.logger.info(f"  Accuracy: {eval_result.get('eval_accuracy', 0):.3f}")
            self.logger.info(f"  Precision: {eval_result.get('eval_precision', 0):.3f}")
            self.logger.info(f"  Recall: {eval_result.get('eval_recall', 0):.3f}")
            self.logger.info(f"  F1 Score: {eval_result.get('eval_f1', 0):.3f}")
        
        # Save the model and tokenizer
        self.logger.info(f"Saving model to {self.output_dir}")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Save training metadata
        metadata_path = self.output_dir / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.training_metadata, f, indent=2, default=str)
        
        self.logger.info("Training completed successfully!")
        
        # Finish W&B run
        if self.use_wandb:
            wandb.finish()
        
        return self.training_metadata
    
    def save_model(self, save_path: Optional[str] = None) -> str:
        """
        Save the trained model.
        
        Args:
            save_path: Optional custom save path
            
        Returns:
            Path where the model was saved
        """
        if not self.trainer:
            raise RuntimeError("No trained model to save")
        
        save_path = save_path or str(self.output_dir)
        self.trainer.save_model(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        self.logger.info(f"Model saved to {save_path}")
        return save_path
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a previously trained model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            True if loading was successful
        """
        try:
            self.logger.info(f"Loading model from {model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
            self.logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            return False
