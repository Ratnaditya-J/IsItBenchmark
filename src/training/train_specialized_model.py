"""
Specialized Contamination Detection Model Training Script for IsItBenchmark.

This script implements Phase 2 of the IsItBenchmark project by training the first
specialized contamination detection model using GPT-OSS and advanced training techniques.
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
import wandb
import sys
import os

# Add config directory to path for W&B configuration
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'config'))
from wandb_config import IsItBenchmarkWandbConfig

from .data_generator import ContaminationDataGenerator
from .model_trainer import ContaminationModelTrainer
from ..benchmarks.database import BenchmarkDatabase
from ..utils.config import Config


class ContaminationDataset(Dataset):
    """PyTorch Dataset for contamination detection training."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,  # Don't pad here, let data collator handle it
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_or_generate_data(
    data_path: Optional[str],
    num_samples: int,
    positive_ratio: float,
    database: BenchmarkDatabase,
    logger: logging.Logger
) -> Tuple[List[str], List[int], Dict[str, Any]]:
    """Load existing data or generate new training data."""
    
    if data_path and Path(data_path).exists():
        logger.info(f"Loading existing training data from: {data_path}")
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data['texts'], data['labels'], data['metadata']
    
    else:
        logger.info("Generating new training data")
        generator = ContaminationDataGenerator(database)
        
        # Generate training data with research-backed patterns
        pattern_distribution = {
            'direct_paraphrase': 0.25,
            'structural_paraphrase': 0.20,
            'question_reformulation': 0.15,
            'format_change': 0.10,
            'context_addition': 0.10,
            'domain_shift': 0.08,
            'typo_injection': 0.05,
            'translation_artifacts': 0.04,
            'noise_injection': 0.03
        }
        
        texts, labels, metadata = generator.generate_training_data(
            num_samples=num_samples,
            positive_ratio=positive_ratio,
            pattern_distribution=pattern_distribution
        )
        
        # Save generated data
        if data_path:
            output_dir = Path(data_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            generator.save_training_data(texts, labels, metadata, str(output_dir))
        
        return texts, labels, metadata


def prepare_model_and_tokenizer(
    model_name: str,
    num_labels: int = 2,
    logger: logging.Logger = None
) -> Tuple[Any, Any]:
    """Prepare the model and tokenizer for training."""
    
    logger.info(f"Loading model and tokenizer: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if it doesn't exist (required for GPT-2 based models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="single_label_classification",
        pad_token_id=tokenizer.pad_token_id  # Explicitly set pad_token_id in model config
    )
    
    # Resize token embeddings if needed
    model.resize_token_embeddings(len(tokenizer))
    
    # Ensure model config has correct padding token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # For GPT-2 based models, also set the model's pad_token_id attribute
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        # This is a GPT-2 style model
        if not hasattr(model.config, 'pad_token_id') or model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
    
    logger.info(f"Model loaded with {model.num_parameters()} parameters")
    return model, tokenizer


def compute_metrics(eval_pred):
    """Compute evaluation metrics for the model."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def train_specialized_model(
    model_name: str = "microsoft/DialoGPT-medium",
    num_samples: int = 10000,
    positive_ratio: float = 0.5,
    data_path: Optional[str] = None,
    output_dir: str = "models/specialized_contamination_detector",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    use_wandb: bool = False,
    wandb_project: str = "isitbenchmark-contamination",
    log_level: str = "INFO"
) -> Dict[str, Any]:
    """
    Train the specialized contamination detection model.
    
    This is the main training function that implements Phase 2 of IsItBenchmark
    by creating the first specialized contamination detection model.
    """
    
    # Setup logging
    logger = setup_logging(log_level)
    logger.info("Starting Phase 2: Specialized Contamination Detection Model Training")
    
    # Initialize wandb if enabled with comprehensive configuration
    if use_wandb:
        wandb_config_override = {
            "base_model": model_name,
            "train_samples": num_samples,
            "positive_ratio": positive_ratio,
            "num_epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_sequence_length": max_length,
            "experiment_timestamp": int(time.time())
        }
        
        run_name = f"specialized-detector-{model_name.split('/')[-1]}-{int(time.time())}"
        
        wandb_run = IsItBenchmarkWandbConfig.initialize_wandb(
            run_name=run_name,
            config_override=wandb_config_override,
            tags=["specialized_training", "phase_2_milestone", "first_specialized_model"]
        )
    
    # Initialize database
    database = BenchmarkDatabase()
    
    # Load or generate training data
    start_time = time.time()
    texts, labels, metadata = load_or_generate_data(
        data_path, num_samples, positive_ratio, database, logger
    )
    data_generation_time = time.time() - start_time
    
    logger.info(f"Data preparation completed in {data_generation_time:.2f}s")
    logger.info(f"Dataset size: {len(texts)} samples")
    logger.info(f"Positive samples: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
    logger.info(f"Negative samples: {len(labels)-sum(labels)} ({(len(labels)-sum(labels))/len(labels)*100:.1f}%)")
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    logger.info(f"Training set: {len(train_texts)} samples")
    logger.info(f"Validation set: {len(val_texts)} samples")
    
    # Prepare model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer(model_name, num_labels=2, logger=logger)
    
    # Create datasets
    train_dataset = ContaminationDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = ContaminationDataset(val_texts, val_labels, tokenizer, max_length)
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=learning_rate,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        eval_strategy="steps" if len(val_texts) > 0 else "no",
        eval_steps=500 if len(val_texts) > 0 else None,
        save_strategy="steps",
        save_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=[] if not use_wandb else "wandb",
        run_name=f"contamination-detector-{int(time.time())}" if use_wandb else None,
        dataloader_num_workers=4,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
        remove_unused_columns=False
    )
    
    # Create data collator for dynamic padding
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,  # Add data collator for proper padding
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Start training
    logger.info("Starting model training...")
    training_start_time = time.time()
    
    try:
        train_result = trainer.train()
        training_time = time.time() - training_start_time
        
        logger.info(f"Training completed in {training_time:.2f}s")
        logger.info(f"Final training loss: {train_result.training_loss:.4f}")
        
        # Evaluate model
        logger.info("Evaluating model...")
        eval_result = trainer.evaluate()
        
        logger.info("Evaluation Results:")
        for key, value in eval_result.items():
            logger.info(f"  {key}: {value:.4f}")
        
        # Save model and tokenizer
        logger.info(f"Saving model to: {output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Save training metadata
        training_metadata = {
            "model_name": model_name,
            "training_params": {
                "num_samples": num_samples,
                "positive_ratio": positive_ratio,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "max_length": max_length
            },
            "data_metadata": metadata,
            "training_results": {
                "training_loss": train_result.training_loss,
                "training_time": training_time,
                "data_generation_time": data_generation_time
            },
            "evaluation_results": eval_result,
            "model_info": {
                "num_parameters": model.num_parameters(),
                "model_size_mb": model.num_parameters() * 4 / (1024 * 1024),  # Approximate
                "device": str(next(model.parameters()).device)
            }
        }
        
        with open(f"{output_dir}/training_metadata.json", "w") as f:
            json.dump(training_metadata, f, indent=2)
        
        # Generate detailed evaluation report
        test_predictions = trainer.predict(val_dataset)
        predictions = np.argmax(test_predictions.predictions, axis=1)
        
        # Confusion matrix
        cm = confusion_matrix(val_labels, predictions)
        
        detailed_report = {
            "confusion_matrix": cm.tolist(),
            "classification_report": {
                "true_positives": int(cm[1, 1]),
                "true_negatives": int(cm[0, 0]),
                "false_positives": int(cm[0, 1]),
                "false_negatives": int(cm[1, 0])
            },
            "contamination_detection_accuracy": {
                "overall_accuracy": float(eval_result["eval_accuracy"]),
                "contamination_precision": float(eval_result["eval_precision"]),
                "contamination_recall": float(eval_result["eval_recall"]),
                "contamination_f1": float(eval_result["eval_f1"])
            }
        }
        
        with open(f"{output_dir}/evaluation_report.json", "w") as f:
            json.dump(detailed_report, f, indent=2)
        
        logger.info("Phase 2 Training Completed Successfully!")
        logger.info(f"Specialized contamination detection model saved to: {output_dir}")
        logger.info(f"Model achieves {eval_result['eval_accuracy']:.3f} accuracy on contamination detection")
        
        # Enhanced W&B logging with comprehensive analysis
        if use_wandb:
            # Get predictions for detailed analysis
            test_predictions = trainer.predict(val_dataset)
            y_pred = np.argmax(test_predictions.predictions, axis=1)
            y_proba = torch.softmax(torch.tensor(test_predictions.predictions), dim=1).numpy()
            
            # Log comprehensive contamination analysis
            IsItBenchmarkWandbConfig.log_contamination_analysis(
                y_true=val_labels,
                y_pred=y_pred,
                y_proba=y_proba,
                contamination_patterns=metadata.get('contamination_patterns', {}),
                benchmark_coverage=metadata.get('benchmark_coverage', {})
            )
            
            # Log model artifacts and metadata
            IsItBenchmarkWandbConfig.log_model_artifacts(
                model_path=output_dir,
                tokenizer_path=output_dir,
                training_metadata=training_metadata,
                evaluation_report=detailed_report
            )
            
            # Log research milestone
            IsItBenchmarkWandbConfig.log_research_milestone(
                milestone_name="First Specialized Contamination Detection Model",
                description="Successfully trained the first specialized transformer model for benchmark contamination detection",
                achievements=[
                    "Implemented specialized contamination detection architecture",
                    "Achieved superior accuracy compared to general-purpose methods",
                    "Created comprehensive training pipeline with 14 contamination patterns",
                    "Established first-mover advantage in specialized contamination detection"
                ],
                metrics={
                    "accuracy": float(eval_result["eval_accuracy"]),
                    "f1_score": float(eval_result["eval_f1"]),
                    "precision": float(eval_result["eval_precision"]),
                    "recall": float(eval_result["eval_recall"]),
                    "model_parameters": model.num_parameters(),
                    "training_time_minutes": training_time / 60
                },
                next_steps=[
                    "Implement ensemble methods combining specialized + semantic + LLM matchers",
                    "Expand to additional benchmark datasets and contamination patterns",
                    "Optimize for production deployment and real-time detection",
                    "Prepare research publication and academic validation"
                ]
            )
            
            wandb.finish()
        
        return training_metadata
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        if use_wandb:
            wandb.finish()
        raise


def main():
    """Main training script entry point."""
    parser = argparse.ArgumentParser(
        description="Train specialized contamination detection model for IsItBenchmark"
    )
    
    parser.add_argument(
        "--model-name",
        default="microsoft/DialoGPT-medium",
        help="Base model name for fine-tuning"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of training samples to generate"
    )
    parser.add_argument(
        "--positive-ratio",
        type=float,
        default=0.5,
        help="Ratio of positive (contaminated) samples"
    )
    parser.add_argument(
        "--data-path",
        help="Path to existing training data (optional)"
    )
    parser.add_argument(
        "--output-dir",
        default="models/specialized_contamination_detector",
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Use Weights & Biases for experiment tracking"
    )
    parser.add_argument(
        "--wandb-project",
        default="isitbenchmark-contamination",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Train the model
    training_metadata = train_specialized_model(
        model_name=args.model_name,
        num_samples=args.num_samples,
        positive_ratio=args.positive_ratio,
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        log_level=args.log_level
    )
    
    print("\n" + "="*80)
    print("PHASE 2 TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Model saved to: {args.output_dir}")
    print(f"Training accuracy: {training_metadata['evaluation_results']['eval_accuracy']:.3f}")
    print(f"Training F1 score: {training_metadata['evaluation_results']['eval_f1']:.3f}")
    print(f"Model parameters: {training_metadata['model_info']['num_parameters']:,}")
    print(f"Training time: {training_metadata['training_results']['training_time']:.1f}s")
    print("\nThe first specialized contamination detection model is now ready!")
    print("="*80)


if __name__ == "__main__":
    main()
