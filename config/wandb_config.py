#!/usr/bin/env python3
"""
Weights & Biases Configuration for IsItBenchmark Project
Comprehensive tracking for contamination detection model training and evaluation.
"""

import wandb
import torch
import numpy as np
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
import time


class IsItBenchmarkWandbConfig:
    """
    Comprehensive W&B configuration for IsItBenchmark contamination detection project.
    """
    
    # Project Configuration
    PROJECT_NAME = "isitbenchmark-contamination-detection"
    ENTITY = None  # Set to your W&B username/team
    
    # Ideal Configuration Parameters
    IDEAL_CONFIG = {
        # Project Metadata
        "project_phase": "phase_2_specialized_training",
        "experiment_type": "contamination_detection",
        "model_architecture": "transformer_classification",
        
        # Model Configuration
        "base_model": "microsoft/DialoGPT-medium",
        "model_type": "specialized_contamination_detector",
        "num_labels": 2,
        "max_sequence_length": 512,
        "dropout_rate": 0.1,
        
        # Training Hyperparameters
        "learning_rate": 2e-5,
        "batch_size": 16,
        "num_epochs": 5,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,
        
        # Data Configuration
        "train_samples": 10000,
        "validation_split": 0.2,
        "positive_ratio": 0.5,
        "data_augmentation": True,
        "contamination_patterns": 14,
        
        # Optimization Settings
        "optimizer": "adamw",
        "lr_scheduler": "linear_with_warmup",
        "early_stopping_patience": 3,
        "save_strategy": "best_model",
        
        # Hardware Configuration
        "device": "auto",
        "mixed_precision": True,
        "gradient_checkpointing": True,
        "dataloader_workers": 4,
        
        # Evaluation Metrics
        "primary_metric": "f1_score",
        "secondary_metrics": ["accuracy", "precision", "recall", "auc"],
        "contamination_threshold": 0.5,
        
        # Research Configuration
        "benchmark_datasets": 9,
        "total_benchmark_questions": 49159,
        "detection_methods": ["semantic", "llm", "ngram", "membership_inference", "specialized"],
        "research_focus": "first_specialized_contamination_model",
        
        # Experiment Tags
        "tags": [
            "contamination_detection",
            "benchmark_analysis", 
            "specialized_model",
            "phase_2",
            "transformer_finetuning",
            "research_innovation"
        ]
    }
    
    @classmethod
    def initialize_wandb(
        cls,
        run_name: Optional[str] = None,
        config_override: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> wandb.run:
        """
        Initialize W&B run with comprehensive configuration.
        
        Args:
            run_name: Custom run name (auto-generated if None)
            config_override: Override default config parameters
            tags: Additional tags for the run
            
        Returns:
            wandb.run: Initialized W&B run
        """
        
        # Generate run name if not provided
        if run_name is None:
            timestamp = int(time.time())
            run_name = f"contamination-detector-{timestamp}"
        
        # Merge configurations
        config = cls.IDEAL_CONFIG.copy()
        if config_override:
            config.update(config_override)
        
        # Merge tags
        run_tags = config["tags"].copy()
        if tags:
            run_tags.extend(tags)
        
        # Initialize W&B
        run = wandb.init(
            project=cls.PROJECT_NAME,
            entity=cls.ENTITY,
            name=run_name,
            config=config,
            tags=run_tags,
            notes="Specialized contamination detection model training for IsItBenchmark Phase 2",
            save_code=True,
            resume="allow"
        )
        
        # Log system information
        cls._log_system_info()
        
        return run
    
    @classmethod
    def _log_system_info(cls):
        """Log system and environment information."""
        system_info = {
            "python_version": f"{torch.__version__}",
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
        }
        
        wandb.log({"system_info": system_info})
    
    @classmethod
    def log_training_metrics(
        cls,
        epoch: int,
        train_loss: float,
        train_accuracy: float,
        val_loss: Optional[float] = None,
        val_accuracy: Optional[float] = None,
        val_f1: Optional[float] = None,
        val_precision: Optional[float] = None,
        val_recall: Optional[float] = None,
        learning_rate: Optional[float] = None,
        step: Optional[int] = None
    ):
        """
        Log comprehensive training metrics to W&B.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            train_accuracy: Training accuracy
            val_loss: Validation loss (optional)
            val_accuracy: Validation accuracy (optional)
            val_f1: Validation F1 score (optional)
            val_precision: Validation precision (optional)
            val_recall: Validation recall (optional)
            learning_rate: Current learning rate (optional)
            step: Current training step (optional)
        """
        
        metrics = {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_accuracy,
        }
        
        if val_loss is not None:
            metrics["val/loss"] = val_loss
        if val_accuracy is not None:
            metrics["val/accuracy"] = val_accuracy
        if val_f1 is not None:
            metrics["val/f1_score"] = val_f1
        if val_precision is not None:
            metrics["val/precision"] = val_precision
        if val_recall is not None:
            metrics["val/recall"] = val_recall
        if learning_rate is not None:
            metrics["train/learning_rate"] = learning_rate
        
        wandb.log(metrics, step=step)
    
    @classmethod
    def log_contamination_analysis(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        contamination_patterns: Optional[Dict[str, int]] = None,
        benchmark_coverage: Optional[Dict[str, float]] = None
    ):
        """
        Log contamination-specific analysis metrics and visualizations.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            contamination_patterns: Count of different contamination patterns detected
            benchmark_coverage: Coverage analysis across different benchmarks
        """
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Clean', 'Contaminated'],
                   yticklabels=['Clean', 'Contaminated'])
        plt.title('Contamination Detection Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Log confusion matrix
        wandb.log({"confusion_matrix": wandb.Image(plt)})
        plt.close()
        
        # Classification metrics
        report = classification_report(y_true, y_pred, output_dict=True)
        
        contamination_metrics = {
            "contamination/true_positives": int(cm[1, 1]),
            "contamination/true_negatives": int(cm[0, 0]),
            "contamination/false_positives": int(cm[0, 1]),
            "contamination/false_negatives": int(cm[1, 0]),
            "contamination/sensitivity": report['1']['recall'],
            "contamination/specificity": report['0']['recall'],
            "contamination/precision": report['1']['precision'],
            "contamination/f1_score": report['1']['f1-score'],
            "contamination/accuracy": report['accuracy']
        }
        
        # Log probability distribution if available
        if y_proba is not None:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.hist(y_proba[y_true == 0, 1], bins=30, alpha=0.7, label='Clean', color='blue')
            plt.hist(y_proba[y_true == 1, 1], bins=30, alpha=0.7, label='Contaminated', color='red')
            plt.xlabel('Contamination Probability')
            plt.ylabel('Frequency')
            plt.title('Probability Distribution by True Label')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.scatter(range(len(y_proba)), y_proba[:, 1], 
                       c=y_true, cmap='coolwarm', alpha=0.6)
            plt.xlabel('Sample Index')
            plt.ylabel('Contamination Probability')
            plt.title('Contamination Probability vs True Labels')
            plt.colorbar(label='True Label')
            
            plt.tight_layout()
            wandb.log({"probability_analysis": wandb.Image(plt)})
            plt.close()
            
            # Add probability-based metrics
            contamination_metrics.update({
                "contamination/mean_prob_clean": float(np.mean(y_proba[y_true == 0, 1])),
                "contamination/mean_prob_contaminated": float(np.mean(y_proba[y_true == 1, 1])),
                "contamination/prob_separation": float(np.mean(y_proba[y_true == 1, 1]) - np.mean(y_proba[y_true == 0, 1]))
            })
        
        # Log contamination pattern analysis
        if contamination_patterns:
            pattern_metrics = {f"patterns/{k}": v for k, v in contamination_patterns.items()}
            contamination_metrics.update(pattern_metrics)
            
            # Create pattern distribution plot
            plt.figure(figsize=(10, 6))
            patterns = list(contamination_patterns.keys())
            counts = list(contamination_patterns.values())
            
            plt.bar(patterns, counts, color='skyblue', edgecolor='navy')
            plt.title('Contamination Pattern Distribution')
            plt.xlabel('Pattern Type')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            wandb.log({"contamination_patterns": wandb.Image(plt)})
            plt.close()
        
        # Log benchmark coverage analysis
        if benchmark_coverage:
            coverage_metrics = {f"coverage/{k}": v for k, v in benchmark_coverage.items()}
            contamination_metrics.update(coverage_metrics)
            
            # Create coverage visualization
            plt.figure(figsize=(12, 6))
            benchmarks = list(benchmark_coverage.keys())
            coverage = list(benchmark_coverage.values())
            
            plt.bar(benchmarks, coverage, color='lightgreen', edgecolor='darkgreen')
            plt.title('Benchmark Coverage Analysis')
            plt.xlabel('Benchmark Dataset')
            plt.ylabel('Coverage Percentage')
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 100)
            
            for i, v in enumerate(coverage):
                plt.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            wandb.log({"benchmark_coverage": wandb.Image(plt)})
            plt.close()
        
        wandb.log(contamination_metrics)
    
    @classmethod
    def log_model_artifacts(
        cls,
        model_path: str,
        tokenizer_path: str,
        training_metadata: Dict[str, Any],
        evaluation_report: Dict[str, Any]
    ):
        """
        Log model artifacts and metadata to W&B.
        
        Args:
            model_path: Path to saved model
            tokenizer_path: Path to saved tokenizer
            training_metadata: Training metadata dictionary
            evaluation_report: Evaluation report dictionary
        """
        
        # Create model artifact
        model_artifact = wandb.Artifact(
            name="specialized-contamination-detector",
            type="model",
            description="Specialized transformer model for benchmark contamination detection",
            metadata={
                "model_type": "specialized_contamination_detector",
                "architecture": "transformer_classification",
                "training_samples": training_metadata.get("training_params", {}).get("num_samples"),
                "accuracy": evaluation_report.get("contamination_detection_accuracy", {}).get("overall_accuracy"),
                "f1_score": evaluation_report.get("contamination_detection_accuracy", {}).get("contamination_f1")
            }
        )
        
        # Add model files
        model_artifact.add_dir(model_path, name="model")
        
        # Create training data artifact
        training_artifact = wandb.Artifact(
            name="training-metadata",
            type="dataset",
            description="Training metadata and evaluation results",
            metadata=training_metadata
        )
        
        # Save metadata as JSON
        with open("training_metadata.json", "w") as f:
            json.dump(training_metadata, f, indent=2)
        with open("evaluation_report.json", "w") as f:
            json.dump(evaluation_report, f, indent=2)
        
        training_artifact.add_file("training_metadata.json")
        training_artifact.add_file("evaluation_report.json")
        
        # Log artifacts
        wandb.log_artifact(model_artifact)
        wandb.log_artifact(training_artifact)
        
        # Log final summary metrics
        final_metrics = {
            "final/model_parameters": training_metadata.get("model_info", {}).get("num_parameters"),
            "final/model_size_mb": training_metadata.get("model_info", {}).get("model_size_mb"),
            "final/training_time": training_metadata.get("training_results", {}).get("training_time"),
            "final/data_generation_time": training_metadata.get("training_results", {}).get("data_generation_time"),
            "final/accuracy": evaluation_report.get("contamination_detection_accuracy", {}).get("overall_accuracy"),
            "final/precision": evaluation_report.get("contamination_detection_accuracy", {}).get("contamination_precision"),
            "final/recall": evaluation_report.get("contamination_detection_accuracy", {}).get("contamination_recall"),
            "final/f1_score": evaluation_report.get("contamination_detection_accuracy", {}).get("contamination_f1")
        }
        
        wandb.log(final_metrics)
    
    @classmethod
    def log_research_milestone(
        cls,
        milestone_name: str,
        description: str,
        achievements: List[str],
        metrics: Dict[str, float],
        next_steps: List[str]
    ):
        """
        Log research milestones and achievements.
        
        Args:
            milestone_name: Name of the milestone
            description: Description of the achievement
            achievements: List of key achievements
            metrics: Key performance metrics
            next_steps: Planned next steps
        """
        
        milestone_data = {
            "research/milestone": milestone_name,
            "research/description": description,
            "research/achievements": achievements,
            "research/next_steps": next_steps
        }
        
        # Add metrics with research prefix
        for key, value in metrics.items():
            milestone_data[f"research/{key}"] = value
        
        wandb.log(milestone_data)
        
        # Create milestone summary table
        milestone_table = wandb.Table(
            columns=["Milestone", "Achievement", "Metric", "Value"],
            data=[]
        )
        
        for achievement in achievements:
            milestone_table.add_data(milestone_name, achievement, "", "")
        
        for metric, value in metrics.items():
            milestone_table.add_data(milestone_name, "", metric, value)
        
        wandb.log({"research_milestones": milestone_table})


# Example usage configuration
EXAMPLE_TRAINING_CONFIG = {
    "model_name": "microsoft/DialoGPT-medium",
    "num_samples": 10000,
    "positive_ratio": 0.5,
    "epochs": 5,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "max_length": 512,
    "use_wandb": True,
    "wandb_project": "isitbenchmark-contamination-detection"
}

# Ideal hyperparameter sweep configuration
HYPERPARAMETER_SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {
        "name": "val/f1_score",
        "goal": "maximize"
    },
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-3
        },
        "batch_size": {
            "values": [8, 16, 32, 64]
        },
        "num_epochs": {
            "values": [3, 5, 7, 10]
        },
        "weight_decay": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 0.1
        },
        "warmup_steps": {
            "values": [100, 500, 1000]
        },
        "positive_ratio": {
            "distribution": "uniform",
            "min": 0.3,
            "max": 0.7
        }
    }
}
