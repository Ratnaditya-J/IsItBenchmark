"""
Contamination Detection Model Trainer for IsItBenchmark.

This module implements the first specialized contamination detection model trainer,
providing fine-tuning capabilities for GPT-OSS and other models specifically
for benchmark contamination detection tasks.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForCausalLM, Trainer, TrainingArguments,
    DataCollatorWithPadding, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np

from ..benchmarks.database import BenchmarkDatabase
from ..utils.config import Config


class ContaminationDataset(Dataset):
    """Dataset class for contamination detection training."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        """
        Initialize contamination dataset.
        
        Args:
            texts: List of text pairs (query, candidate)
            labels: List of contamination labels (0=clean, 1=contaminated)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize the text pair
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


class ContaminationTrainer:
    """
    Specialized trainer for contamination detection models.
    
    This is the first implementation of a specialized contamination detection
    model trainer, designed to create models specifically optimized for
    detecting benchmark contamination in AI/ML systems.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the contamination trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Training configuration
        self.model_name = self.config.get("model_name", "microsoft/DialoGPT-medium")
        self.output_dir = self.config.get("output_dir", "models/contamination_detector")
        self.max_length = self.config.get("max_length", 512)
        self.batch_size = self.config.get("batch_size", 8)
        self.learning_rate = self.config.get("learning_rate", 2e-5)
        self.num_epochs = self.config.get("num_epochs", 3)
        self.warmup_steps = self.config.get("warmup_steps", 500)
        self.weight_decay = self.config.get("weight_decay", 0.01)
        self.save_steps = self.config.get("save_steps", 1000)
        self.eval_steps = self.config.get("eval_steps", 500)
        
        # Model components
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # Data
        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None
        
        # Training history
        self.training_history = {
            "train_loss": [],
            "eval_loss": [],
            "eval_accuracy": [],
            "eval_f1": [],
            "eval_auc": []
        }
        
        # Initialize database for benchmark data
        self.database = BenchmarkDatabase()
        
        self.logger.info(f"ContaminationTrainer initialized with model: {self.model_name}")
    
    def initialize_model(self) -> bool:
        """Initialize the model and tokenizer for training."""
        try:
            self.logger.info(f"Initializing model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Add special tokens if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model for sequence classification (contamination detection)
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=2,  # Binary classification: clean vs contaminated
                    trust_remote_code=True
                )
            except Exception:
                # Fallback: use causal LM and add classification head
                self.logger.info("Using causal LM with custom classification head")
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                
                # Create custom model with classification head
                self.model = ContaminationClassifier(base_model, num_labels=2)
            
            self.logger.info("Model and tokenizer initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {str(e)}")
            return False
    
    def generate_training_data(self, num_samples: int = 10000) -> Tuple[List[str], List[int]]:
        """
        Generate training data for contamination detection.
        
        Args:
            num_samples: Number of training samples to generate
            
        Returns:
            Tuple of (texts, labels) for training
        """
        self.logger.info(f"Generating {num_samples} training samples")
        
        texts = []
        labels = []
        
        # Get all benchmark questions from database
        benchmark_questions = self._get_benchmark_questions()
        
        if len(benchmark_questions) < 100:
            self.logger.warning(f"Only {len(benchmark_questions)} benchmark questions available")
        
        # Generate positive samples (contaminated)
        positive_samples = num_samples // 2
        for i in range(positive_samples):
            if i < len(benchmark_questions):
                # Use actual benchmark question
                original = benchmark_questions[i]
                contaminated = self._create_contaminated_version(original)
                
                # Create text pair for classification
                text_pair = f"Query: {contaminated} [SEP] Candidate: {original}"
                texts.append(text_pair)
                labels.append(1)  # Contaminated
            else:
                # Generate synthetic contaminated pairs
                idx1, idx2 = np.random.choice(len(benchmark_questions), 2, replace=False)
                original = benchmark_questions[idx1]
                contaminated = self._create_contaminated_version(benchmark_questions[idx2])
                
                text_pair = f"Query: {contaminated} [SEP] Candidate: {original}"
                texts.append(text_pair)
                labels.append(1)  # Contaminated
        
        # Generate negative samples (clean)
        negative_samples = num_samples - positive_samples
        for i in range(negative_samples):
            # Create unrelated text pairs
            idx1, idx2 = np.random.choice(len(benchmark_questions), 2, replace=False)
            text1 = benchmark_questions[idx1]
            text2 = self._create_unrelated_text(benchmark_questions[idx2])
            
            text_pair = f"Query: {text2} [SEP] Candidate: {text1}"
            texts.append(text_pair)
            labels.append(0)  # Clean
        
        self.logger.info(f"Generated {len(texts)} training samples")
        return texts, labels
    
    def _get_benchmark_questions(self) -> List[str]:
        """Get all benchmark questions from the database."""
        try:
            conn = self.database._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT question_text FROM benchmark_questions LIMIT 50000")
            questions = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            return questions
            
        except Exception as e:
            self.logger.error(f"Failed to get benchmark questions: {str(e)}")
            return []
    
    def _create_contaminated_version(self, original: str) -> str:
        """Create a contaminated version of a benchmark question."""
        contamination_strategies = [
            self._paraphrase_question,
            self._add_context,
            self._change_format,
            self._translate_back,
            self._add_noise
        ]
        
        strategy = np.random.choice(contamination_strategies)
        return strategy(original)
    
    def _paraphrase_question(self, text: str) -> str:
        """Create a paraphrased version of the question."""
        # Simple paraphrasing strategies
        paraphrases = [
            lambda x: x.replace("What is", "Can you tell me what"),
            lambda x: x.replace("How", "In what way"),
            lambda x: x.replace("Why", "What is the reason"),
            lambda x: f"Please explain: {x}",
            lambda x: f"I need to know: {x}",
            lambda x: x.replace("?", " exactly?")
        ]
        
        strategy = np.random.choice(paraphrases)
        return strategy(text)
    
    def _add_context(self, text: str) -> str:
        """Add contextual information to the question."""
        contexts = [
            f"For my research project, {text}",
            f"I'm studying this topic: {text}",
            f"Can you help me understand: {text}",
            f"In the context of machine learning, {text}",
            f"From an academic perspective, {text}"
        ]
        
        return np.random.choice(contexts)
    
    def _change_format(self, text: str) -> str:
        """Change the format of the question."""
        if "?" not in text:
            text += "?"
        
        formats = [
            f"Question: {text}",
            f"Q: {text}",
            f"Query: {text}",
            f"Problem: {text.replace('?', '.')}"
        ]
        
        return np.random.choice(formats)
    
    def _translate_back(self, text: str) -> str:
        """Simulate translation artifacts."""
        # Simple translation-like modifications
        modifications = [
            lambda x: x.replace("the", "a"),
            lambda x: x.replace("is", "was"),
            lambda x: x.replace("are", "were"),
            lambda x: x.replace("will", "would"),
            lambda x: x.replace("can", "could")
        ]
        
        strategy = np.random.choice(modifications)
        return strategy(text)
    
    def _add_noise(self, text: str) -> str:
        """Add subtle noise to the question."""
        # Add minor typos or formatting changes
        noise_strategies = [
            lambda x: x.replace(" ", "  "),  # Double spaces
            lambda x: x.lower(),
            lambda x: x.upper(),
            lambda x: x.replace(".", " ."),
            lambda x: x.replace(",", " ,")
        ]
        
        strategy = np.random.choice(noise_strategies)
        return strategy(text)
    
    def _create_unrelated_text(self, base_text: str) -> str:
        """Create unrelated text for negative samples."""
        unrelated_templates = [
            "What's the weather like today?",
            "How do I cook pasta?",
            "What time is it?",
            "Where is the nearest store?",
            "How to fix a computer?",
            "What's the capital of France?",
            "How to learn programming?",
            "What is machine learning?",
            "How to write a resume?",
            "What are the benefits of exercise?"
        ]
        
        return np.random.choice(unrelated_templates)
    
    def prepare_datasets(self, texts: List[str], labels: List[int], test_size: float = 0.2, val_size: float = 0.1):
        """Prepare training, validation, and test datasets."""
        self.logger.info("Preparing datasets for training")
        
        # Split data
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels, test_size=val_size, random_state=42, stratify=train_labels
        )
        
        # Create datasets
        self.train_dataset = ContaminationDataset(
            train_texts, train_labels, self.tokenizer, self.max_length
        )
        self.eval_dataset = ContaminationDataset(
            val_texts, val_labels, self.tokenizer, self.max_length
        )
        self.test_dataset = ContaminationDataset(
            test_texts, test_labels, self.tokenizer, self.max_length
        )
        
        self.logger.info(f"Dataset sizes - Train: {len(self.train_dataset)}, "
                        f"Val: {len(self.eval_dataset)}, Test: {len(self.test_dataset)}")
    
    def train(self) -> Dict[str, Any]:
        """Train the contamination detection model."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")
        
        if not self.train_dataset or not self.eval_dataset:
            raise RuntimeError("Datasets not prepared. Call prepare_datasets() first.")
        
        self.logger.info("Starting contamination detection model training")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            learning_rate=self.learning_rate,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=100,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            report_to=None,  # Disable wandb/tensorboard
            seed=42,
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train the model
        start_time = time.time()
        train_result = self.trainer.train()
        training_time = time.time() - start_time
        
        # Save the model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Evaluate on test set
        test_results = self.trainer.evaluate(eval_dataset=self.test_dataset)
        
        # Compile results
        results = {
            "training_time": training_time,
            "train_loss": train_result.training_loss,
            "test_results": test_results,
            "model_path": self.output_dir,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "num_trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        # Save training results
        with open(f"{self.output_dir}/training_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        self.logger.info(f"Model saved to: {self.output_dir}")
        
        return results
    
    def _compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Compute metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        
        try:
            auc = roc_auc_score(labels, predictions)
        except ValueError:
            auc = 0.0  # Handle case where only one class is present
        
        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "auc": auc
        }
    
    def evaluate_model(self, test_data: Optional[Tuple[List[str], List[int]]] = None) -> Dict[str, Any]:
        """Evaluate the trained model."""
        if not self.trainer:
            raise RuntimeError("Model not trained. Call train() first.")
        
        if test_data:
            # Use provided test data
            test_texts, test_labels = test_data
            test_dataset = ContaminationDataset(
                test_texts, test_labels, self.tokenizer, self.max_length
            )
        else:
            # Use prepared test dataset
            test_dataset = self.test_dataset
        
        if not test_dataset:
            raise RuntimeError("No test dataset available")
        
        self.logger.info("Evaluating contamination detection model")
        
        # Evaluate
        results = self.trainer.evaluate(eval_dataset=test_dataset)
        
        self.logger.info(f"Evaluation results: {results}")
        return results
    
    def save_specialized_model(self, model_name: str = "specialized_contamination_detector"):
        """Save the specialized contamination detection model."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("No trained model to save")
        
        save_path = f"models/{model_name}"
        os.makedirs(save_path, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save configuration
        config = {
            "model_type": "specialized_contamination_detector",
            "base_model": self.model_name,
            "training_config": self.config,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": "First specialized contamination detection model for IsItBenchmark"
        }
        
        with open(f"{save_path}/model_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Specialized model saved to: {save_path}")
        return save_path


class ContaminationClassifier(nn.Module):
    """Custom contamination classifier with pre-trained base model."""
    
    def __init__(self, base_model, num_labels: int = 2):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        
        # Get hidden size from base model
        hidden_size = base_model.config.hidden_size
        
        # Add classification head
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use last hidden state for classification
        hidden_states = outputs.hidden_states[-1]
        pooled_output = hidden_states[:, -1, :]  # Use last token
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.hidden_states,
        }
