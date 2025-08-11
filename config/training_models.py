"""
Training Model Configuration for IsItBenchmark

Provides CPU-compatible model options for specialized contamination detection training.
"""

from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for a training model."""
    name: str
    model_id: str
    description: str
    parameters: str
    cpu_optimized: bool
    recommended_batch_size: int
    max_length: int
    training_type: str  # "classification" or "generative"

class TrainingModelRegistry:
    """Registry of available CPU-compatible models for contamination detection training."""
    
    # CPU-optimized models for contamination detection
    MODELS = {
        "distilbert": ModelConfig(
            name="DistilBERT Base",
            model_id="distilbert-base-uncased",
            description="Lightweight BERT variant, optimized for CPU training. Best overall choice for classification.",
            parameters="66M",
            cpu_optimized=True,
            recommended_batch_size=16,
            max_length=512,
            training_type="classification"
        ),
        "bert": ModelConfig(
            name="BERT Base",
            model_id="bert-base-uncased",
            description="Classic BERT model, excellent for sequence classification tasks.",
            parameters="110M",
            cpu_optimized=True,
            recommended_batch_size=8,
            max_length=512,
            training_type="classification"
        ),
        "roberta": ModelConfig(
            name="RoBERTa Base",
            model_id="roberta-base",
            description="Robust BERT variant with improved training methodology.",
            parameters="125M",
            cpu_optimized=True,
            recommended_batch_size=8,
            max_length=512,
            training_type="classification"
        ),
        "dialogpt": ModelConfig(
            name="DialoGPT Medium",
            model_id="microsoft/DialoGPT-medium",
            description="Generative model suitable for prompt-based contamination detection.",
            parameters="345M",
            cpu_optimized=True,
            recommended_batch_size=4,
            max_length=512,
            training_type="generative"
        ),
        "gpt2": ModelConfig(
            name="GPT-2 Base",
            model_id="gpt2",
            description="Classic generative model, good for prompt-based classification.",
            parameters="124M",
            cpu_optimized=True,
            recommended_batch_size=8,
            max_length=512,
            training_type="generative"
        )
    }
    
    # GPU-only models (for reference/future use)
    GPU_MODELS = {
        "gpt-oss-20b": ModelConfig(
            name="GPT-OSS 20B",
            model_id="openai/gpt-oss-20b",
            description="Large-scale GPT model with MXFP4 quantization. Requires GPU.",
            parameters="20B",
            cpu_optimized=False,
            recommended_batch_size=1,
            max_length=2048,
            training_type="generative"
        ),
        "gpt-oss-120b": ModelConfig(
            name="GPT-OSS 120B",
            model_id="openai/gpt-oss-120b",
            description="Massive GPT model with MXFP4 quantization. Requires GPU.",
            parameters="120B",
            cpu_optimized=False,
            recommended_batch_size=1,
            max_length=2048,
            training_type="generative"
        )
    }
    
    @classmethod
    def get_cpu_models(cls) -> Dict[str, ModelConfig]:
        """Get all CPU-compatible models."""
        return cls.MODELS.copy()
    
    @classmethod
    def get_gpu_models(cls) -> Dict[str, ModelConfig]:
        """Get all GPU-only models."""
        return cls.GPU_MODELS.copy()
    
    @classmethod
    def get_all_models(cls) -> Dict[str, ModelConfig]:
        """Get all available models."""
        return {**cls.MODELS, **cls.GPU_MODELS}
    
    @classmethod
    def get_model_config(cls, model_key: str) -> ModelConfig:
        """Get configuration for a specific model."""
        all_models = cls.get_all_models()
        if model_key not in all_models:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(all_models.keys())}")
        return all_models[model_key]
    
    @classmethod
    def get_recommended_model(cls) -> str:
        """Get the recommended default model."""
        return "distilbert"  # Best balance of performance and efficiency
    
    @classmethod
    def list_models(cls, cpu_only: bool = True) -> List[str]:
        """List available model keys."""
        if cpu_only:
            return list(cls.MODELS.keys())
        return list(cls.get_all_models().keys())
    
    @classmethod
    def print_model_info(cls, cpu_only: bool = True) -> None:
        """Print detailed information about available models."""
        models = cls.get_cpu_models() if cpu_only else cls.get_all_models()
        
        print("Available Models for Contamination Detection Training:")
        print("=" * 60)
        
        for key, config in models.items():
            status = "✅ CPU" if config.cpu_optimized else "🔥 GPU"
            print(f"\n{key}: {config.name} ({status})")
            print(f"  Model ID: {config.model_id}")
            print(f"  Parameters: {config.parameters}")
            print(f"  Type: {config.training_type}")
            print(f"  Batch Size: {config.recommended_batch_size}")
            print(f"  Description: {config.description}")
        
        print(f"\nRecommended: {cls.get_recommended_model()}")
