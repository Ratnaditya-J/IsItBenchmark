#!/usr/bin/env python3
"""
Pre-trained Model Loader for IsItBenchmark

Handles automatic loading and initialization of the pre-trained specialized
contamination detection model for out-of-the-box use.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PretrainedModelLoader:
    """Loader for pre-trained specialized contamination detection model."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.models_dir = self.project_root / "models"
        self.pretrained_model_dir = self.models_dir / "pretrained_specialized_detector"
        self.model_metadata_file = self.pretrained_model_dir / "package" / "model_metadata.json"
        
        # Model download configuration (for future GitHub releases)
        self.model_download_config = {
            "github_repo": "Ratnaditya-J/IsItBenchmark",
            "model_release_tag": "v1.0-pretrained-model",
            "model_filename": "pretrained_specialized_detector.tar.gz",
            "expected_size_mb": 500  # Approximate model size
        }
    
    def is_model_available(self) -> bool:
        """Check if pre-trained model is available locally."""
        return (
            self.pretrained_model_dir.exists() and
            self.model_metadata_file.exists() and
            (self.pretrained_model_dir / "pytorch_model.bin").exists()
        )
    
    def get_model_metadata(self) -> Optional[Dict[str, Any]]:
        """Get metadata for the pre-trained model."""
        if not self.model_metadata_file.exists():
            return None
        
        try:
            with open(self.model_metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load model metadata: {e}")
            return None
    
    def download_pretrained_model(self, force_download: bool = False) -> bool:
        """Download pre-trained model from GitHub releases (future feature)."""
        if self.is_model_available() and not force_download:
            logger.info("Pre-trained model already available locally")
            return True
        
        logger.info("Pre-trained model download feature will be available in future releases")
        logger.info("For now, please generate the model using: python scripts/generate_pretrained_model.py")
        return False
    
    def validate_model_integrity(self) -> bool:
        """Validate the integrity of the pre-trained model."""
        if not self.is_model_available():
            return False
        
        try:
            metadata = self.get_model_metadata()
            if not metadata:
                logger.error("Model metadata not found or corrupted")
                return False
            
            # Check required files
            required_files = [
                "pytorch_model.bin",
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json"
            ]
            
            for file_name in required_files:
                file_path = self.pretrained_model_dir / file_name
                if not file_path.exists():
                    logger.error(f"Required model file missing: {file_name}")
                    return False
            
            logger.info("Pre-trained model integrity validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model integrity validation failed: {e}")
            return False
    
    def get_model_config_for_matcher(self) -> Dict[str, Any]:
        """Get configuration for initializing the specialized matcher."""
        if not self.is_model_available():
            raise FileNotFoundError("Pre-trained model not available. Run: python scripts/generate_pretrained_model.py")
        
        metadata = self.get_model_metadata()
        if not metadata:
            raise ValueError("Model metadata not available")
        
        return {
            "model_path": str(self.pretrained_model_dir),
            "model_name": "pretrained_specialized_detector",
            "confidence_threshold": metadata.get("usage_info", {}).get("recommended_threshold", 0.8),
            "max_length": metadata.get("usage_info", {}).get("max_input_length", 512),
            "model_version": metadata.get("model_info", {}).get("version", "1.0.0"),
            "is_pretrained": True
        }
    
    def print_model_info(self):
        """Print information about the available pre-trained model."""
        if not self.is_model_available():
            print("❌ Pre-trained model not available")
            print("📥 Generate model: python scripts/generate_pretrained_model.py")
            return
        
        metadata = self.get_model_metadata()
        if not metadata:
            print("⚠️  Model available but metadata missing")
            return
        
        model_info = metadata.get("model_info", {})
        training_info = metadata.get("training_info", {})
        performance = metadata.get("performance_benchmarks", {})
        
        print("🤖 IsItBenchmark Pre-trained Specialized Model")
        print("=" * 50)
        print(f"📊 Version: {model_info.get('version', 'Unknown')}")
        print(f"🏗️  Architecture: {model_info.get('architecture', 'Unknown')}")
        print(f"📏 Parameters: {model_info.get('parameters', 0):,}")
        print(f"💾 Size: {model_info.get('model_size_mb', 0):.1f} MB")
        print(f"📅 Training Date: {model_info.get('training_date', 'Unknown')}")
        print()
        print("📈 Performance Metrics:")
        print(f"  • Accuracy: {performance.get('accuracy', 0):.3f}")
        print(f"  • F1-Score: {performance.get('f1_score', 0):.3f}")
        print(f"  • Precision: {performance.get('precision', 0):.3f}")
        print(f"  • Recall: {performance.get('recall', 0):.3f}")
        print()
        print("🎯 Training Coverage:")
        print(f"  • Training Samples: {training_info.get('training_samples', 0):,}")
        print(f"  • Benchmark Questions: {training_info.get('benchmark_questions_used', 0):,}")
        print(f"  • Contamination Patterns: {training_info.get('contamination_patterns', 0)}")
        print()
        print("✅ Ready for production use!")


def ensure_pretrained_model() -> Dict[str, Any]:
    """Ensure pre-trained model is available and return its configuration."""
    loader = PretrainedModelLoader()
    
    if not loader.is_model_available():
        logger.warning("Pre-trained model not found. Please generate it first.")
        logger.info("Run: python scripts/generate_pretrained_model.py")
        raise FileNotFoundError("Pre-trained specialized model not available")
    
    if not loader.validate_model_integrity():
        logger.error("Pre-trained model validation failed")
        raise ValueError("Pre-trained model is corrupted or incomplete")
    
    return loader.get_model_config_for_matcher()


def main():
    """Main function to check pre-trained model status."""
    loader = PretrainedModelLoader()
    loader.print_model_info()


if __name__ == "__main__":
    main()
