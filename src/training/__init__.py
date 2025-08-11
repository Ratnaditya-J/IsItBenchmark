"""
Training module for IsItBenchmark.

This module provides specialized model training and fine-tuning capabilities
for benchmark contamination detection, including the first specialized
contamination detection models.
"""

from .contamination_trainer import ContaminationTrainer
from .data_generator import ContaminationDataGenerator
from .specialized_matcher import SpecializedContaminationMatcher

__all__ = [
    'ContaminationTrainer',
    'ContaminationDataGenerator', 
    'SpecializedContaminationMatcher'
]
