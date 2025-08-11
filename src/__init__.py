"""
IsItBenchmark - Benchmark Contamination Detection for AI/ML Systems

A powerful tool that analyzes prompts and provides probability scores 
for whether they originate from known benchmark datasets.
"""

__version__ = "0.1.0"
__author__ = "Ratnaditya J"
__email__ = "ratnaditya.j@example.com"

from .detection.detector import BenchmarkDetector
from .detection.models import DetectionResult, BenchmarkMatch

__all__ = [
    "BenchmarkDetector",
    "DetectionResult", 
    "BenchmarkMatch",
]
