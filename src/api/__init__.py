"""
API components for IsItBenchmark.

This package provides REST API endpoints and web interface
for benchmark contamination detection.
"""

from .server import app, create_app
from .models import AnalysisRequest, AnalysisResponse

__all__ = ["app", "create_app", "AnalysisRequest", "AnalysisResponse"]
