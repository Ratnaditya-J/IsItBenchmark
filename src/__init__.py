"""
IsItBenchmark - Benchmark Contamination and Evaluation Awareness Detection.

The v1 API exposes benchmark-contamination detection. The v2 API adds
transcript-level evaluation-awareness leakage analysis. Heavy v1 detector
classes are imported lazily so lightweight v2 tooling can run without optional
ML dependencies installed.
"""

__version__ = "0.2.0"
__author__ = "Ratnaditya J"
__email__ = "ratnaditya.j@example.com"

from .eval_awareness import EvaluationAwarenessDetector, LeakageReport, TranscriptInput

_LAZY_EXPORTS = {
    "BenchmarkDetector": (".detection.detector", "BenchmarkDetector"),
    "DetectionResult": (".detection.models", "DetectionResult"),
    "BenchmarkMatch": (".detection.models", "BenchmarkMatch"),
}


def __getattr__(name):
    """Lazily import heavyweight v1 objects on demand."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'src' has no attribute {name!r}")

    import importlib

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = [
    "BenchmarkDetector",
    "DetectionResult",
    "BenchmarkMatch",
    "EvaluationAwarenessDetector",
    "LeakageReport",
    "TranscriptInput",
]
