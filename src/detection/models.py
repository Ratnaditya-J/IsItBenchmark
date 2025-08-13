"""
Data models for benchmark detection results and related structures.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum


class BenchmarkType(Enum):
    """Types of benchmarks supported by the system."""
    LANGUAGE_UNDERSTANDING = "language_understanding"
    MATHEMATICAL_REASONING = "mathematical_reasoning"
    CODE_GENERATION = "code_generation"
    COMMON_SENSE = "common_sense"
    COMMONSENSE_REASONING = "commonsense_reasoning"
    SCIENTIFIC_REASONING = "scientific_reasoning"
    SAFETY_EVALUATION = "safety_evaluation"
    READING_COMPREHENSION = "reading_comprehension"
    MULTILINGUAL = "multilingual"
    SPECIALIZED = "specialized"
    MULTIMODAL = "multimodal"


class ConfidenceLevel(Enum):
    """Confidence levels for detection results."""
    VERY_LOW = "very_low"      # 0.0 - 0.2
    LOW = "low"                # 0.2 - 0.4
    MEDIUM = "medium"          # 0.4 - 0.6
    HIGH = "high"              # 0.6 - 0.8
    VERY_HIGH = "very_high"    # 0.8 - 1.0


@dataclass
class MatchResult:
    """Result of a similarity matching operation."""
    text: str
    similarity_score: float
    metadata: Dict[str, Any]
    exact_match: bool = False
    benchmark_type: Optional[str] = None
    
    def __post_init__(self):
        """Validate similarity score range."""
        if not 0.0 <= self.similarity_score <= 1.0:
            raise ValueError("Similarity score must be between 0.0 and 1.0")


@dataclass
class BenchmarkMatch:
    """Represents a potential match with a benchmark dataset."""
    benchmark_name: str
    benchmark_type: BenchmarkType
    similarity_score: float
    exact_match: bool
    matched_text: str
    source_url: Optional[str] = None
    publication_date: Optional[str] = None
    dataset_version: Optional[str] = None
    
    def __post_init__(self):
        """Validate similarity score range."""
        if not 0.0 <= self.similarity_score <= 1.0:
            raise ValueError("Similarity score must be between 0.0 and 1.0")


@dataclass
class DetectionResult:
    """Main result object for benchmark detection analysis."""
    input_prompt: str
    probability: float
    confidence: ConfidenceLevel
    matches: List[BenchmarkMatch]
    analysis_time_ms: float
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate probability and set confidence level."""
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("Probability must be between 0.0 and 1.0")
        
        # Auto-set confidence level based on probability
        if self.probability < 0.2:
            self.confidence = ConfidenceLevel.VERY_LOW
        elif self.probability < 0.4:
            self.confidence = ConfidenceLevel.LOW
        elif self.probability < 0.6:
            self.confidence = ConfidenceLevel.MEDIUM
        elif self.probability < 0.8:
            self.confidence = ConfidenceLevel.HIGH
        else:
            self.confidence = ConfidenceLevel.VERY_HIGH
    
    @property
    def is_likely_benchmark(self) -> bool:
        """Returns True if probability indicates likely benchmark origin."""
        return self.probability >= 0.5
    
    @property
    def top_match(self) -> Optional[BenchmarkMatch]:
        """Returns the highest scoring match, if any."""
        if not self.matches:
            return None
        return max(self.matches, key=lambda m: m.similarity_score)
    
    def get_matches_by_type(self, benchmark_type: BenchmarkType) -> List[BenchmarkMatch]:
        """Filter matches by benchmark type."""
        return [match for match in self.matches if match.benchmark_type == benchmark_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "input_prompt": self.input_prompt,
            "probability": self.probability,
            "confidence": self.confidence.value,
            "is_likely_benchmark": self.is_likely_benchmark,
            "analysis_time_ms": self.analysis_time_ms,
            "matches": [
                {
                    "benchmark_name": match.metadata.get("benchmark_name", "Unknown"),
                    "benchmark_type": match.metadata.get("benchmark_type", "unknown"),
                    "similarity_score": match.similarity_score,
                    "exact_match": match.exact_match,
                    "matched_text": match.text,
                    "source_url": match.metadata.get("source_url", ""),
                    "publication_date": match.metadata.get("publication_date", ""),
                    "dataset_version": match.metadata.get("dataset_version", ""),
                }
                for match in self.matches
            ],
            "top_match": {
                "benchmark_name": self.top_match.metadata.get("benchmark_name", "Unknown"),
                "similarity_score": self.top_match.similarity_score,
            } if self.top_match else None,
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkInfo:
    """Information about a benchmark dataset."""
    name: str
    type: BenchmarkType
    description: str
    source_url: str
    publication_date: str
    num_examples: int
    languages: List[str]
    domains: List[str]
    license: Optional[str] = None
    citation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "source_url": self.source_url,
            "publication_date": self.publication_date,
            "num_examples": self.num_examples,
            "languages": self.languages,
            "domains": self.domains,
            "license": self.license,
            "citation": self.citation,
        }
