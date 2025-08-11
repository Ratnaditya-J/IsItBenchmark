"""
Base matcher interface for IsItBenchmark.

This module defines the abstract base class for all similarity matchers,
ensuring consistent interface across different matching strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from enum import Enum

try:
    from ..detection.models import MatchResult
except ImportError:
    # Fallback for when running as standalone
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'detection'))
    from models import MatchResult


class MatcherType(Enum):
    """Types of available matchers."""
    SEMANTIC = "semantic"
    LLM = "llm"
    ENSEMBLE = "ensemble"
    NGRAM = "ngram"
    MEMBERSHIP_INFERENCE = "membership_inference"
    HYBRID = "hybrid"
    SPECIALIZED = "specialized"


class BaseMatcher(ABC):
    """
    Abstract base class for similarity matchers.
    
    All matcher implementations must inherit from this class and implement
    the required methods for consistent behavior across the system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the matcher.
        
        Args:
            config: Configuration dictionary for the matcher
        """
        self.config = config or {}
        self.is_initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the matcher (load models, prepare resources).
        
        Returns:
            True if initialization was successful
        """
        pass
    
    @abstractmethod
    def find_matches(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]], 
        threshold: float = 0.7,
        max_matches: int = 10
    ) -> List[MatchResult]:
        """
        Find matches for a query against candidate texts.
        
        Args:
            query: Text to match against
            candidates: List of candidate texts with metadata
            threshold: Minimum similarity threshold
            max_matches: Maximum number of matches to return
            
        Returns:
            List of match results sorted by similarity score
        """
        pass
    
    @abstractmethod
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity score between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        pass
    
    def get_matcher_info(self) -> Dict[str, Any]:
        """
        Get information about the matcher.
        
        Returns:
            Dictionary with matcher metadata
        """
        return {
            "matcher_type": self.__class__.__name__,
            "config": getattr(self, 'config', {})
        }
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text before matching (can be overridden).
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Default preprocessing - can be overridden by subclasses
        return text.strip()
    
    def batch_compute_similarity(
        self, 
        query: str, 
        candidates: List[str]
    ) -> List[float]:
        """
        Compute similarity scores for a query against multiple candidates.
        
        Args:
            query: Query text
            candidates: List of candidate texts
            
        Returns:
            List of similarity scores
        """
        # Default implementation - can be optimized by subclasses
        return [self.compute_similarity(query, candidate) for candidate in candidates]
    
    def explain_match(self, query: str, match: str, score: float) -> Dict[str, Any]:
        """
        Provide explanation for a match (optional, can be overridden).
        
        Args:
            query: Original query
            match: Matched text
            score: Similarity score
            
        Returns:
            Dictionary with match explanation
        """
        return {
            "query": query,
            "match": match,
            "score": score,
            "explanation": f"Similarity score: {score:.3f}",
            "matcher_type": self.get_matcher_info().get("type", "unknown")
        }
    
    def validate_config(self) -> bool:
        """
        Validate matcher configuration.
        
        Returns:
            True if configuration is valid
        """
        return True
    
    def cleanup(self):
        """Clean up resources (optional, can be overridden)."""
        pass
