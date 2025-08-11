"""
Similarity matching modules for benchmark detection.

This package provides various similarity matching algorithms:
- Text-based fuzzy matching
- Semantic similarity using embeddings
- Pattern-based matching
"""

from .base_matcher import BaseMatcher, MatcherType
from .semantic_matcher import SemanticMatcher
from .llm_matcher import LLMMatcher, HuggingFaceLLMMatcher
from .matcher_factory import MatcherFactory, HybridMatcher, create_matcher, get_available_matchers

__all__ = [
    "BaseMatcher",
    "MatcherType",
    "SemanticMatcher", 
    "LLMMatcher",
    "HuggingFaceLLMMatcher",
    "MatcherFactory",
    "HybridMatcher",
    "create_matcher",
    "get_available_matchers",
]
