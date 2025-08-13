"""
Matcher factory for IsItBenchmark.

This module provides a factory for creating different types of similarity matchers
based on user preferences and configuration.
"""

import logging
from typing import Dict, Any, Optional, List, Union
from enum import Enum

try:
    from .base_matcher import BaseMatcher, MatcherType
    from .semantic_matcher import SemanticMatcher
    from .llm_matcher import LLMMatcher, HuggingFaceLLMMatcher
    from .ngram_matcher import NgramMatcher
    from .membership_inference_matcher import MembershipInferenceMatcher
    from .ensemble_matcher import EnsembleMatcher, EnsembleConfig
    from ..training.specialized_matcher import SpecializedContaminationMatcher
    from ..detection.models import MatchResult
except ImportError:
    # Fallback for standalone execution
    import sys
    import os
    current_dir = os.path.dirname(__file__)
    sys.path.append(current_dir)
    sys.path.append(os.path.join(current_dir, '..', 'detection'))
    sys.path.append(os.path.join(current_dir, '..', 'training'))
    from base_matcher import BaseMatcher, MatcherType
    from .semantic_matcher import SemanticMatcher
    from .llm_matcher import LLMMatcher, HuggingFaceLLMMatcher
    from .ngram_matcher import NgramMatcher
    from .membership_inference_matcher import MembershipInferenceMatcher
    from .ensemble_matcher import EnsembleMatcher, EnsembleConfig
    from ..training.specialized_matcher import SpecializedContaminationMatcher
    from models import MatchResult


class MatcherFactory:
    """
    Factory class for creating similarity matchers.
    
    Provides a unified interface for creating and configuring different
    types of matchers based on user requirements.
    """
    
    def __init__(self):
        """Initialize the matcher factory."""
        self.logger = logging.getLogger(__name__)
        
        # Registry of available matchers
        self._matcher_registry = {
            MatcherType.SEMANTIC: SemanticMatcher,
            MatcherType.LLM: HuggingFaceLLMMatcher,
            MatcherType.NGRAM: NgramMatcher,
            MatcherType.MEMBERSHIP_INFERENCE: MembershipInferenceMatcher,
            MatcherType.ENSEMBLE: EnsembleMatcher,
            MatcherType.SPECIALIZED: SpecializedContaminationMatcher,
        }
        
        # Default configurations for each matcher type
        self._default_configs = {
            MatcherType.SEMANTIC: {
                "use_semantic": True,
                "semantic_model": "all-MiniLM-L6-v2",
                "exact_threshold": 0.95,
                "fuzzy_threshold": 0.8,
                "semantic_threshold": 0.7,
            },
            MatcherType.LLM: {
                "model_name": "microsoft/DialoGPT-medium",
                "device": "auto",
                "max_length": 512,
                "temperature": 0.1,
                "batch_size": 1,
            },
            MatcherType.NGRAM: {
                "ngram_size": 13,  # GPT-3 approach
                "char_window_size": 50,  # GPT-4 approach
                "min_overlap_threshold": 0.7,
                "use_word_ngrams": True,
                "use_char_ngrams": True,
            },
            MatcherType.MEMBERSHIP_INFERENCE: {
                "perplexity_threshold": 10.0,
                "confidence_threshold": 0.8,
                "sample_size": 100,
                "use_statistical_test": True,
            },
            MatcherType.ENSEMBLE: {
                "semantic_weight": 0.3,
                "llm_weight": 0.3,
                "ngram_weight": 0.2,
                "membership_inference_weight": 0.2,
                "high_confidence_threshold": 0.8,
                "medium_confidence_threshold": 0.6,
                "combination_method": "max"
            },
            MatcherType.SPECIALIZED: {
                "model_path": "models/specialized_contamination_detector",
                "device": "auto",
                "max_length": 512,
                "batch_size": 8,
                "confidence_threshold": 0.8
            }
        }
    
    def create_matcher(
        self, 
        matcher_type: Union[str, MatcherType], 
        config: Optional[Dict[str, Any]] = None
    ) -> BaseMatcher:
        """
        Create a matcher of the specified type.
        
        Args:
            matcher_type: Type of matcher to create
            config: Optional configuration dictionary
            
        Returns:
            Initialized matcher instance
            
        Raises:
            ValueError: If matcher type is not supported
            RuntimeError: If matcher initialization fails
        """
        # Convert string to enum if necessary
        if isinstance(matcher_type, str):
            try:
                matcher_type = MatcherType(matcher_type.lower())
            except ValueError:
                raise ValueError(f"Unsupported matcher type: {matcher_type}")
        
        # Check if matcher type is supported
        if matcher_type not in self._matcher_registry:
            raise ValueError(f"Matcher type {matcher_type} not registered")
        
        # Get matcher class
        matcher_class = self._matcher_registry[matcher_type]
        
        # Merge configuration with defaults
        final_config = self._default_configs.get(matcher_type, {}).copy()
        if config:
            final_config.update(config)
        
        self.logger.info(f"Creating {matcher_type.value} matcher with config: {final_config}")
        
        try:
            # Create matcher instance with special handling for ensemble
            if matcher_type == MatcherType.ENSEMBLE:
                # Convert dict config to EnsembleConfig object
                ensemble_config = EnsembleConfig(**final_config)
                matcher = matcher_class(ensemble_config)
            else:
                matcher = matcher_class(final_config)
            
            # Validate configuration
            if not matcher.validate_config():
                raise RuntimeError(f"Invalid configuration for {matcher_type.value} matcher")
            
            # Initialize matcher
            if not matcher.initialize():
                raise RuntimeError(f"Failed to initialize {matcher_type.value} matcher")
            
            self.logger.info(f"{matcher_type.value} matcher created successfully")
            return matcher
            
        except Exception as e:
            self.logger.error(f"Failed to create {matcher_type.value} matcher: {str(e)}")
            raise RuntimeError(f"Matcher creation failed: {str(e)}")
    
    def get_available_matchers(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available matchers.
        
        Returns:
            Dictionary with matcher information
        """
        available = {}
        
        for matcher_type in self._matcher_registry:
            matcher_class = self._matcher_registry[matcher_type]
            
            # Create a temporary instance to get info
            try:
                temp_matcher = matcher_class()
                info = temp_matcher.get_matcher_info()
                info["default_config"] = self._default_configs.get(matcher_type, {})
                available[matcher_type.value] = info
            except Exception as e:
                self.logger.warning(f"Could not get info for {matcher_type.value}: {str(e)}")
                available[matcher_type.value] = {
                    "type": matcher_type.value,
                    "status": "error",
                    "error": str(e)
                }
        
        return available
    
    def register_matcher(
        self, 
        matcher_type: MatcherType, 
        matcher_class: type,
        default_config: Optional[Dict[str, Any]] = None
    ):
        """
        Register a new matcher type.
        
        Args:
            matcher_type: Type identifier for the matcher
            matcher_class: Class implementing the matcher
            default_config: Default configuration for the matcher
        """
        if not issubclass(matcher_class, BaseMatcher):
            raise ValueError("Matcher class must inherit from BaseMatcher")
        
        self._matcher_registry[matcher_type] = matcher_class
        
        if default_config:
            self._default_configs[matcher_type] = default_config
        
        self.logger.info(f"Registered matcher type: {matcher_type.value}")
    
    def get_default_config(self, matcher_type: Union[str, MatcherType]) -> Dict[str, Any]:
        """
        Get default configuration for a matcher type.
        
        Args:
            matcher_type: Type of matcher
            
        Returns:
            Default configuration dictionary
        """
        if isinstance(matcher_type, str):
            matcher_type = MatcherType(matcher_type.lower())
        
        return self._default_configs.get(matcher_type, {}).copy()
    
    def create_hybrid_matcher(
        self, 
        primary_type: Union[str, MatcherType],
        secondary_type: Union[str, MatcherType],
        primary_config: Optional[Dict[str, Any]] = None,
        secondary_config: Optional[Dict[str, Any]] = None,
        hybrid_config: Optional[Dict[str, Any]] = None
    ) -> 'HybridMatcher':
        """
        Create a hybrid matcher that combines two matcher types.
        
        Args:
            primary_type: Primary matcher type (for initial filtering)
            secondary_type: Secondary matcher type (for verification)
            primary_config: Configuration for primary matcher
            secondary_config: Configuration for secondary matcher
            hybrid_config: Configuration for hybrid behavior
            
        Returns:
            Hybrid matcher instance
        """
        primary_matcher = self.create_matcher(primary_type, primary_config)
        secondary_matcher = self.create_matcher(secondary_type, secondary_config)
        
        return HybridMatcher(
            primary_matcher=primary_matcher,
            secondary_matcher=secondary_matcher,
            config=hybrid_config or {}
        )


class HybridMatcher(BaseMatcher):
    """
    Hybrid matcher that combines two different matcher types.
    
    Uses a fast primary matcher for initial filtering and a more accurate
    secondary matcher for verification of high-confidence matches.
    """
    
    def __init__(
        self, 
        primary_matcher: BaseMatcher,
        secondary_matcher: BaseMatcher,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize hybrid matcher.
        
        Args:
            primary_matcher: Fast matcher for initial filtering
            secondary_matcher: Accurate matcher for verification
            config: Hybrid matcher configuration
        """
        super().__init__(config)
        
        self.primary_matcher = primary_matcher
        self.secondary_matcher = secondary_matcher
        
        # Configuration
        self.primary_threshold = self.config.get("primary_threshold", 0.5)
        self.secondary_threshold = self.config.get("secondary_threshold", 0.7)
        self.max_primary_candidates = self.config.get("max_primary_candidates", 50)
        
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> bool:
        """Initialize both matchers."""
        try:
            primary_init = self.primary_matcher.initialize() if not self.primary_matcher.is_initialized else True
            secondary_init = self.secondary_matcher.initialize() if not self.secondary_matcher.is_initialized else True
            
            self.is_initialized = primary_init and secondary_init
            return self.is_initialized
            
        except Exception as e:
            self.logger.error(f"Failed to initialize hybrid matcher: {str(e)}")
            return False
    
    def find_matches(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]], 
        threshold: float = 0.7,
        max_matches: int = 10
    ) -> List[MatchResult]:
        """Find matches using hybrid approach."""
        if not self.is_initialized:
            raise RuntimeError("Hybrid matcher not initialized")
        
        # Stage 1: Fast primary filtering
        self.logger.info(f"Stage 1: Primary filtering with {self.primary_matcher.__class__.__name__}")
        primary_matches = self.primary_matcher.find_matches(
            query=query,
            candidates=candidates,
            threshold=self.primary_threshold,
            max_matches=self.max_primary_candidates
        )
        
        if not primary_matches:
            return []
        
        # Stage 2: Secondary verification
        self.logger.info(f"Stage 2: Secondary verification with {self.secondary_matcher.__class__.__name__}")
        
        # Convert primary matches back to candidate format for secondary matcher
        secondary_candidates = []
        for match in primary_matches:
            candidate = {
                "question_text": match.text,
                "benchmark_name": match.metadata.get("benchmark_name", "Unknown"),
                "benchmark_type": match.metadata.get("benchmark_type", "unknown"),
                "source_url": match.metadata.get("source_url", ""),
                "publication_date": match.metadata.get("publication_date", ""),
                "dataset_version": match.metadata.get("dataset_version", ""),
            }
            secondary_candidates.append(candidate)
        
        # Get secondary verification
        verified_matches = self.secondary_matcher.find_matches(
            query=query,
            candidates=secondary_candidates,
            threshold=threshold,
            max_matches=max_matches
        )
        
        self.logger.info(f"Hybrid matching: {len(primary_matches)} → {len(verified_matches)} matches")
        
        return verified_matches
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity using secondary matcher for accuracy."""
        return self.secondary_matcher.compute_similarity(text1, text2)
    
    def get_matcher_info(self) -> Dict[str, Any]:
        """Get information about the hybrid matcher."""
        return {
            "type": MatcherType.HYBRID.value,
            "primary_matcher": self.primary_matcher.get_matcher_info(),
            "secondary_matcher": self.secondary_matcher.get_matcher_info(),
            "is_initialized": self.is_initialized,
            "thresholds": {
                "primary": self.primary_threshold,
                "secondary": self.secondary_threshold,
            },
            "capabilities": [
                "fast_filtering",
                "accurate_verification",
                "two_stage_processing"
            ]
        }
    
    def cleanup(self):
        """Clean up both matchers."""
        try:
            self.primary_matcher.cleanup()
            self.secondary_matcher.cleanup()
            self.is_initialized = False
        except Exception as e:
            self.logger.warning(f"Error during hybrid matcher cleanup: {str(e)}")


# Global factory instance
matcher_factory = MatcherFactory()


def create_matcher(
    matcher_type: Union[str, MatcherType], 
    config: Optional[Dict[str, Any]] = None
) -> BaseMatcher:
    """
    Convenience function to create a matcher.
    
    Args:
        matcher_type: Type of matcher to create
        config: Optional configuration
        
    Returns:
        Initialized matcher instance
    """
    return matcher_factory.create_matcher(matcher_type, config)


def get_available_matchers() -> Dict[str, Dict[str, Any]]:
    """
    Get information about available matchers.
    
    Returns:
        Dictionary with matcher information
    """
    return matcher_factory.get_available_matchers()
