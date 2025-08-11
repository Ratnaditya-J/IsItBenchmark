#!/usr/bin/env python3
"""
Ensemble Matcher for IsItBenchmark

Combines multiple detection methods (semantic, LLM, n-gram, membership inference)
to provide comprehensive contamination detection with weighted scoring and
confidence estimation.

This represents the state-of-the-art in benchmark contamination detection,
leveraging the strengths of all available detection methods.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from statistics import mean, stdev

try:
    from .base_matcher import BaseMatcher
    from .semantic_matcher import SemanticMatcher
    from .llm_matcher import HuggingFaceLLMMatcher
    from .ngram_matcher import NgramMatcher
    from .membership_inference_matcher import MembershipInferenceMatcher
    from ..detection.models import MatchResult
except ImportError:
    # Fallback for standalone execution
    import sys
    import os
    current_dir = os.path.dirname(__file__)
    sys.path.append(current_dir)
    sys.path.append(os.path.join(current_dir, '..', 'detection'))
    from base_matcher import BaseMatcher
    from semantic_matcher import SemanticMatcher
    from llm_matcher import HuggingFaceLLMMatcher
    from ngram_matcher import NgramMatcher
    from membership_inference_matcher import MembershipInferenceMatcher
    from models import MatchResult

logger = logging.getLogger(__name__)

@dataclass
class EnsembleConfig:
    """Configuration for ensemble matcher."""
    # Matcher weights (should sum to 1.0)
    semantic_weight: float = 0.3
    llm_weight: float = 0.3
    ngram_weight: float = 0.2
    membership_inference_weight: float = 0.2
    
    # Confidence thresholds
    high_confidence_threshold: float = 0.8
    medium_confidence_threshold: float = 0.6
    
    # Ensemble method
    combination_method: str = "weighted_average"  # "weighted_average", "voting", "max"
    
    # Individual matcher configs
    semantic_config: Dict[str, Any] = None
    llm_config: Dict[str, Any] = None
    ngram_config: Dict[str, Any] = None
    membership_inference_config: Dict[str, Any] = None
    
    def __post_init__(self):
        """Set default configs if not provided."""
        if self.semantic_config is None:
            self.semantic_config = {
                'model_name': 'all-MiniLM-L6-v2',
                'similarity_threshold': 0.7
            }
        
        if self.llm_config is None:
            self.llm_config = {
                'model_name': 'microsoft/DialoGPT-medium',
                'max_length': 512,
                'temperature': 0.1
            }
        
        if self.ngram_config is None:
            self.ngram_config = {
                'ngram_size': 13,
                'char_window_size': 50,
                'min_overlap_threshold': 0.7,
                'use_word_ngrams': True,
                'use_char_ngrams': True
            }
        
        if self.membership_inference_config is None:
            self.membership_inference_config = {
                'perplexity_threshold': 10.0,
                'confidence_threshold': 0.8,
                'sample_size': 100,
                'use_statistical_test': True
            }

@dataclass
class EnsembleMatchResult(MatchResult):
    """Enhanced match result with ensemble-specific information."""
    individual_scores: Dict[str, float] = None
    confidence_level: str = "medium"  # "high", "medium", "low"
    agreement_score: float = 0.0  # How much matchers agree (0-1)
    method_contributions: Dict[str, float] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.individual_scores is None:
            self.individual_scores = {}
        if self.method_contributions is None:
            self.method_contributions = {}

class EnsembleMatcher(BaseMatcher):
    """
    Ensemble matcher that combines multiple detection methods.
    
    This matcher represents the state-of-the-art in contamination detection,
    combining semantic similarity, LLM-based analysis, n-gram overlap,
    and membership inference to provide comprehensive detection with
    confidence estimation.
    """
    
    def __init__(self, config: EnsembleConfig):
        """Initialize ensemble matcher with individual matchers."""
        self.config = config
        self.matchers = {}
        self.is_initialized = False
        
        logger.info(f"Ensemble matcher initializing with config: {config}")
        
        # Validate weights
        total_weight = (config.semantic_weight + config.llm_weight + 
                       config.ngram_weight + config.membership_inference_weight)
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Matcher weights sum to {total_weight}, not 1.0. Normalizing...")
            # Normalize weights
            config.semantic_weight /= total_weight
            config.llm_weight /= total_weight
            config.ngram_weight /= total_weight
            config.membership_inference_weight /= total_weight
    
    def initialize(self) -> bool:
        """Initialize all individual matchers."""
        try:
            logger.info("Initializing individual matchers for ensemble...")
            
            # Initialize semantic matcher
            if self.config.semantic_weight > 0:
                self.matchers['semantic'] = SemanticMatcher(self.config.semantic_config)
                self.matchers['semantic'].initialize()
                logger.info("Semantic matcher initialized")
            
            # Initialize LLM matcher
            if self.config.llm_weight > 0:
                self.matchers['llm'] = HuggingFaceLLMMatcher(self.config.llm_config)
                self.matchers['llm'].initialize()
                logger.info("LLM matcher initialized")
            
            # Initialize n-gram matcher
            if self.config.ngram_weight > 0:
                self.matchers['ngram'] = NgramMatcher(self.config.ngram_config)
                self.matchers['ngram'].initialize()
                logger.info("N-gram matcher initialized")
            
            # Initialize membership inference matcher
            if self.config.membership_inference_weight > 0:
                self.matchers['membership_inference'] = MembershipInferenceMatcher(self.config.membership_inference_config)
                self.matchers['membership_inference'].initialize()
                logger.info("Membership inference matcher initialized")
            
            self.is_initialized = True
            logger.info("Ensemble matcher initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ensemble matcher: {e}")
            return False
    
    def cleanup(self):
        """Clean up all individual matchers."""
        for matcher_name, matcher in self.matchers.items():
            try:
                if hasattr(matcher, 'cleanup'):
                    matcher.cleanup()
                logger.info(f"Cleaned up {matcher_name} matcher")
            except Exception as e:
                logger.warning(f"Error cleaning up {matcher_name} matcher: {e}")
        
        self.matchers.clear()
        self.is_initialized = False
        logger.info("Ensemble matcher cleaned up")
    
    def find_matches(self, query: str, candidates: List[Dict[str, Any]], 
                    max_matches: int = 10) -> List[EnsembleMatchResult]:
        """Find matches using ensemble of all available methods."""
        if not self.is_initialized:
            logger.error("Ensemble matcher not initialized")
            return []
        
        logger.info(f"Finding ensemble matches for query: '{query[:50]}...'")
        
        # Collect results from all matchers
        all_results = {}
        individual_scores = {}
        
        for matcher_name, matcher in self.matchers.items():
            try:
                logger.info(f"Running {matcher_name} matcher...")
                results = matcher.find_matches(query, candidates, max_matches)
                all_results[matcher_name] = results
                
                # Extract scores for each candidate
                scores = {}
                for result in results:
                    # Handle different result formats from different matchers
                    if hasattr(result, 'benchmark_name'):
                        benchmark_name = result.benchmark_name
                    elif hasattr(result, 'metadata') and 'benchmark_name' in result.metadata:
                        benchmark_name = result.metadata['benchmark_name']
                    else:
                        # Fallback: try to extract from text or skip
                        logger.warning(f"Could not extract benchmark_name from {matcher_name} result")
                        continue
                    
                    # Handle different score formats
                    if hasattr(result, 'score'):
                        score = result.score
                    elif hasattr(result, 'similarity_score'):
                        score = result.similarity_score
                    else:
                        logger.warning(f"Could not extract score from {matcher_name} result")
                        score = 0.0
                    
                    scores[benchmark_name] = score
                individual_scores[matcher_name] = scores
                
                logger.info(f"{matcher_name} found {len(results)} matches")
                
            except Exception as e:
                logger.error(f"Error running {matcher_name} matcher: {e}")
                all_results[matcher_name] = []
                individual_scores[matcher_name] = {}
        
        # Combine results using ensemble method
        ensemble_results = self._combine_results(
            all_results, individual_scores, candidates, max_matches
        )
        
        logger.info(f"Ensemble found {len(ensemble_results)} combined matches")
        return ensemble_results
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute ensemble similarity between two texts."""
        if not self.is_initialized:
            logger.error("Ensemble matcher not initialized")
            return 0.0
        
        similarities = {}
        weights = {
            'semantic': self.config.semantic_weight,
            'llm': self.config.llm_weight,
            'ngram': self.config.ngram_weight,
            'membership_inference': self.config.membership_inference_weight
        }
        
        # Get similarity from each matcher
        for matcher_name, matcher in self.matchers.items():
            try:
                similarity = matcher.compute_similarity(text1, text2)
                similarities[matcher_name] = similarity
                logger.debug(f"{matcher_name} similarity: {similarity:.3f}")
            except Exception as e:
                logger.warning(f"Error computing {matcher_name} similarity: {e}")
                similarities[matcher_name] = 0.0
        
        # Compute weighted average
        weighted_sum = sum(similarities[name] * weights[name] 
                          for name in similarities.keys() if name in weights)
        total_weight = sum(weights[name] for name in similarities.keys() if name in weights)
        
        if total_weight == 0:
            return 0.0
        
        ensemble_similarity = weighted_sum / total_weight
        logger.info(f"Ensemble similarity: {ensemble_similarity:.3f}")
        return ensemble_similarity
    
    def _combine_results(self, all_results: Dict[str, List[MatchResult]], 
                        individual_scores: Dict[str, Dict[str, float]],
                        candidates: List[Dict[str, Any]], 
                        max_matches: int) -> List[EnsembleMatchResult]:
        """Combine results from all matchers using the configured method."""
        
        # Get all unique benchmark names
        all_benchmarks = set()
        for results in all_results.values():
            for result in results:
                # Handle different result formats
                if hasattr(result, 'benchmark_name'):
                    benchmark_name = result.benchmark_name
                elif hasattr(result, 'metadata') and 'benchmark_name' in result.metadata:
                    benchmark_name = result.metadata['benchmark_name']
                else:
                    continue  # Skip results without benchmark name
                all_benchmarks.add(benchmark_name)
        
        ensemble_results = []
        
        for benchmark_name in all_benchmarks:
            # Collect scores from all matchers for this benchmark
            matcher_scores = {}
            matcher_results = {}
            
            for matcher_name, results in all_results.items():
                # Find result for this benchmark
                found_match = False
                for result in results:
                    # Extract benchmark name from result
                    result_benchmark_name = None
                    if hasattr(result, 'benchmark_name'):
                        result_benchmark_name = result.benchmark_name
                    elif hasattr(result, 'metadata') and 'benchmark_name' in result.metadata:
                        result_benchmark_name = result.metadata['benchmark_name']
                    
                    if result_benchmark_name == benchmark_name:
                        # Extract score from result
                        if hasattr(result, 'score'):
                            score = result.score
                        elif hasattr(result, 'similarity_score'):
                            score = result.similarity_score
                        else:
                            score = 0.0
                        
                        matcher_scores[matcher_name] = score
                        matcher_results[matcher_name] = result
                        found_match = True
                        break
                
                if not found_match:
                    # No result from this matcher
                    matcher_scores[matcher_name] = 0.0
            
            # Calculate ensemble score
            ensemble_score = self._calculate_ensemble_score(matcher_scores)
            
            # Calculate confidence and agreement
            confidence_level, agreement_score = self._calculate_confidence(matcher_scores)
            
            # Calculate method contributions
            method_contributions = self._calculate_contributions(matcher_scores)
            
            # Create ensemble result
            # Use the first available result as template
            template_result = None
            for results in all_results.values():
                for result in results:
                    if result.benchmark_name == benchmark_name:
                        template_result = result
                        break
                if template_result:
                    break
            
            if template_result:
                ensemble_result = EnsembleMatchResult(
                    benchmark_name=benchmark_name,
                    score=ensemble_score,
                    text=f"Ensemble detection from {len(matcher_scores)} methods: {benchmark_name}",
                    method="ensemble",
                    metadata={
                        'individual_results': matcher_results,
                        'ensemble_method': self.config.combination_method,
                        'total_matchers': len(self.matchers)
                    },
                    individual_scores=matcher_scores,
                    confidence_level=confidence_level,
                    agreement_score=agreement_score,
                    method_contributions=method_contributions
                )
                ensemble_results.append(ensemble_result)
        
        # Sort by ensemble score and limit results
        ensemble_results.sort(key=lambda x: x.score, reverse=True)
        return ensemble_results[:max_matches]
    
    def _calculate_ensemble_score(self, matcher_scores: Dict[str, float]) -> float:
        """Calculate ensemble score using the configured method."""
        if not matcher_scores:
            return 0.0
        
        weights = {
            'semantic': self.config.semantic_weight,
            'llm': self.config.llm_weight,
            'ngram': self.config.ngram_weight,
            'membership_inference': self.config.membership_inference_weight
        }
        
        if self.config.combination_method == "weighted_average":
            weighted_sum = sum(matcher_scores.get(name, 0.0) * weights.get(name, 0.0) 
                              for name in matcher_scores.keys())
            total_weight = sum(weights.get(name, 0.0) for name in matcher_scores.keys())
            return weighted_sum / max(total_weight, 0.001)
        
        elif self.config.combination_method == "voting":
            # Count how many matchers exceed their individual thresholds
            votes = 0
            total_votes = 0
            for name, score in matcher_scores.items():
                total_votes += 1
                # Use different thresholds for different matchers
                threshold = 0.5  # Default threshold
                if name == 'semantic':
                    threshold = self.config.semantic_config.get('similarity_threshold', 0.7)
                elif name == 'membership_inference':
                    threshold = self.config.membership_inference_config.get('confidence_threshold', 0.8)
                elif name == 'ngram':
                    threshold = self.config.ngram_config.get('min_overlap_threshold', 0.7)
                
                if score >= threshold:
                    votes += 1
            
            return votes / max(total_votes, 1)
        
        elif self.config.combination_method == "max":
            return max(matcher_scores.values())
        
        else:
            # Default to weighted average
            return self._calculate_ensemble_score(matcher_scores)
    
    def _calculate_confidence(self, matcher_scores: Dict[str, float]) -> Tuple[str, float]:
        """Calculate confidence level and agreement score."""
        if not matcher_scores:
            return "low", 0.0
        
        scores = list(matcher_scores.values())
        
        # Calculate agreement (inverse of standard deviation)
        if len(scores) > 1:
            try:
                score_std = stdev(scores)
                agreement_score = max(0.0, 1.0 - score_std)
            except:
                agreement_score = 0.0
        else:
            agreement_score = 1.0
        
        # Calculate average score
        avg_score = mean(scores)
        
        # Determine confidence level
        if avg_score >= self.config.high_confidence_threshold and agreement_score >= 0.7:
            confidence_level = "high"
        elif avg_score >= self.config.medium_confidence_threshold and agreement_score >= 0.5:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        return confidence_level, agreement_score
    
    def _calculate_contributions(self, matcher_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate how much each method contributes to the final score."""
        weights = {
            'semantic': self.config.semantic_weight,
            'llm': self.config.llm_weight,
            'ngram': self.config.ngram_weight,
            'membership_inference': self.config.membership_inference_weight
        }
        
        contributions = {}
        total_weighted_score = sum(matcher_scores.get(name, 0.0) * weights.get(name, 0.0) 
                                  for name in matcher_scores.keys())
        
        if total_weighted_score > 0:
            for name, score in matcher_scores.items():
                weight = weights.get(name, 0.0)
                contribution = (score * weight) / total_weighted_score
                contributions[name] = contribution
        else:
            # Equal contribution if no scores
            num_matchers = len(matcher_scores)
            for name in matcher_scores.keys():
                contributions[name] = 1.0 / num_matchers if num_matchers > 0 else 0.0
        
        return contributions
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the ensemble matcher."""
        return {
            'matcher_type': 'EnsembleMatcher',
            'config': {
                'combination_method': self.config.combination_method,
                'weights': {
                    'semantic': self.config.semantic_weight,
                    'llm': self.config.llm_weight,
                    'ngram': self.config.ngram_weight,
                    'membership_inference': self.config.membership_inference_weight
                },
                'thresholds': {
                    'high_confidence': self.config.high_confidence_threshold,
                    'medium_confidence': self.config.medium_confidence_threshold
                }
            },
            'active_matchers': list(self.matchers.keys()),
            'is_initialized': self.is_initialized
        }
