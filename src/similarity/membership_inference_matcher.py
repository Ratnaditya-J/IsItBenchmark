"""
Membership inference detection for benchmark contamination.

This module implements membership inference attacks to determine if a prompt
was likely seen during model training by analyzing model confidence patterns.
"""

import logging
import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
import statistics

try:
    from .base_matcher import BaseMatcher, MatchResult
    from ..detection.models import BenchmarkInfo
except ImportError:
    # Fallback for standalone execution
    import sys
    import os
    current_dir = os.path.dirname(__file__)
    sys.path.append(current_dir)
    sys.path.append(os.path.join(current_dir, '..', 'detection'))
    from base_matcher import BaseMatcher, MatchResult
    from models import BenchmarkInfo

logger = logging.getLogger(__name__)


@dataclass
class PerplexityScore:
    """Represents perplexity analysis results for a text."""
    text: str
    perplexity: float
    log_likelihood: float
    token_count: int
    avg_token_probability: float


class MembershipInferenceMatcher(BaseMatcher):
    """
    Membership inference matcher for detecting benchmark contamination.
    
    Uses statistical analysis of model confidence (perplexity) to determine
    if a prompt was likely in the model's training data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the membership inference matcher.
        
        Args:
            config: Configuration dictionary containing:
                - perplexity_threshold: Threshold for perplexity-based detection (default: 10.0)
                - confidence_threshold: Confidence threshold for membership inference (default: 0.8)
                - sample_size: Number of samples for statistical analysis (default: 100)
                - use_statistical_test: Whether to use statistical tests (default: True)
        """
        self.config = config
        self.perplexity_threshold = config.get('perplexity_threshold', 10.0)
        self.confidence_threshold = config.get('confidence_threshold', 0.8)
        self.sample_size = config.get('sample_size', 100)
        self.use_statistical_test = config.get('use_statistical_test', True)
        self.initialized = False
        
        # Reference perplexity statistics (computed from non-benchmark text)
        self.reference_perplexity_mean: Optional[float] = None
        self.reference_perplexity_std: Optional[float] = None
        self.reference_samples: List[float] = []
        
        logger.info(f"Membership inference matcher initialized with config: {config}")
    
    def initialize(self) -> bool:
        """
        Initialize the matcher (validate configuration and prepare for inference).
        
        Returns:
            True if initialization was successful
        """
        try:
            # Validate configuration parameters
            if self.perplexity_threshold <= 0:
                logger.error("Perplexity threshold must be positive")
                return False
            if not (0.0 <= self.confidence_threshold <= 1.0):
                logger.error("Confidence threshold must be between 0.0 and 1.0")
                return False
            if self.sample_size <= 0:
                logger.error("Sample size must be positive")
                return False
            
            self.initialized = True
            logger.info("Membership inference matcher initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize membership inference matcher: {e}")
            return False
    
    def _calculate_perplexity(self, text: str) -> PerplexityScore:
        """
        Calculate perplexity score for given text.
        
        Note: This is a simplified implementation. In practice, you would use
        a language model (like GPT-2, GPT-3, etc.) to calculate actual perplexity.
        
        Args:
            text: Input text to analyze
            
        Returns:
            PerplexityScore with calculated metrics
        """
        # Simplified perplexity calculation based on text characteristics
        # In a real implementation, this would use a language model
        
        # Basic text preprocessing
        tokens = text.lower().split()
        token_count = len(tokens)
        
        if token_count == 0:
            return PerplexityScore(
                text=text,
                perplexity=float('inf'),
                log_likelihood=-float('inf'),
                token_count=0,
                avg_token_probability=0.0
            )
        
        # Simulate perplexity calculation using heuristics
        # Lower perplexity = higher model confidence = more likely in training data
        
        # Factors that suggest lower perplexity (higher confidence):
        # 1. Common words and phrases
        # 2. Standard question formats
        # 3. Typical benchmark question patterns
        
        common_words = {
            'what', 'is', 'the', 'a', 'an', 'of', 'in', 'to', 'for', 'and', 'or',
            'which', 'how', 'when', 'where', 'why', 'who', 'can', 'will', 'would',
            'should', 'could', 'may', 'might', 'do', 'does', 'did', 'have', 'has',
            'had', 'are', 'was', 'were', 'be', 'been', 'being'
        }
        
        question_patterns = [
            'what is', 'which of', 'how many', 'what are', 'which is',
            'what does', 'how does', 'what would', 'which would',
            'select the', 'choose the', 'identify the', 'find the'
        ]
        
        # Calculate heuristic confidence score
        common_word_ratio = sum(1 for token in tokens if token in common_words) / token_count
        
        # Check for question patterns
        text_lower = text.lower()
        pattern_matches = sum(1 for pattern in question_patterns if pattern in text_lower)
        pattern_score = min(1.0, pattern_matches * 0.3)
        
        # Length penalty (very short or very long texts are less common)
        length_penalty = 1.0
        if token_count < 5 or token_count > 50:
            length_penalty = 0.8
        
        # Calculate simulated confidence (inverse of perplexity)
        base_confidence = (common_word_ratio * 0.4 + pattern_score * 0.4 + length_penalty * 0.2)
        
        # Add some randomness to simulate model uncertainty
        import random
        random.seed(hash(text) % 2**32)  # Deterministic randomness based on text
        noise = random.gauss(0, 0.1)
        confidence = max(0.1, min(0.9, base_confidence + noise))
        
        # Convert confidence to perplexity (inverse relationship)
        perplexity = 1.0 / confidence if confidence > 0 else float('inf')
        
        # Calculate derived metrics
        log_likelihood = -np.log(perplexity) * token_count
        avg_token_probability = confidence
        
        return PerplexityScore(
            text=text,
            perplexity=perplexity,
            log_likelihood=log_likelihood,
            token_count=token_count,
            avg_token_probability=avg_token_probability
        )
    
    def _build_reference_distribution(self, candidates: List[BenchmarkInfo]) -> None:
        """
        Build reference perplexity distribution from non-benchmark text.
        
        This creates a baseline for comparison to detect anomalously low perplexity.
        """
        if self.reference_perplexity_mean is not None:
            return  # Already built
        
        logger.info("Building reference perplexity distribution...")
        
        # Generate reference texts (simulated non-benchmark text)
        reference_texts = [
            "The weather today is quite pleasant and sunny.",
            "I need to go to the grocery store later.",
            "My favorite color is blue because it reminds me of the ocean.",
            "The meeting is scheduled for tomorrow at 3 PM.",
            "She enjoys reading books in her spare time.",
            "The restaurant serves excellent Italian food.",
            "We should plan our vacation for next summer.",
            "The new software update includes several improvements.",
            "He works as a software engineer at a tech company.",
            "The garden looks beautiful with all the flowers blooming.",
            "I'm learning to play the guitar in my free time.",
            "The movie was entertaining but a bit too long.",
            "She graduated from university with honors last year.",
            "The traffic was heavy during rush hour this morning.",
            "We're planning to renovate the kitchen next month.",
            "The conference will feature speakers from around the world.",
            "He enjoys hiking in the mountains on weekends.",
            "The book provides valuable insights into modern psychology.",
            "She's working on a research project about climate change.",
            "The team celebrated their victory with a dinner party."
        ]
        
        # Calculate perplexity for reference texts
        reference_perplexities = []
        for text in reference_texts:
            score = self._calculate_perplexity(text)
            reference_perplexities.append(score.perplexity)
        
        # Calculate statistics
        self.reference_samples = reference_perplexities
        self.reference_perplexity_mean = statistics.mean(reference_perplexities)
        self.reference_perplexity_std = statistics.stdev(reference_perplexities) if len(reference_perplexities) > 1 else 1.0
        
        logger.info(f"Reference distribution built: mean={self.reference_perplexity_mean:.2f}, "
                   f"std={self.reference_perplexity_std:.2f}")
    
    def _calculate_membership_probability(self, prompt_score: PerplexityScore) -> float:
        """
        Calculate the probability that the prompt was in training data.
        
        Args:
            prompt_score: Perplexity score for the prompt
            
        Returns:
            Probability between 0 and 1
        """
        if self.reference_perplexity_mean is None:
            # Fallback: simple threshold-based approach
            if prompt_score.perplexity < self.perplexity_threshold:
                return 0.9  # High confidence
            else:
                return 0.1  # Low confidence
        
        # Statistical approach: compare to reference distribution
        z_score = (self.reference_perplexity_mean - prompt_score.perplexity) / self.reference_perplexity_std
        
        # Convert z-score to probability using sigmoid function
        # Higher z-score (lower perplexity than reference) = higher membership probability
        membership_prob = 1.0 / (1.0 + np.exp(-z_score))
        
        return membership_prob
    
    def find_matches(self, query: str, candidates: List[Dict[str, Any]], threshold: float = 0.7, max_matches: int = 10) -> List[MatchResult]:
        """
        Find potential contamination using membership inference.
        
        Args:
            query: Input query to analyze
            candidates: List of benchmark candidates (dictionaries) to search in
            threshold: Minimum confidence threshold
            max_matches: Maximum number of matches to return
            
        Returns:
            List of match results with contamination probabilities
        """
        # Build reference distribution if needed
        self._build_reference_distribution(candidates)
        
        # Calculate perplexity for the query
        query_score = self._calculate_perplexity(query)
        
        # Calculate membership probability
        membership_prob = self._calculate_membership_probability(query_score)
        
        logger.info(f"Membership inference: perplexity={query_score.perplexity:.2f}, "
                   f"membership_prob={membership_prob:.3f}")
        
        # If membership probability is below threshold, no contamination detected
        if membership_prob < threshold:
            return []
        
        # Find the most likely source benchmarks by comparing with benchmark questions
        candidate_matches = []
        
        for candidate in candidates:
            benchmark_name = candidate.get('benchmark_name', 'unknown')
            question_text = candidate.get('question_text', '')
            
            if not question_text:
                continue
            
            # Calculate perplexity for this candidate question
            candidate_score = self._calculate_perplexity(question_text)
            
            # Calculate similarity based on perplexity closeness
            perplexity_diff = abs(query_score.perplexity - candidate_score.perplexity)
            max_diff = max(query_score.perplexity, candidate_score.perplexity, 1.0)
            similarity_score = max(0.0, 1.0 - (perplexity_diff / max_diff))
            
            # Weight by membership probability
            final_score = similarity_score * membership_prob
            
            if final_score >= threshold:
                match = MatchResult(
                    text=f"Membership inference suggests contamination from {benchmark_name}",
                    similarity_score=final_score,
                    exact_match=False,
                    metadata={
                        'benchmark_name': benchmark_name,
                        'benchmark_type': candidate.get('benchmark_type', 'unknown'),
                        'query_perplexity': query_score.perplexity,
                        'candidate_perplexity': candidate_score.perplexity,
                        'membership_probability': membership_prob,
                        'detection_method': 'membership_inference',
                        'statistical_significance': perplexity_diff < (self.reference_perplexity_std or 1.0),
                        'matched_text': question_text,
                        'source_url': candidate.get('source_url', ''),
                        'publication_date': candidate.get('publication_date', ''),
                        'dataset_version': candidate.get('dataset_version', '')
                    }
                )
                candidate_matches.append(match)
        
        # Sort by similarity score and limit results
        candidate_matches.sort(key=lambda x: x.similarity_score, reverse=True)
        result_matches = candidate_matches[:max_matches]
        
        logger.info(f"Membership inference found {len(result_matches)} potential contamination sources")
        return result_matches
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts using membership inference approach.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Calculate perplexity for both texts
        score1 = self._calculate_perplexity(text1)
        score2 = self._calculate_perplexity(text2)
        
        # Calculate similarity based on perplexity closeness
        perplexity_diff = abs(score1.perplexity - score2.perplexity)
        max_diff = max(score1.perplexity, score2.perplexity, 1.0)
        similarity = max(0.0, 1.0 - (perplexity_diff / max_diff))
        
        return similarity
