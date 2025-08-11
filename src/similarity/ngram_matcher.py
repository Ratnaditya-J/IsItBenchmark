"""
N-gram overlap detection for benchmark contamination.

This module implements n-gram based contamination detection as used by GPT-3 and GPT-4 teams.
It detects exact and near-exact text reuse by finding overlapping sequences of words.
"""

import re
import logging
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass

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
class NgramMatch:
    """Represents an n-gram match found in the database."""
    ngram: str
    benchmark_name: str
    benchmark_type: str
    question_text: str
    overlap_length: int
    position_in_prompt: int
    position_in_benchmark: int


class NgramMatcher(BaseMatcher):
    """
    N-gram overlap matcher for detecting benchmark contamination.
    
    Implements the approach used by GPT-3 (13-gram) and GPT-4 (50-character) teams
    to detect exact and near-exact text reuse from benchmark datasets.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the N-gram matcher.
        
        Args:
            config: Configuration dictionary containing:
                - ngram_size: Size of n-grams to use (default: 5)
                - char_window_size: Size of character windows (default: 20)
                - min_overlap_threshold: Minimum overlap threshold (default: 0.5)
                - use_word_ngrams: Whether to use word-based n-grams (default: True)
                - use_char_ngrams: Whether to use character-based n-grams (default: True)
        """
        self.config = config
        self.ngram_size = config.get('ngram_size', 5)
        self.char_window_size = config.get('char_window_size', 20)
        self.min_overlap_threshold = config.get('min_overlap_threshold', 0.5)
        self.use_word_ngrams = config.get('use_word_ngrams', True)
        self.use_char_ngrams = config.get('use_char_ngrams', True)
        self.initialized = False
        
        # Precomputed n-gram index for fast lookup
        self.word_ngram_index: Dict[str, List[NgramMatch]] = defaultdict(list)
        self.char_ngram_index: Dict[str, List[NgramMatch]] = defaultdict(list)
        
        logger.info(f"N-gram matcher initialized with config: {config}")
    
    def _build_index_from_candidates(self, candidates: List[Dict[str, Any]]) -> None:
        """
        Build n-gram indices from benchmark candidates.
        
        Args:
            candidates: List of benchmark candidates to index
        """
        logger.info(f"Building n-gram index from {len(candidates)} candidates...")
        
        # Clear existing indices
        self.word_ngram_index.clear()
        self.char_ngram_index.clear()
        
        for candidate in candidates:
            text = candidate.get('question', '') or candidate.get('text', '')
            if not text:
                continue
                
            processed_text = self._preprocess_text(text)
            
            # Build word n-gram index
            if self.use_word_ngrams:
                word_ngrams = self._extract_word_ngrams(processed_text)
                for ngram in word_ngrams:
                    match = NgramMatch(
                        ngram=ngram,
                        source_text=text,
                        benchmark_id=candidate.get('id', ''),
                        benchmark_name=candidate.get('benchmark_name', ''),
                        position=0  # Simplified for now
                    )
                    self.word_ngram_index[ngram].append(match)
            
            # Build character n-gram index
            if self.use_char_ngrams:
                char_ngrams = self._extract_char_ngrams(processed_text)
                for ngram in char_ngrams:
                    match = NgramMatch(
                        ngram=ngram,
                        source_text=text,
                        benchmark_id=candidate.get('id', ''),
                        benchmark_name=candidate.get('benchmark_name', ''),
                        position=0  # Simplified for now
                    )
                    self.char_ngram_index[ngram].append(match)
        
        logger.info(f"Built n-gram index: {len(self.word_ngram_index)} word n-grams, "
                   f"{len(self.char_ngram_index)} char n-grams")
    
    def initialize(self) -> bool:
        """
        Initialize the matcher (no special initialization needed for N-gram matcher).
        
        Returns:
            True if initialization was successful
        """
        try:
            # N-gram matcher doesn't need special initialization
            # Just validate configuration
            if self.ngram_size <= 0:
                logger.error("N-gram size must be positive")
                return False
            if self.char_window_size <= 0:
                logger.error("Character window size must be positive")
                return False
            if not (0.0 <= self.min_overlap_threshold <= 1.0):
                logger.error("Overlap threshold must be between 0.0 and 1.0")
                return False
            
            self.initialized = True
            logger.info("N-gram matcher initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize N-gram matcher: {e}")
            return False
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for n-gram extraction."""
        # Normalize whitespace and convert to lowercase
        text = re.sub(r'\s+', ' ', text.strip().lower())
        return text
    
    def _extract_word_ngrams(self, text: str) -> List[Tuple[str, int]]:
        """
        Extract word-based n-grams from text.
        
        Returns:
            List of (ngram, position) tuples
        """
        words = text.split()
        ngrams = []
        
        for i in range(len(words) - self.ngram_size + 1):
            ngram = ' '.join(words[i:i + self.ngram_size])
            ngrams.append((ngram, i))
        
        return ngrams
    
    def _extract_char_ngrams(self, text: str) -> List[Tuple[str, int]]:
        """
        Extract character-based n-grams from text.
        
        Returns:
            List of (ngram, position) tuples
        """
        ngrams = []
        
        for i in range(len(text) - self.char_window_size + 1):
            ngram = text[i:i + self.char_window_size]
            ngrams.append((ngram, i))
        
        return ngrams
    
    def build_index(self, benchmarks: List[BenchmarkInfo]) -> None:
        """
        Build n-gram index from benchmark data for fast lookup.
        
        Args:
            benchmarks: List of benchmark information to index
        """
        logger.info(f"Building n-gram index for {len(benchmarks)} benchmarks...")
        
        # Clear existing indices
        self.word_ngram_index.clear()
        self.char_ngram_index.clear()
        
        total_questions = 0
        
        for benchmark in benchmarks:
            # Get questions from database for this benchmark
            questions = self.database.get_questions_by_benchmark(benchmark.name)
            total_questions += len(questions)
            
            for question in questions:
                question_text = question.get('question_text', '')
                if not question_text:
                    continue
                
                processed_text = self._preprocess_text(question_text)
                
                # Build word n-gram index
                if self.use_word_ngrams:
                    word_ngrams = self._extract_word_ngrams(processed_text)
                    for ngram, position in word_ngrams:
                        match = NgramMatch(
                            ngram=ngram,
                            benchmark_name=benchmark.name,
                            benchmark_type=benchmark.type.value,
                            question_text=question_text,
                            overlap_length=len(ngram.split()),
                            position_in_prompt=0,  # Will be set during matching
                            position_in_benchmark=position
                        )
                        self.word_ngram_index[ngram].append(match)
                
                # Build character n-gram index
                if self.use_char_ngrams:
                    char_ngrams = self._extract_char_ngrams(processed_text)
                    for ngram, position in char_ngrams:
                        match = NgramMatch(
                            ngram=ngram,
                            benchmark_name=benchmark.name,
                            benchmark_type=benchmark.type.value,
                            question_text=question_text,
                            overlap_length=len(ngram),
                            position_in_prompt=0,  # Will be set during matching
                            position_in_benchmark=position
                        )
                        self.char_ngram_index[ngram].append(match)
        
        logger.info(f"N-gram index built: {len(self.word_ngram_index)} word n-grams, "
                   f"{len(self.char_ngram_index)} char n-grams from {total_questions} questions")
    
    def find_matches(self, query: str, candidates: List[Dict[str, Any]], threshold: float = 0.7, max_matches: int = 10) -> List[MatchResult]:
        """
        Find n-gram overlaps between query and benchmark candidates.
        
        Args:
            query: Input query to analyze
            candidates: List of benchmark candidates (dictionaries) to search in
            threshold: Minimum similarity threshold
            max_matches: Maximum number of matches to return
            
        Returns:
            List of match results with similarity scores
        """
        if not self.word_ngram_index and not self.char_ngram_index:
            # Build index if not already built - convert dict candidates to BenchmarkInfo for indexing
            benchmark_candidates = []
            for candidate in candidates:
                # Convert dictionary to BenchmarkInfo-like structure for indexing
                benchmark_candidates.append(candidate)
            self._build_index_from_candidates(benchmark_candidates)
        
        processed_query = self._preprocess_text(query)
        all_matches = []
        
        # Find word n-gram matches
        if self.use_word_ngrams:
            word_matches = self._find_word_ngram_matches(processed_query, candidates)
            all_matches.extend(word_matches)
        
        # Find character n-gram matches
        if self.use_char_ngrams:
            char_matches = self._find_char_ngram_matches(processed_query, candidates)
            all_matches.extend(char_matches)
        
        # Aggregate and score matches
        aggregated_matches = self._aggregate_matches(all_matches, query)
        
        # Filter by threshold and limit results
        filtered_matches = [match for match in aggregated_matches if match.similarity_score >= threshold]
        filtered_matches.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Limit to max_matches
        result_matches = filtered_matches[:max_matches]
        
        logger.info(f"N-gram matching found {len(result_matches)} matches for query: '{query[:50]}...'")
        return result_matches
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts using n-gram overlap.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        processed_text1 = self._preprocess_text(text1)
        processed_text2 = self._preprocess_text(text2)
        
        # Calculate word n-gram overlap
        word_overlap = 0.0
        if self.use_word_ngrams:
            ngrams1 = set([ngram for ngram, _ in self._extract_word_ngrams(processed_text1)])
            ngrams2 = set([ngram for ngram, _ in self._extract_word_ngrams(processed_text2)])
            if ngrams1 or ngrams2:
                word_overlap = len(ngrams1.intersection(ngrams2)) / max(len(ngrams1.union(ngrams2)), 1)
        
        # Calculate character n-gram overlap
        char_overlap = 0.0
        if self.use_char_ngrams:
            ngrams1 = set([ngram for ngram, _ in self._extract_char_ngrams(processed_text1)])
            ngrams2 = set([ngram for ngram, _ in self._extract_char_ngrams(processed_text2)])
            if ngrams1 or ngrams2:
                char_overlap = len(ngrams1.intersection(ngrams2)) / max(len(ngrams1.union(ngrams2)), 1)
        
        # Combine word and character overlap scores
        if self.use_word_ngrams and self.use_char_ngrams:
            similarity = (word_overlap + char_overlap) / 2.0
        elif self.use_word_ngrams:
            similarity = word_overlap
        else:
            similarity = char_overlap
        
        return similarity
    
    def _find_word_ngram_matches(self, prompt: str, candidates: List[Dict[str, Any]]) -> List[NgramMatch]:
        """Find word-based n-gram matches."""
        matches = []
        candidate_names = {b.get('benchmark_name', b.get('name', '')) for b in candidates}
        
        word_ngrams = self._extract_word_ngrams(prompt)
        
        for ngram, prompt_position in word_ngrams:
            if ngram in self.word_ngram_index:
                for match in self.word_ngram_index[ngram]:
                    # Only include matches from candidate benchmarks
                    if match.benchmark_name in candidate_names:
                        # Update prompt position
                        match.position_in_prompt = prompt_position
                        matches.append(match)
        
        return matches
    
    def _find_char_ngram_matches(self, prompt: str, candidates: List[Dict[str, Any]]) -> List[NgramMatch]:
        """Find character-based n-gram matches."""
        matches = []
        candidate_names = {b.get('benchmark_name', b.get('name', '')) for b in candidates}
        
        char_ngrams = self._extract_char_ngrams(prompt)
        
        for ngram, prompt_position in char_ngrams:
            if ngram in self.char_ngram_index:
                for match in self.char_ngram_index[ngram]:
                    # Only include matches from candidate benchmarks
                    if match.benchmark_name in candidate_names:
                        # Update prompt position
                        match.position_in_prompt = prompt_position
                        matches.append(match)
        
        return matches
    
    def _aggregate_matches(self, matches: List[NgramMatch], original_prompt: str) -> List[MatchResult]:
        """
        Aggregate n-gram matches by benchmark question and calculate similarity scores.
        """
        # Group matches by (benchmark_name, question_text)
        grouped_matches = defaultdict(list)
        
        for match in matches:
            key = (match.benchmark_name, match.question_text)
            grouped_matches[key].append(match)
        
        aggregated_results = []
        
        for (benchmark_name, question_text), question_matches in grouped_matches.items():
            # Calculate overlap statistics
            total_overlap_chars = sum(len(match.ngram) for match in question_matches)
            unique_ngrams = len(set(match.ngram for match in question_matches))
            
            # Calculate similarity score based on overlap
            prompt_length = len(original_prompt)
            question_length = len(question_text)
            max_possible_overlap = min(prompt_length, question_length)
            
            if max_possible_overlap == 0:
                similarity_score = 0.0
            else:
                # Normalize by the maximum possible overlap
                similarity_score = min(1.0, total_overlap_chars / max_possible_overlap)
            
            # Get benchmark type from first match
            benchmark_type = question_matches[0].benchmark_type
            
            # Create aggregated match result
            result = MatchResult(
                text=question_text,
                similarity_score=similarity_score,
                exact_match=similarity_score >= 0.95,  # Consider 95%+ as exact match
                metadata={
                    'benchmark_name': benchmark_name,
                    'benchmark_type': benchmark_type,
                    'total_overlap_chars': total_overlap_chars,
                    'unique_ngrams': unique_ngrams,
                    'match_count': len(question_matches),
                    'detection_method': 'ngram_overlap'
                }
            )
            
            aggregated_results.append(result)
        
        return aggregated_results
