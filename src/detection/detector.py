"""
Core benchmark detection engine for IsItBenchmark.

This module implements the main BenchmarkDetector class that analyzes prompts
and determines the probability they originate from known benchmark datasets.
"""

import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

from .models import DetectionResult, BenchmarkMatch, BenchmarkType, BenchmarkInfo, MatchResult
from ..similarity import create_matcher, MatcherType, BaseMatcher
from ..benchmarks.database import BenchmarkDatabase
from ..utils.preprocessing import TextPreprocessor


class BenchmarkDetector:
    """
    Main detector class for analyzing prompts and detecting benchmark contamination.
    
    This class combines multiple detection strategies:
    1. Exact text matching for direct copies
    2. Fuzzy text matching for slight modifications
    3. Semantic similarity matching for paraphrased content
    4. Pattern-based detection for common benchmark structures
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        database_path: Optional[str] = None,
        similarity_threshold: float = 0.7,
        enable_semantic_matching: bool = True,
        enable_fuzzy_matching: bool = True,
        max_matches: int = 10,
        scope: str = "all",
    ):
        """
        Initialize the benchmark detector.
        
        Args:
            config: Configuration dictionary (new style)
            database_path: Path to benchmark database (legacy)
            similarity_threshold: Minimum similarity score for matches (legacy)
            enable_semantic_matching: Whether to use semantic similarity (legacy)
            enable_fuzzy_matching: Whether to use fuzzy matching (legacy)
            max_matches: Maximum number of matches (legacy)
            scope: Benchmark scope - 'all' for all benchmarks, 'safety' for safety benchmarks only
        """
        # Handle both new config-based and legacy parameter-based initialization
        if config is not None:
            self.config = config
            # Use config values
            self.matcher_type = self.config.get("matcher_type", "semantic")
            self.matcher_config = self.config.get("matcher_config", {})
            self.similarity_threshold = self.config.get("similarity_threshold", 0.7)
            self.max_matches = self.config.get("max_matches", 10)
            self.enable_fuzzy_matching = self.config.get("enable_fuzzy_matching", True)
            self.scope = self.config.get("scope", scope)
        else:
            # Legacy parameter-based initialization
            self.config = {}
            self.matcher_type = "semantic" if enable_semantic_matching else "text"
            self.matcher_config = {}
            self.similarity_threshold = similarity_threshold
            self.max_matches = max_matches
            self.enable_fuzzy_matching = enable_fuzzy_matching
            self.scope = scope
        
        self.matcher = None
        self.enable_semantic_matching = self.matcher_type == "semantic"
        
        # Initialize components
        self.logger = logging.getLogger(__name__)
        self.preprocessor = TextPreprocessor()
        self.database = BenchmarkDatabase()
        
        # Initialize matcher
        self._initialize_matcher()
        
        self.logger.info(f"BenchmarkDetector initialized with {self.matcher_type} matcher successfully")
    
    def analyze(self, prompt: str, **kwargs) -> DetectionResult:
        """
        Analyze a prompt for potential benchmark contamination.
        
        Args:
            prompt: The input prompt to analyze
            **kwargs: Additional analysis options
            
        Returns:
            DetectionResult containing probability, matches, and metadata
        """
        start_time = time.time()
        
        try:
            # Preprocess the input prompt
            processed_prompt = self.preprocessor.clean_text(prompt)
            
            # Get filtered benchmarks based on scope
            benchmark_infos = self._get_filtered_benchmarks()
            # Convert BenchmarkInfo objects to dictionaries for matcher compatibility
            candidates = [benchmark.to_dict() for benchmark in benchmark_infos]
            
            # Collect all matches from different detection methods
            all_matches = []
            
            # 1. Exact text matching
            exact_matches = self._find_exact_matches(processed_prompt, candidates)
            all_matches.extend(exact_matches)
            
            # 2. Fuzzy text matching
            if self.enable_fuzzy_matching:
                fuzzy_matches = self._find_fuzzy_matches(processed_prompt, candidates)
                all_matches.extend(fuzzy_matches)
            
            # 3. Semantic/LLM matching using the configured matcher
            try:
                logger.info(f"About to call matcher.find_matches with {len(candidates)} candidates")
                matches = self.matcher.find_matches(
                    query=processed_prompt,
                    candidates=candidates,
                    threshold=self.similarity_threshold,
                    max_matches=self.max_matches
                )
                logger.info(f"Matcher returned {len(matches)} matches")
                for i, match in enumerate(matches[:3]):  # Log first 3 matches
                    logger.info(f"Match {i}: type={type(match)}, similarity={match.similarity_score}, metadata_keys={list(match.metadata.keys()) if hasattr(match, 'metadata') else 'no metadata'}")
                all_matches.extend(matches)
            except Exception as e:
                logger.error(f"Error in matcher.find_matches: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
            
            # Remove duplicates and sort by similarity score
            unique_matches = self._deduplicate_matches(all_matches)
            top_matches = sorted(
                unique_matches, 
                key=lambda m: m.similarity_score, 
                reverse=True
            )[:self.max_matches]
            
            # Convert MatchResult objects to BenchmarkMatch objects for probability calculation
            benchmark_matches = self._convert_to_benchmark_matches(top_matches)
            
            # Calculate overall probability
            probability = self._calculate_probability(benchmark_matches)
            
            # Gather metadata
            analysis_time_ms = (time.time() - start_time) * 1000
            metadata = {
                "processed_prompt_length": len(processed_prompt),
                "total_matches_found": len(all_matches),
                "unique_matches": len(unique_matches),
                "detection_methods_used": self._get_detection_methods_used(),
                "similarity_threshold": self.similarity_threshold,
                "matcher_type": self.matcher_type,
                "matcher_info": self.matcher.get_matcher_info() if self.matcher else {},
            }
            
            return DetectionResult(
                input_prompt=prompt,
                probability=probability,
                confidence=None,  # Will be auto-set in __post_init__
                matches=benchmark_matches,
                analysis_time_ms=analysis_time_ms,
                metadata=metadata,
            )
            
        except Exception as e:
            self.logger.error(f"Error during analysis: {str(e)}")
            # Return a safe default result
            return DetectionResult(
                input_prompt=prompt,
                probability=0.0,
                confidence=None,
                matches=[],
                analysis_time_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(e)},
            )
    
    def _find_exact_matches(self, prompt: str, candidates: List[BenchmarkInfo]) -> List[BenchmarkMatch]:
        """Find exact text matches in the filtered benchmark candidates."""
        matches = []
        
        # Create a set of allowed benchmark names for filtering
        allowed_benchmarks = {benchmark['name'] for benchmark in candidates}
        
        # Query database for exact matches
        exact_results = self.database.find_exact_matches(prompt)
        
        for result in exact_results:
            # Only include matches from filtered benchmarks
            if result["benchmark_name"] in allowed_benchmarks:
                match = BenchmarkMatch(
                    benchmark_name=result["benchmark_name"],
                    benchmark_type=BenchmarkType(result["benchmark_type"]),
                    similarity_score=1.0,  # Exact match
                    exact_match=True,
                    matched_text=result["matched_text"],
                    source_url=result.get("source_url"),
                    publication_date=result.get("publication_date"),
                    dataset_version=result.get("dataset_version"),
                )
                matches.append(match)
        
        return matches
    
    def _find_fuzzy_matches(self, prompt: str, candidates: List[BenchmarkInfo]) -> List[MatchResult]:
        """Find fuzzy text matches using string similarity algorithms in filtered benchmark candidates."""
        matches = []
        
        # Create a set of allowed benchmark names for filtering
        allowed_benchmarks = {benchmark['name'] for benchmark in candidates}
        
        # Use search_questions for fuzzy matching (SQL LIKE matching)
        search_results = self.database.search_questions(prompt, limit=self.max_matches)
        
        for result in search_results:
            # Only include matches from filtered benchmarks
            if result.get("benchmark_name", "Unknown") in allowed_benchmarks:
                # Calculate a simple similarity score based on text overlap
                similarity_score = self._calculate_text_similarity(prompt, result.get("question_text", ""))
                
                if similarity_score >= self.similarity_threshold:
                    match = MatchResult(
                        text=result.get("question_text", ""),
                        similarity_score=similarity_score,
                        exact_match=False,
                        metadata={
                            "benchmark_name": result.get("benchmark_name", "Unknown"),
                            "benchmark_type": result.get("benchmark_type", "unknown"),
                            "source_url": result.get("source_url"),
                            "publication_date": result.get("publication_date"),
                            "category": result.get("category"),
                            "difficulty": result.get("difficulty"),
                        }
                    )
                    matches.append(match)
        
        return matches
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity based on word overlap."""
        if not text1 or not text2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _deduplicate_matches(self, matches: List[MatchResult]) -> List[MatchResult]:
        """Remove duplicate matches based on benchmark name and similarity."""
        seen = set()
        unique_matches = []
        
        for match in matches:
            # Create a key based on benchmark name and matched text
            benchmark_name = match.metadata.get("benchmark_name", "Unknown")
            key = (benchmark_name, match.text)
            
            if key not in seen:
                seen.add(key)
                unique_matches.append(match)
            else:
                # If we've seen this match before, keep the one with higher similarity
                for i, existing_match in enumerate(unique_matches):
                    existing_benchmark_name = existing_match.metadata.get("benchmark_name", "Unknown")
                    if (existing_benchmark_name == benchmark_name and 
                        existing_match.text == match.text):
                        if match.similarity_score > existing_match.similarity_score:
                            unique_matches[i] = match
                        break
        
        return unique_matches
    
    def _convert_to_benchmark_matches(self, match_results: List[MatchResult]) -> List[BenchmarkMatch]:
        """Convert MatchResult objects to BenchmarkMatch objects."""
        benchmark_matches = []
        
        for match in match_results:
            benchmark_name = match.metadata.get("benchmark_name", "Unknown")
            benchmark_type_str = match.metadata.get("benchmark_type", "unknown")
            
            # Convert string to BenchmarkType enum
            try:
                if hasattr(benchmark_type_str, 'upper'):
                    benchmark_type = BenchmarkType(benchmark_type_str.upper())
                else:
                    benchmark_type = BenchmarkType.SPECIALIZED
            except (ValueError, AttributeError):
                benchmark_type = BenchmarkType.SPECIALIZED
            
            benchmark_match = BenchmarkMatch(
                benchmark_name=benchmark_name,
                benchmark_type=benchmark_type,
                similarity_score=match.similarity_score,
                exact_match=match.exact_match,
                matched_text=match.text,
                source_url=match.metadata.get("source_url", ""),
                publication_date=match.metadata.get("publication_date", ""),
                dataset_version=match.metadata.get("dataset_version", "")
            )
            benchmark_matches.append(benchmark_match)
        
        return benchmark_matches
    
    def _calculate_probability(self, matches: List[BenchmarkMatch]) -> float:
        """
        Calculate overall probability of benchmark contamination.
        
        This uses a weighted combination of:
        1. Highest similarity score
        2. Number of matches
        3. Exact match bonus
        4. Benchmark type diversity
        """
        if not matches:
            return 0.0
        
        # Base probability from highest similarity score
        max_similarity = max(match.similarity_score for match in matches)
        base_probability = max_similarity
        
        # Bonus for exact matches
        exact_match_bonus = 0.1 if any(match.exact_match for match in matches) else 0.0
        
        # Bonus for multiple matches (diminishing returns)
        num_matches = len(matches)
        multiple_match_bonus = min(0.2, (num_matches - 1) * 0.05)
        
        # Bonus for diverse benchmark types
        unique_types = len(set(match.benchmark_type for match in matches))
        diversity_bonus = min(0.1, (unique_types - 1) * 0.03)
        
        # Combine all factors
        total_probability = min(1.0, 
            base_probability + 
            exact_match_bonus + 
            multiple_match_bonus + 
            diversity_bonus
        )
        
        return round(total_probability, 3)
    
    def _get_filtered_benchmarks(self) -> List[BenchmarkInfo]:
        """Get benchmarks filtered by scope parameter."""
        all_benchmarks = self.database.get_all_benchmarks()
        
        if self.scope == "safety":
            # Filter to only include safety evaluation benchmarks
            safety_benchmarks = [
                benchmark for benchmark in all_benchmarks
                if benchmark.type == BenchmarkType.SAFETY_EVALUATION
            ]
            self.logger.info(f"Filtered to {len(safety_benchmarks)} safety benchmarks out of {len(all_benchmarks)} total")
            return safety_benchmarks
        else:
            # Return all benchmarks (default behavior)
            self.logger.info(f"Using all {len(all_benchmarks)} benchmarks")
            return all_benchmarks
    
    def _get_detection_methods_used(self) -> List[str]:
        """Get list of detection methods that are enabled."""
        methods = ["exact_matching"]
        
        if self.enable_fuzzy_matching:
            methods.append("fuzzy_matching")
        
        if self.matcher_type == "semantic":
            methods.append("semantic_matching")
        
        return methods
    
    def get_supported_benchmarks(self) -> List[BenchmarkInfo]:
        """Get list of all supported benchmarks."""
        return self.database.get_all_benchmarks()
    
    def get_benchmark_info(self, benchmark_name: str) -> Optional[BenchmarkInfo]:
        """Get detailed information about a specific benchmark."""
        return self.database.get_benchmark_info(benchmark_name)
    
    def _initialize_matcher(self):
        """Initialize the similarity matcher based on configuration."""
        try:
            self.matcher = create_matcher(self.matcher_type, self.matcher_config)
            self.logger.info(f"Initialized {self.matcher_type} matcher successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.matcher_type} matcher: {str(e)}")
            # Fallback to semantic matcher
            self.logger.info("Falling back to semantic matcher")
            self.matcher = create_matcher("semantic", {})
            self.matcher_type = "semantic"
    
    def update_database(self, force_refresh: bool = False) -> bool:
        """Update the benchmark database with latest data."""
        try:
            return self.database.update(force_refresh=force_refresh)
        except Exception as e:
            self.logger.error(f"Failed to update database: {str(e)}")
            return False
