"""Text similarity matching for IsItBenchmark.

This module provides text-based similarity matching using various techniques
including exact matching, fuzzy matching, and semantic similarity.
"""

import re
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from difflib import SequenceMatcher
from collections import Counter
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .base_matcher import BaseMatcher, MatcherType
from ..detection.models import MatchResult, BenchmarkType
from ..utils.preprocessing import TextPreprocessor


class TextMatcher(BaseMatcher):
    """Text similarity matcher using multiple matching strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Configuration
        self.use_semantic = self.config.get("use_semantic", True)
        self.semantic_model_name = self.config.get("semantic_model", "all-MiniLM-L6-v2")
        self.exact_threshold = self.config.get("exact_threshold", 0.95)
        self.fuzzy_threshold = self.config.get("fuzzy_threshold", 0.8)
        self.semantic_threshold = self.config.get("semantic_threshold", 0.7)
        
        # Initialize components
        self.preprocessor = TextPreprocessor()
        self.stemmer = PorterStemmer()
        self.semantic_model = None
        self.stop_words = set()
        
        self.logger = logging.getLogger(__name__)
    
    def find_matches(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]], 
        threshold: float = 0.7,
        max_matches: int = 10
    ) -> List[MatchResult]:
        """
        Find similar texts using multiple matching strategies.
        
        Args:
            query: Text to match against
            candidates: List of candidate texts with metadata
            threshold: Minimum similarity threshold
            max_matches: Maximum number of matches to return
            
        Returns:
            List of match results sorted by similarity score
        """
        if not self.is_initialized:
            raise RuntimeError("TextMatcher not initialized")
        
        results = []
        
        for candidate in candidates:
            similarity_scores = self.compute_similarity(query, candidate["text"])
            overall_score = max(similarity_scores["exact"], similarity_scores["fuzzy"], similarity_scores["semantic"])
            
            if overall_score >= threshold:
                result = {
                    "benchmark_name": candidate["benchmark_name"],
                    "benchmark_type": candidate["benchmark_type"],
                    "similarity_score": overall_score,
                    "matched_text": candidate["text"],
                    "source_url": candidate["source_url"],
                    "publication_date": candidate["publication_date"],
                    "matched_text": benchmark["text"],
                    "source_url": benchmark["source_url"],
                    "publication_date": benchmark["publication_date"],
                    "detailed_scores": similarity_scores,
                }
                results.append(result)
        
        # Sort by similarity score and limit results
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:max_results]
    
    def batch_similarity(
        self, 
        queries: List[str], 
        threshold: float = 0.7
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Calculate similarities for multiple queries efficiently.
        
        Args:
            queries: List of query texts
            threshold: Minimum similarity threshold
            
        Returns:
            Dictionary mapping queries to their similarity results
        """
        results = {}
        
        for query in queries:
            results[query] = self.find_similar(query, threshold)
        
        return results
