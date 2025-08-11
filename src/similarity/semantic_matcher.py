"""
Semantic similarity matching using embeddings and neural models.

This module implements semantic similarity detection for identifying
paraphrased or semantically equivalent benchmark questions.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import torch

try:
    from .base_matcher import BaseMatcher, MatcherType
    from ..detection.models import MatchResult
except ImportError:
    # Fallback for standalone execution
    import sys
    import os
    current_dir = os.path.dirname(__file__)
    sys.path.append(current_dir)
    sys.path.append(os.path.join(current_dir, '..', 'detection'))
    from base_matcher import BaseMatcher, MatcherType
    from models import MatchResult


class SemanticMatcher(BaseMatcher):
    """
    Semantic similarity matcher using sentence embeddings.
    
    Uses pre-trained sentence transformer models to detect semantic
    similarity between prompts and benchmark questions.
    """
    
    def __init__(
        self, 
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the semantic matcher.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Extract configuration
        self.model_name = self.config.get("semantic_model", "all-MiniLM-L6-v2")
        self.batch_size = self.config.get("batch_size", 32)
        device = self.config.get("device")
        
        self.logger = logging.getLogger(__name__)
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load the sentence transformer model
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.logger.info(f"Loaded semantic model: {self.model_name} on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load semantic model: {str(e)}")
            self.model = None
    
    def validate_config(self) -> bool:
        """Validate the matcher configuration."""
        try:
            # Check if model name is valid
            model_name = self.config.get("semantic_model", "all-MiniLM-L6-v2")
            if not isinstance(model_name, str) or not model_name.strip():
                return False
            
            # Check batch size
            batch_size = self.config.get("batch_size", 32)
            if not isinstance(batch_size, int) or batch_size <= 0:
                return False
            
            return True
        except Exception:
            return False
    
    def find_matches(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        threshold: float = 0.7,
        max_matches: int = 10
    ) -> List[MatchResult]:
        """Find semantic matches using sentence embeddings."""
        if not self.model:
            self.logger.warning("Semantic model not available, returning empty results")
            return []
        
        if not query.strip():
            return []
        
        results = []
        
        try:
            # For now, use the mock implementation
            # In a real implementation, this would use the candidates parameter
            mock_results = self._mock_semantic_query(query, threshold, max_matches)
            
            for result in mock_results:
                match_result = MatchResult(
                    text=result["matched_text"],
                    similarity_score=result["similarity_score"],
                    exact_match=result["similarity_score"] >= 0.95,
                    metadata={
                        "benchmark_name": result["benchmark_name"],
                        "benchmark_type": result["benchmark_type"],
                        "source_url": result.get("source_url"),
                        "publication_date": result.get("publication_date"),
                        "embedding_model": result.get("embedding_model"),
                    }
                )
                results.append(match_result)
            
        except Exception as e:
            self.logger.error(f"Error finding semantic matches: {str(e)}")
        
        return results
    
    def get_matcher_info(self) -> Dict[str, Any]:
        """Get information about this matcher."""
        info = super().get_matcher_info()
        info.update({
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "model_loaded": self.model is not None,
        })
        
        if self.model:
            try:
                info["embedding_dimension"] = self.model.get_sentence_embedding_dimension()
            except Exception:
                pass
        
        return info
    
    def initialize(self) -> bool:
        """Initialize the matcher."""
        try:
            # Model is already loaded in __init__, just validate it's working
            if self.model is None:
                return False
            
            # Test the model with a simple encoding
            test_embedding = self.model.encode(["test"])
            return test_embedding is not None and len(test_embedding) > 0
        except Exception as e:
            self.logger.error(f"Failed to initialize semantic matcher: {str(e)}")
            return False
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts."""
        return self.calculate_semantic_similarity(text1, text2)
    
    def find_similar(
        self, 
        query: str, 
        threshold: float = 0.7,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Find semantically similar texts in the benchmark database.
        
        Args:
            query: Input text to match against
            threshold: Minimum similarity threshold (0.0 to 1.0)
            max_results: Maximum number of results to return
            
        Returns:
            List of similarity results with scores and metadata
        """
        if not self.model:
            self.logger.warning("Semantic model not available, returning empty results")
            return []
        
        if not query.strip():
            return []
        
        # This is a placeholder - in a real implementation, this would
        # query the actual benchmark database with pre-computed embeddings
        return self._mock_semantic_query(query, threshold, max_results)
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        if not self.model:
            return 0.0
        
        try:
            # Generate embeddings
            embeddings = self.model.encode([text1, text2])
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(embeddings[0], embeddings[1])
            
            return float(similarity)
        except Exception as e:
            self.logger.error(f"Error calculating semantic similarity: {str(e)}")
            return 0.0
    
    def batch_encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode multiple texts into embeddings efficiently.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Array of embeddings
        """
        if not self.model:
            return np.array([])
        
        try:
            embeddings = self.model.encode(
                texts, 
                batch_size=self.batch_size,
                show_progress_bar=len(texts) > 100
            )
            return embeddings
        except Exception as e:
            self.logger.error(f"Error in batch encoding: {str(e)}")
            return np.array([])
    
    def _cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        try:
            # Normalize vectors
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            
            # Ensure result is in [0, 1] range
            return max(0.0, min(1.0, (similarity + 1) / 2))
        except:
            return 0.0
    
    def _mock_semantic_query(
        self, 
        query: str, 
        threshold: float, 
        max_results: int
    ) -> List[Dict[str, Any]]:
        """
        Mock semantic database query for demonstration purposes.
        In a real implementation, this would use pre-computed embeddings.
        """
        # Sample benchmark questions with semantic variations
        sample_benchmarks = [
            {
                "text": "What is the capital city of France?",
                "benchmark_name": "MMLU",
                "benchmark_type": "language_understanding",
                "source_url": "https://github.com/hendrycks/test",
                "publication_date": "2020-09-07",
            },
            {
                "text": "Paris is the capital of which European country?",
                "benchmark_name": "MMLU",
                "benchmark_type": "language_understanding",
                "source_url": "https://github.com/hendrycks/test",
                "publication_date": "2020-09-07",
            },
            {
                "text": "Find the value of x when 2x + 5 equals 13",
                "benchmark_name": "GSM8K",
                "benchmark_type": "mathematical_reasoning",
                "source_url": "https://github.com/openai/grade-school-math",
                "publication_date": "2021-10-27",
            },
            {
                "text": "Create a function that computes the factorial of a given number",
                "benchmark_name": "HumanEval",
                "benchmark_type": "code_generation",
                "source_url": "https://github.com/openai/human-eval",
                "publication_date": "2021-07-07",
            },
            {
                "text": "When someone enters a library, what is their most probable action?",
                "benchmark_name": "CommonsenseQA",
                "benchmark_type": "common_sense",
                "source_url": "https://www.tau-nlp.org/commonsenseqa",
                "publication_date": "2019-04-01",
            },
        ]
        
        results = []
        
        for benchmark in sample_benchmarks:
            similarity_score = self.calculate_semantic_similarity(query, benchmark["text"])
            
            if similarity_score >= threshold:
                result = {
                    "benchmark_name": benchmark["benchmark_name"],
                    "benchmark_type": benchmark["benchmark_type"],
                    "similarity_score": similarity_score,
                    "matched_text": benchmark["text"],
                    "source_url": benchmark["source_url"],
                    "publication_date": benchmark["publication_date"],
                    "embedding_model": self.model_name,
                }
                results.append(result)
        
        # Sort by similarity score and limit results
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:max_results]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.model:
            return {"status": "not_loaded"}
        
        return {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "status": "loaded"
        }
