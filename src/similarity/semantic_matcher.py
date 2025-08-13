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
            # Real semantic matching against database questions
            results = self._perform_real_semantic_matching(query, candidates, threshold, max_matches)
            
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
    
    def _perform_real_semantic_matching(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]], 
        threshold: float, 
        max_matches: int
    ) -> List[MatchResult]:
        """
        Perform real semantic matching against database questions.
        
        Args:
            query: Input query to match
            candidates: List of benchmark candidates (dictionaries)
            threshold: Minimum similarity threshold
            max_matches: Maximum number of matches to return
            
        Returns:
            List of MatchResult objects
        """
        results = []
        
        try:
            import sqlite3
            import os
            
            # Connect to database
            db_path = os.path.join(os.getcwd(), 'data', 'benchmarks.db')
            if not os.path.exists(db_path):
                self.logger.warning(f"Database not found at {db_path}")
                return results
                
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get benchmark names from candidates
            benchmark_names = [candidate.get('name', '') for candidate in candidates]
            benchmark_names = [name for name in benchmark_names if name]  # Filter empty names
            
            if not benchmark_names:
                self.logger.warning("No valid benchmark names in candidates")
                return results
            
            # Query database for questions from specified benchmarks
            # Only select questions with non-empty text and prioritize benchmarks with content
            placeholders = ','.join(['?' for _ in benchmark_names])
            query_sql = f'''
            SELECT q.question_text, b.name, b.type 
            FROM questions q 
            JOIN benchmarks b ON q.benchmark_id = b.id 
            WHERE b.name IN ({placeholders}) 
                AND q.question_text IS NOT NULL 
                AND q.question_text != ''
            ORDER BY b.name
            LIMIT 1000
            '''
            
            self.logger.info(f"Executing query with benchmark names: {benchmark_names}")
            self.logger.info(f"SQL query: {query_sql}")
            cursor.execute(query_sql, benchmark_names)
            db_questions = cursor.fetchall()
            
            self.logger.info(f"Retrieved {len(db_questions)} questions from database for semantic matching")
            
            # Debug: show first few questions to verify data quality
            if db_questions:
                self.logger.info(f"Sample questions from database:")
                for i, (question_text, benchmark_name, benchmark_type) in enumerate(db_questions[:3]):
                    self.logger.info(f"  {i+1}. [{benchmark_name}] {question_text}")
            
            if not db_questions:
                self.logger.warning("No questions found in database for specified benchmarks")
                conn.close()
                return results
            
            # Compute embeddings for query
            query_embedding = self.model.encode([query], normalize_embeddings=True)
            
            # Compute embeddings for database questions in batches
            question_texts = [q[0] for q in db_questions]
            question_embeddings = self.model.encode(question_texts, batch_size=self.batch_size, normalize_embeddings=True)
            
            # Calculate similarities using numpy (more robust than sklearn for edge cases)
            import numpy as np
            
            # Ensure embeddings are valid (no NaN or inf)
            query_emb = query_embedding[0]
            if np.any(np.isnan(query_emb)) or np.any(np.isinf(query_emb)):
                self.logger.error("Query embedding contains NaN or inf values")
                return results
            
            # Check for NaN/inf in question embeddings
            valid_indices = []
            valid_embeddings = []
            valid_questions = []
            
            for i, emb in enumerate(question_embeddings):
                if not (np.any(np.isnan(emb)) or np.any(np.isinf(emb))):
                    valid_indices.append(i)
                    valid_embeddings.append(emb)
                    valid_questions.append(db_questions[i])
            
            if not valid_embeddings:
                self.logger.error("No valid question embeddings found")
                return results
            
            self.logger.info(f"Using {len(valid_embeddings)} valid embeddings out of {len(question_embeddings)} total")
            
            # Calculate cosine similarities manually
            valid_embeddings = np.array(valid_embeddings)
            similarities = np.dot(valid_embeddings, query_emb)
            
            # Update db_questions to only include valid ones
            db_questions = valid_questions
            
            # Find matches above threshold and log top scores for debugging
            all_matches = []
            matches = []
            for i, (similarity, (question_text, benchmark_name, benchmark_type)) in enumerate(zip(similarities, db_questions)):
                all_matches.append({
                    'similarity': similarity,
                    'question': question_text,
                    'benchmark_name': benchmark_name,
                    'benchmark_type': benchmark_type
                })
                if similarity >= threshold:
                    matches.append({
                        'similarity': similarity,
                        'question': question_text,
                        'benchmark_name': benchmark_name,
                        'benchmark_type': benchmark_type
                    })
            
            # Sort all matches by similarity for debugging
            all_matches.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Log top 5 similarity scores for debugging
            self.logger.info(f"Top 5 similarity scores: {[f'{m["similarity"]:.4f}' for m in all_matches[:5]]}")
            if all_matches:
                self.logger.info(f"Highest similarity: {all_matches[0]['similarity']:.4f} for question: '{all_matches[0]['question']}'")
                # Debug embedding values
                self.logger.info(f"Query embedding shape: {query_emb.shape}, norm: {np.linalg.norm(query_emb):.4f}")
                self.logger.info(f"First question embedding shape: {valid_embeddings[0].shape}, norm: {np.linalg.norm(valid_embeddings[0]):.4f}")
            
            # Sort matches above threshold by similarity (highest first) and limit results
            matches.sort(key=lambda x: x['similarity'], reverse=True)
            matches = matches[:max_matches]
            
            self.logger.info(f"Found {len(matches)} semantic matches above threshold {threshold:.3f}")
            
            # Convert to MatchResult objects
            for match in matches:
                match_result = MatchResult(
                    text=match['question'],
                    similarity_score=float(match['similarity']),
                    exact_match=match['similarity'] >= 0.95,
                    metadata={
                        "benchmark_name": match['benchmark_name'],
                        "benchmark_type": match['benchmark_type'],
                        "embedding_model": self.model_name,
                        "similarity_threshold": threshold
                    }
                )
                results.append(match_result)
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error in real semantic matching: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        
        return results
    
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
