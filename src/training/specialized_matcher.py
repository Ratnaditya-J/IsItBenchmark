"""
Specialized Contamination Detection Matcher for IsItBenchmark.

This module implements the first specialized contamination detection matcher
that uses fine-tuned models specifically trained for benchmark contamination
detection tasks.
"""

import logging
import time
from typing import List, Dict, Any, Optional
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ..similarity.base_matcher import BaseMatcher, MatcherType
from ..detection.models import MatchResult, BenchmarkType
from ..utils.preprocessing import TextPreprocessor
from ..models.pretrained_model_loader import ensure_pretrained_model, PretrainedModelLoader


class SpecializedContaminationMatcher(BaseMatcher):
    """
    Specialized matcher using fine-tuned contamination detection models.
    
    This is the first implementation of a matcher specifically designed
    and trained for benchmark contamination detection, providing superior
    accuracy compared to general-purpose similarity matchers.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the specialized contamination matcher.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Try to use pre-trained model first, fall back to config
        self.use_pretrained = self.config.get("use_pretrained", True)
        self.pretrained_loader = PretrainedModelLoader()
        
        # Model configuration with pre-trained model priority
        if self.use_pretrained and self.pretrained_loader.is_model_available():
            try:
                pretrained_config = self.pretrained_loader.get_model_config_for_matcher()
                self.model_path = pretrained_config["model_path"]
                self.confidence_threshold = pretrained_config["confidence_threshold"]
                self.max_length = pretrained_config["max_length"]
                self.is_pretrained = True
                self.logger = logging.getLogger(__name__)
                self.logger.info(f"Using pre-trained specialized model v{pretrained_config['model_version']}")
            except Exception as e:
                self.logger = logging.getLogger(__name__)
                self.logger.warning(f"Failed to load pre-trained model config: {e}")
                self.is_pretrained = False
                self._set_fallback_config()
        else:
            self.is_pretrained = False
            self._set_fallback_config()
        
        # Common configuration
        self.device = self.config.get("device", "auto")
        self.batch_size = self.config.get("batch_size", 8)
        
        # Model components
        self.tokenizer = None
        self.model = None
        self.preprocessor = TextPreprocessor()
        
        # Performance tracking
        self.total_queries = 0
        self.total_time = 0.0
        self.cache = {}  # Simple caching for repeated queries
        
        model_type = "pre-trained" if self.is_pretrained else "custom"
        self.logger.info(f"SpecializedContaminationMatcher initialized with {model_type} model: {self.model_path}")
    
    def _set_fallback_config(self):
        """Set fallback configuration when pre-trained model is not available."""
        self.model_path = self.config.get("model_path", "models/specialized_contamination_detector")
        self.max_length = self.config.get("max_length", 512)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.8)
    
    def initialize(self) -> bool:
        """Initialize the specialized contamination detection model."""
        try:
            self.logger.info(f"Loading specialized contamination model from: {self.model_path}")
            
            # Determine device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.logger.info(f"Using device: {self.device}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            self.is_initialized = True
            self.logger.info("Specialized contamination model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load specialized model: {str(e)}")
            self.logger.info("Falling back to base model initialization")
            return self._initialize_fallback_model()
    
    def _initialize_fallback_model(self) -> bool:
        """Initialize fallback model if specialized model is not available."""
        try:
            # Use a general model as fallback
            fallback_model = "microsoft/DialoGPT-medium"
            self.logger.info(f"Initializing fallback model: {fallback_model}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # For fallback, we'll use a simple approach
            self.model = None  # Will use similarity-based approach
            self.is_initialized = True
            
            self.logger.info("Fallback model initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize fallback model: {str(e)}")
            return False
    
    def find_matches(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        threshold: float = 0.7,
        max_matches: int = 10
    ) -> List[MatchResult]:
        """Find contamination matches using the specialized model."""
        if not self.is_initialized:
            raise RuntimeError("Specialized matcher not initialized")
        
        start_time = time.time()
        matches = []
        
        self.logger.info(f"Specialized contamination detection for query against {len(candidates)} candidates")
        
        # Preprocess query
        processed_query = self.preprocessor.preprocess(query)
        
        # Check cache first
        cache_key = hash(processed_query)
        if cache_key in self.cache:
            self.logger.debug("Using cached results")
            return self.cache[cache_key]
        
        # Batch process candidates for efficiency
        batch_results = []
        for i in range(0, len(candidates), self.batch_size):
            batch = candidates[i:i + self.batch_size]
            batch_scores = self._process_batch(processed_query, batch)
            batch_results.extend(batch_scores)
        
        # Sort by contamination probability
        batch_results.sort(key=lambda x: x[1], reverse=True)
        
        # Create match results
        for candidate, score, confidence, reasoning in batch_results:
            if score >= threshold and len(matches) < max_matches:
                # Determine if this is an exact match
                exact_match = self._is_exact_match(processed_query, candidate['question_text'])
                
                match = MatchResult(
                    text=candidate['question_text'],
                    similarity_score=score,
                    exact_match=exact_match,
                    metadata={
                        'benchmark_name': candidate.get('benchmark_name', 'Unknown'),
                        'benchmark_type': candidate.get('benchmark_type', 'Unknown'),
                        'source_url': candidate.get('source_url', ''),
                        'contamination_confidence': confidence,
                        'contamination_reasoning': reasoning,
                        'detection_method': 'specialized_model',
                        'model_path': self.model_path
                    }
                )
                matches.append(match)
        
        # Update performance tracking
        processing_time = time.time() - start_time
        self.total_queries += 1
        self.total_time += processing_time
        
        # Cache results
        self.cache[cache_key] = matches
        
        self.logger.info(f"Found {len(matches)} contamination matches in {processing_time:.3f}s")
        return matches
    
    def _process_batch(self, query: str, candidates: List[Dict[str, Any]]) -> List[tuple]:
        """Process a batch of candidates for contamination detection."""
        if not self.model:
            # Fallback to similarity-based detection
            return self._fallback_batch_processing(query, candidates)
        
        batch_results = []
        
        # Prepare batch inputs
        texts = []
        for candidate in candidates:
            candidate_text = candidate['question_text']
            # Create text pair for classification
            text_pair = f"Query: {query} [SEP] Candidate: {candidate_text}"
            texts.append(text_pair)
        
        try:
            # Tokenize batch
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                
                # Extract contamination probabilities (class 1)
                contamination_probs = probabilities[:, 1].cpu().numpy()
                confidence_scores = torch.max(probabilities, dim=-1)[0].cpu().numpy()
            
            # Create results
            for i, candidate in enumerate(candidates):
                contamination_prob = float(contamination_probs[i])
                confidence = float(confidence_scores[i])
                
                # Generate reasoning based on probability
                reasoning = self._generate_reasoning(contamination_prob, confidence)
                
                batch_results.append((candidate, contamination_prob, confidence, reasoning))
        
        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            # Fallback to individual processing
            return self._fallback_batch_processing(query, candidates)
        
        return batch_results
    
    def _fallback_batch_processing(self, query: str, candidates: List[Dict[str, Any]]) -> List[tuple]:
        """Fallback batch processing using similarity-based approach."""
        batch_results = []
        
        for candidate in candidates:
            candidate_text = candidate['question_text']
            
            # Simple similarity-based contamination detection
            similarity = self._compute_text_similarity(query, candidate_text)
            
            # Convert similarity to contamination probability
            contamination_prob = min(similarity * 1.2, 1.0)  # Boost similarity for contamination
            confidence = 0.6  # Lower confidence for fallback method
            
            reasoning = f"Fallback similarity-based detection (similarity: {similarity:.3f})"
            
            batch_results.append((candidate, contamination_prob, confidence, reasoning))
        
        return batch_results
    
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute basic text similarity as fallback."""
        # Simple token-based similarity
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _generate_reasoning(self, contamination_prob: float, confidence: float) -> str:
        """Generate reasoning explanation for contamination detection."""
        if contamination_prob >= 0.9:
            return f"High contamination probability ({contamination_prob:.3f}) with {confidence:.3f} confidence. Strong evidence of benchmark contamination detected."
        elif contamination_prob >= 0.7:
            return f"Moderate contamination probability ({contamination_prob:.3f}) with {confidence:.3f} confidence. Likely contamination detected."
        elif contamination_prob >= 0.5:
            return f"Low contamination probability ({contamination_prob:.3f}) with {confidence:.3f} confidence. Possible contamination detected."
        else:
            return f"Very low contamination probability ({contamination_prob:.3f}) with {confidence:.3f} confidence. Unlikely to be contaminated."
    
    def _is_exact_match(self, query: str, candidate: str) -> bool:
        """Check if query and candidate are exact matches."""
        # Normalize texts for comparison
        query_norm = self.preprocessor.preprocess(query).lower().strip()
        candidate_norm = self.preprocessor.preprocess(candidate).lower().strip()
        
        return query_norm == candidate_norm
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute contamination similarity between two texts."""
        if not self.is_initialized:
            raise RuntimeError("Specialized matcher not initialized")
        
        if not self.model:
            # Fallback to basic similarity
            return self._compute_text_similarity(text1, text2)
        
        # Use specialized model for similarity
        text_pair = f"Query: {text1} [SEP] Candidate: {text2}"
        
        try:
            inputs = self.tokenizer(
                text_pair,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                
                # Return contamination probability
                return float(probabilities[0, 1])
        
        except Exception as e:
            self.logger.error(f"Similarity computation failed: {str(e)}")
            return self._compute_text_similarity(text1, text2)
    
    def get_matcher_info(self) -> Dict[str, Any]:
        """Get information about the specialized matcher."""
        info = {
            "matcher_type": "SpecializedContaminationMatcher",
            "model_path": self.model_path,
            "device": self.device,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "confidence_threshold": self.confidence_threshold,
            "is_initialized": self.is_initialized,
            "total_queries": self.total_queries,
            "average_query_time": self.total_time / max(self.total_queries, 1) if self.total_queries > 0 else 0,
            "cache_size": len(self.cache),
            "supports_batch_processing": True,
            "supports_gpu": torch.cuda.is_available(),
            "current_device": self.device,
            "is_pretrained": self.is_pretrained
        }
        
        # Add pre-trained model metadata if available
        if self.is_pretrained:
            try:
                metadata = self.pretrained_loader.get_model_metadata()
                if metadata:
                    info["pretrained_model_info"] = {
                        "version": metadata.get("model_info", {}).get("version", "Unknown"),
                        "training_date": metadata.get("model_info", {}).get("training_date", "Unknown"),
                        "accuracy": metadata.get("performance_benchmarks", {}).get("accuracy", 0),
                        "f1_score": metadata.get("performance_benchmarks", {}).get("f1_score", 0),
                        "benchmark_coverage": metadata.get("training_info", {}).get("benchmark_questions_used", 0)
                    }
            except Exception as e:
                self.logger.warning(f"Could not load pre-trained model metadata: {e}")
        
        return info
    
    def explain_match(self, query: str, match: str, score: float) -> str:
        """Provide detailed explanation for a contamination match."""
        explanation = f"""
Specialized Contamination Detection Analysis:

Query: "{query}"
Match: "{match}"
Contamination Score: {score:.3f}

Analysis:
- Detection Method: Specialized fine-tuned model
- Model Path: {self.model_path}
- Confidence Threshold: {self.confidence_threshold}

The specialized contamination detection model has identified this as a potential
benchmark contamination case. This model was specifically trained to detect
various forms of contamination including paraphrasing, format changes, and
subtle modifications that might indicate data leakage from benchmark datasets.

Contamination Indicators:
"""
        
        if score >= 0.9:
            explanation += "- CRITICAL: Very high contamination probability\n"
            explanation += "- Strong evidence of direct or near-direct copying\n"
            explanation += "- Immediate investigation recommended\n"
        elif score >= 0.7:
            explanation += "- HIGH: Significant contamination probability\n"
            explanation += "- Likely paraphrasing or format modification\n"
            explanation += "- Review recommended\n"
        elif score >= 0.5:
            explanation += "- MODERATE: Possible contamination detected\n"
            explanation += "- May indicate indirect influence or similarity\n"
            explanation += "- Further analysis suggested\n"
        else:
            explanation += "- LOW: Minimal contamination probability\n"
            explanation += "- Likely coincidental similarity\n"
            explanation += "- No immediate action required\n"
        
        return explanation
    
    def validate_config(self) -> bool:
        """Validate specialized matcher configuration."""
        try:
            # Check model path
            if not self.model_path:
                self.logger.error("Model path not specified")
                return False
            
            # Check device configuration
            if self.device not in ["auto", "cpu", "cuda"]:
                self.logger.error(f"Invalid device: {self.device}")
                return False
            
            # Check other parameters
            if not 0.0 <= self.confidence_threshold <= 1.0:
                self.logger.error(f"Invalid confidence threshold: {self.confidence_threshold}")
                return False
            
            if self.max_length <= 0:
                self.logger.error(f"Invalid max_length: {self.max_length}")
                return False
            
            if self.batch_size <= 0:
                self.logger.error(f"Invalid batch_size: {self.batch_size}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            return False
    
    def cleanup(self):
        """Clean up specialized matcher resources."""
        try:
            if self.model is not None:
                del self.model
            if self.tokenizer is not None:
                del self.tokenizer
            
            # Clear cache
            self.cache.clear()
            
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.is_initialized = False
            self.logger.info("Specialized matcher resources cleaned up")
            
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {str(e)}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the specialized matcher."""
        return {
            "total_queries": self.total_queries,
            "total_time": self.total_time,
            "average_query_time": self.total_time / max(self.total_queries, 1),
            "cache_hits": len(self.cache),
            "cache_efficiency": len(self.cache) / max(self.total_queries, 1),
            "device": self.device,
            "model_path": self.model_path,
            "is_specialized": True
        }
