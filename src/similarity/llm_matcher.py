"""
LLM-based similarity matcher for IsItBenchmark.

This module implements an LLM-based matcher that uses large language models
to detect semantic similarity and benchmark contamination with high accuracy.
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from .base_matcher import BaseMatcher, MatcherType
from ..detection.models import MatchResult, BenchmarkType
from ..utils.preprocessing import TextPreprocessor


class LLMMatcher(BaseMatcher):
    """
    LLM-based similarity matcher using HuggingFace transformers.
    
    Uses large language models to perform sophisticated similarity detection
    with reasoning capabilities and detailed explanations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM matcher.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Default configuration
        self.model_name = self.config.get("model_name", "microsoft/DialoGPT-medium")
        self.device = self.config.get("device", "auto")
        self.max_length = self.config.get("max_length", 512)
        self.temperature = self.config.get("temperature", 0.1)
        self.batch_size = self.config.get("batch_size", 1)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.preprocessor = TextPreprocessor()
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.total_queries = 0
        self.total_time = 0.0
    
    def initialize(self) -> bool:
        """Initialize the LLM model and tokenizer."""
        try:
            self.logger.info(f"Initializing LLM matcher with model: {self.model_name}")
            
            # Determine device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.logger.info(f"Using device: {self.device}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Add pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device if self.device != "cpu" else None,
            )
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            
            self.is_initialized = True
            self.logger.info("LLM matcher initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM matcher: {str(e)}")
            return False
    
    def find_matches(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]], 
        threshold: float = 0.7,
        max_matches: int = 10
    ) -> List[MatchResult]:
        """Find matches using LLM-based similarity detection."""
        if not self.is_initialized:
            raise RuntimeError("LLM matcher not initialized")
        
        start_time = time.time()
        matches = []
        
        self.logger.info(f"LLM matching query against {len(candidates)} candidates")
        
        for candidate in candidates:
            try:
                # Extract candidate information
                candidate_text = candidate.get("question_text", "")
                benchmark_name = candidate.get("benchmark_name", "Unknown")
                benchmark_type = candidate.get("benchmark_type", "multiple_choice")
                
                # Compute similarity using LLM
                similarity_result = self._compute_llm_similarity(query, candidate_text)
                
                if similarity_result["score"] >= threshold:
                    # Create match result
                    match = MatchResult(
                        benchmark_name=benchmark_name,
                        benchmark_type=BenchmarkType(benchmark_type) if isinstance(benchmark_type, str) else benchmark_type,
                        similarity_score=similarity_result["score"],
                        matched_text=candidate_text,
                        exact_match=similarity_result["score"] > 0.95,
                        source_url=candidate.get("source_url"),
                        publication_date=candidate.get("publication_date"),
                        dataset_version=candidate.get("dataset_version"),
                        explanation=similarity_result["explanation"],
                        metadata={
                            "llm_reasoning": similarity_result["reasoning"],
                            "confidence": similarity_result["confidence"],
                            "analysis_time_ms": similarity_result["analysis_time_ms"],
                        }
                    )
                    matches.append(match)
                
            except Exception as e:
                self.logger.warning(f"Error processing candidate: {str(e)}")
                continue
        
        # Sort by similarity score (descending)
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Limit results
        matches = matches[:max_matches]
        
        # Update performance tracking
        elapsed_time = time.time() - start_time
        self.total_queries += 1
        self.total_time += elapsed_time
        
        self.logger.info(f"LLM matching completed: {len(matches)} matches found in {elapsed_time:.2f}s")
        
        return matches
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity score between two texts using LLM."""
        if not self.is_initialized:
            raise RuntimeError("LLM matcher not initialized")
        
        result = self._compute_llm_similarity(text1, text2)
        return result["score"]
    
    def _compute_llm_similarity(self, query: str, candidate: str) -> Dict[str, Any]:
        """
        Use LLM to compute similarity and provide reasoning.
        
        Args:
            query: Query text
            candidate: Candidate text
            
        Returns:
            Dictionary with score, reasoning, and metadata
        """
        start_time = time.time()
        
        # Create prompt for similarity analysis
        prompt = self._create_similarity_prompt(query, candidate)
        
        try:
            # Generate response using LLM
            response = self.pipeline(
                prompt,
                max_length=len(prompt.split()) + 150,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False,
            )
            
            # Extract and parse the response
            generated_text = response[0]["generated_text"].strip()
            result = self._parse_llm_response(generated_text)
            
            # Add timing information
            result["analysis_time_ms"] = (time.time() - start_time) * 1000
            
            return result
            
        except Exception as e:
            self.logger.error(f"LLM similarity computation failed: {str(e)}")
            return {
                "score": 0.0,
                "reasoning": f"Analysis failed: {str(e)}",
                "confidence": "low",
                "explanation": "LLM analysis encountered an error",
                "analysis_time_ms": (time.time() - start_time) * 1000,
            }
    
    def _create_similarity_prompt(self, query: str, candidate: str) -> str:
        """Create a prompt for LLM similarity analysis."""
        prompt = f"""Analyze if these two texts are semantically similar or if one could be a paraphrased version of the other. Consider benchmark contamination detection context.

Text 1: "{query}"
Text 2: "{candidate}"

Provide your analysis in this exact format:
SCORE: [0.0-1.0]
CONFIDENCE: [high/medium/low]
REASONING: [brief explanation]
EXPLANATION: [detailed analysis]

Analysis:"""
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract structured information."""
        try:
            lines = response.strip().split('\n')
            result = {
                "score": 0.0,
                "confidence": "low",
                "reasoning": "No analysis provided",
                "explanation": "Failed to parse LLM response"
            }
            
            for line in lines:
                line = line.strip()
                if line.startswith("SCORE:"):
                    try:
                        score_str = line.replace("SCORE:", "").strip()
                        result["score"] = max(0.0, min(1.0, float(score_str)))
                    except ValueError:
                        pass
                elif line.startswith("CONFIDENCE:"):
                    confidence = line.replace("CONFIDENCE:", "").strip().lower()
                    if confidence in ["high", "medium", "low"]:
                        result["confidence"] = confidence
                elif line.startswith("REASONING:"):
                    result["reasoning"] = line.replace("REASONING:", "").strip()
                elif line.startswith("EXPLANATION:"):
                    result["explanation"] = line.replace("EXPLANATION:", "").strip()
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM response: {str(e)}")
            return {
                "score": 0.0,
                "confidence": "low",
                "reasoning": "Response parsing failed",
                "explanation": f"Error parsing LLM output: {str(e)}"
            }
    
    def get_matcher_info(self) -> Dict[str, Any]:
        """Get information about the LLM matcher."""
        return {
            "type": MatcherType.LLM.value,
            "model_name": self.model_name,
            "device": self.device,
            "is_initialized": self.is_initialized,
            "total_queries": self.total_queries,
            "average_time_per_query": self.total_time / max(1, self.total_queries),
            "capabilities": [
                "semantic_similarity",
                "reasoning_explanation",
                "confidence_scoring",
                "paraphrase_detection"
            ]
        }
    
    def explain_match(self, query: str, match: str, score: float) -> Dict[str, Any]:
        """Provide detailed explanation for a match using LLM reasoning."""
        if not self.is_initialized:
            return super().explain_match(query, match, score)
        
        # Get detailed analysis from LLM
        result = self._compute_llm_similarity(query, match)
        
        return {
            "query": query,
            "match": match,
            "score": score,
            "confidence": result.get("confidence", "unknown"),
            "reasoning": result.get("reasoning", "No reasoning provided"),
            "explanation": result.get("explanation", "No explanation provided"),
            "matcher_type": "llm",
            "model_name": self.model_name,
        }
    
    def validate_config(self) -> bool:
        """Validate LLM matcher configuration."""
        try:
            # Check if model name is valid
            if not self.model_name:
                self.logger.error("Model name not specified")
                return False
            
            # Check device configuration
            if self.device not in ["auto", "cpu", "cuda"]:
                self.logger.error(f"Invalid device: {self.device}")
                return False
            
            # Check other parameters
            if not 0.0 <= self.temperature <= 2.0:
                self.logger.error(f"Invalid temperature: {self.temperature}")
                return False
            
            if self.max_length <= 0:
                self.logger.error(f"Invalid max_length: {self.max_length}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            return False
    
    def cleanup(self):
        """Clean up LLM resources."""
        try:
            if self.model is not None:
                del self.model
            if self.tokenizer is not None:
                del self.tokenizer
            if self.pipeline is not None:
                del self.pipeline
            
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.is_initialized = False
            self.logger.info("LLM matcher resources cleaned up")
            
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {str(e)}")


class HuggingFaceLLMMatcher(LLMMatcher):
    """
    Specialized LLM matcher for HuggingFace models.
    
    Optimized for specific HuggingFace model architectures and provides
    enhanced prompt engineering for benchmark detection tasks.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize HuggingFace LLM matcher."""
        super().__init__(config)
        
        # Use a more suitable model for text analysis
        # Note: The specified model might not exist, using a reliable alternative
        self.model_name = self.config.get("model_name", "microsoft/DialoGPT-medium")
        
        # Enhanced prompt templates
        self.prompt_templates = {
            "similarity": self._get_similarity_template(),
            "contamination": self._get_contamination_template(),
        }
    
    def _get_similarity_template(self) -> str:
        """Get enhanced prompt template for similarity analysis."""
        return """You are an expert at detecting semantic similarity and benchmark contamination in AI/ML datasets.

Task: Determine if Text 2 could be from the same source as Text 1, or if they are semantically equivalent.

Text 1: "{query}"
Text 2: "{candidate}"

Consider:
- Exact matches (score: 1.0)
- Paraphrased versions (score: 0.7-0.9)
- Semantically similar but different questions (score: 0.4-0.6)
- Unrelated content (score: 0.0-0.3)

Respond in this format:
SCORE: [0.0-1.0]
CONFIDENCE: [high/medium/low]
REASONING: [why you assigned this score]
EXPLANATION: [detailed analysis of similarities/differences]

Analysis:"""
    
    def _get_contamination_template(self) -> str:
        """Get enhanced prompt template for contamination detection."""
        return """You are a benchmark contamination detector for AI/ML evaluation integrity.

Query: "{query}"
Benchmark Question: "{candidate}"

Assess if the query could be derived from or contaminated by the benchmark question.

Consider:
- Direct copies or minimal modifications
- Paraphrasing that preserves the core question
- Translation or format changes
- Domain-specific variations

SCORE: [0.0-1.0 contamination probability]
CONFIDENCE: [high/medium/low]
REASONING: [evidence for contamination]
EXPLANATION: [detailed contamination analysis]

Analysis:"""
