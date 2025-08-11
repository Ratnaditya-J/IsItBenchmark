"""
Contamination Data Generator for IsItBenchmark.

This module generates synthetic training data for contamination detection models,
implementing various contamination patterns and techniques found in research
literature to create comprehensive training datasets.
"""

import json
import logging
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass

from ..benchmarks.database import BenchmarkDatabase


@dataclass
class ContaminationPattern:
    """Represents a contamination pattern with metadata."""
    name: str
    description: str
    severity: float  # 0.0 to 1.0, higher means more obvious contamination
    category: str  # 'paraphrase', 'format', 'context', 'noise', 'translation'


class ContaminationDataGenerator:
    """
    Advanced contamination data generator for training specialized models.
    
    Implements research-backed contamination patterns and techniques to create
    realistic training data that covers various forms of benchmark contamination
    found in real-world scenarios.
    """
    
    def __init__(self, database: Optional[BenchmarkDatabase] = None):
        """
        Initialize the contamination data generator.
        
        Args:
            database: BenchmarkDatabase instance for accessing benchmark questions
        """
        self.database = database or BenchmarkDatabase()
        self.logger = logging.getLogger(__name__)
        
        # Initialize contamination patterns
        self.contamination_patterns = self._initialize_patterns()
        
        # Load templates and resources
        self.paraphrase_templates = self._load_paraphrase_templates()
        self.context_templates = self._load_context_templates()
        self.format_templates = self._load_format_templates()
        self.noise_patterns = self._load_noise_patterns()
        
        # Statistics tracking
        self.generation_stats = {
            'total_generated': 0,
            'positive_samples': 0,
            'negative_samples': 0,
            'pattern_usage': {}
        }
        
        self.logger.info("ContaminationDataGenerator initialized")
    
    def _initialize_patterns(self) -> List[ContaminationPattern]:
        """Initialize contamination patterns based on research literature."""
        return [
            # Paraphrasing patterns
            ContaminationPattern(
                name="direct_paraphrase",
                description="Direct paraphrasing with synonym replacement",
                severity=0.8,
                category="paraphrase"
            ),
            ContaminationPattern(
                name="structural_paraphrase",
                description="Structural changes while preserving meaning",
                severity=0.7,
                category="paraphrase"
            ),
            ContaminationPattern(
                name="question_reformulation",
                description="Reformulating questions with different phrasing",
                severity=0.6,
                category="paraphrase"
            ),
            
            # Format patterns
            ContaminationPattern(
                name="format_change",
                description="Changing question format (Q:, Question:, etc.)",
                severity=0.4,
                category="format"
            ),
            ContaminationPattern(
                name="multiple_choice_removal",
                description="Removing multiple choice options",
                severity=0.5,
                category="format"
            ),
            ContaminationPattern(
                name="prompt_wrapping",
                description="Wrapping in different prompt formats",
                severity=0.3,
                category="format"
            ),
            
            # Context patterns
            ContaminationPattern(
                name="context_addition",
                description="Adding contextual information or backstory",
                severity=0.6,
                category="context"
            ),
            ContaminationPattern(
                name="domain_shift",
                description="Shifting to different domain while preserving core",
                severity=0.7,
                category="context"
            ),
            ContaminationPattern(
                name="persona_injection",
                description="Adding persona or role-playing context",
                severity=0.5,
                category="context"
            ),
            
            # Noise patterns
            ContaminationPattern(
                name="typo_injection",
                description="Adding realistic typos and spelling errors",
                severity=0.2,
                category="noise"
            ),
            ContaminationPattern(
                name="punctuation_changes",
                description="Modifying punctuation patterns",
                severity=0.1,
                category="noise"
            ),
            ContaminationPattern(
                name="case_changes",
                description="Changing capitalization patterns",
                severity=0.1,
                category="noise"
            ),
            
            # Translation patterns
            ContaminationPattern(
                name="translation_artifacts",
                description="Simulating translation back-and-forth artifacts",
                severity=0.4,
                category="translation"
            ),
            ContaminationPattern(
                name="grammar_shifts",
                description="Subtle grammar pattern changes",
                severity=0.3,
                category="translation"
            )
        ]
    
    def _load_paraphrase_templates(self) -> List[Dict[str, Any]]:
        """Load paraphrasing templates and patterns."""
        return [
            {
                "pattern": r"What is (.+)\?",
                "replacements": [
                    "Can you tell me what {} is?",
                    "I need to know what {} is.",
                    "Please explain what {} is.",
                    "Could you define {}?",
                    "What would you say {} is?"
                ]
            },
            {
                "pattern": r"How (.+)\?",
                "replacements": [
                    "In what way {}?",
                    "What is the method to {}?",
                    "Can you explain how {}?",
                    "What's the process to {}?",
                    "By what means {}?"
                ]
            },
            {
                "pattern": r"Why (.+)\?",
                "replacements": [
                    "What is the reason {}?",
                    "For what purpose {}?",
                    "What causes {}?",
                    "What's the explanation for {}?",
                    "What makes {}?"
                ]
            },
            {
                "pattern": r"Which (.+)\?",
                "replacements": [
                    "What {} would you choose?",
                    "Among {}, which one?",
                    "What is the correct {}?",
                    "Can you identify the {}?",
                    "What {} is it?"
                ]
            }
        ]
    
    def _load_context_templates(self) -> List[str]:
        """Load context addition templates."""
        return [
            "For my research project, {}",
            "I'm studying this topic: {}",
            "Can you help me understand: {}",
            "In the context of machine learning, {}",
            "From an academic perspective, {}",
            "As part of my homework, {}",
            "I'm curious about: {}",
            "For educational purposes, {}",
            "In my analysis, I need to know: {}",
            "While working on this problem: {}",
            "During my investigation: {}",
            "As I'm learning about this field: {}",
            "In preparation for my exam: {}",
            "For my thesis research: {}",
            "While reviewing the literature: {}"
        ]
    
    def _load_format_templates(self) -> List[Dict[str, str]]:
        """Load format change templates."""
        return [
            {"prefix": "Question: ", "suffix": ""},
            {"prefix": "Q: ", "suffix": ""},
            {"prefix": "Query: ", "suffix": ""},
            {"prefix": "Problem: ", "suffix": ""},
            {"prefix": "Task: ", "suffix": ""},
            {"prefix": "", "suffix": " (Please answer)"},
            {"prefix": "", "suffix": " - Explain."},
            {"prefix": "Consider: ", "suffix": ""},
            {"prefix": "Analyze: ", "suffix": ""},
            {"prefix": "Evaluate: ", "suffix": ""}
        ]
    
    def _load_noise_patterns(self) -> Dict[str, List[str]]:
        """Load noise injection patterns."""
        return {
            "common_typos": [
                ("the", "teh"), ("and", "adn"), ("you", "yuo"),
                ("that", "taht"), ("with", "wiht"), ("have", "ahve"),
                ("this", "tihs"), ("will", "wil"), ("from", "form"),
                ("they", "tehy"), ("know", "konw"), ("want", "watn")
            ],
            "punctuation_changes": [
                (".", " ."), (",", " ,"), ("?", " ?"), ("!", " !"),
                (":", " :"), (";", " ;"), ("'", " '"), ('"', ' "')
            ],
            "spacing_changes": [
                ("  ", " "), (" ", "  "), ("\n", " "), ("\t", " ")
            ]
        }
    
    def generate_training_data(
        self,
        num_samples: int = 10000,
        positive_ratio: float = 0.5,
        pattern_distribution: Optional[Dict[str, float]] = None
    ) -> Tuple[List[str], List[int], Dict[str, Any]]:
        """
        Generate comprehensive training data for contamination detection.
        
        Args:
            num_samples: Total number of samples to generate
            positive_ratio: Ratio of positive (contaminated) samples
            pattern_distribution: Custom distribution of contamination patterns
            
        Returns:
            Tuple of (texts, labels, metadata)
        """
        self.logger.info(f"Generating {num_samples} training samples")
        
        # Get benchmark questions
        benchmark_questions = self._get_benchmark_questions()
        if len(benchmark_questions) < 100:
            self.logger.warning(f"Limited benchmark questions available: {len(benchmark_questions)}")
        
        # Calculate sample counts
        num_positive = int(num_samples * positive_ratio)
        num_negative = num_samples - num_positive
        
        texts = []
        labels = []
        metadata = []
        
        # Generate positive samples (contaminated)
        self.logger.info(f"Generating {num_positive} positive samples")
        for i in range(num_positive):
            if i < len(benchmark_questions):
                original = benchmark_questions[i]
            else:
                # Reuse questions with different patterns
                original = benchmark_questions[i % len(benchmark_questions)]
            
            # Apply contamination pattern
            contaminated, pattern_info = self._apply_contamination_pattern(
                original, pattern_distribution
            )
            
            # Create text pair
            text_pair = f"Query: {contaminated} [SEP] Candidate: {original}"
            texts.append(text_pair)
            labels.append(1)  # Contaminated
            
            metadata.append({
                "sample_type": "positive",
                "original_text": original,
                "contaminated_text": contaminated,
                "pattern": pattern_info["pattern"],
                "severity": pattern_info["severity"],
                "category": pattern_info["category"]
            })
        
        # Generate negative samples (clean)
        self.logger.info(f"Generating {num_negative} negative samples")
        for i in range(num_negative):
            # Create unrelated text pairs
            idx1 = random.randint(0, len(benchmark_questions) - 1)
            idx2 = random.randint(0, len(benchmark_questions) - 1)
            
            # Ensure they're different
            while idx1 == idx2 and len(benchmark_questions) > 1:
                idx2 = random.randint(0, len(benchmark_questions) - 1)
            
            text1 = benchmark_questions[idx1]
            text2 = self._generate_unrelated_text(benchmark_questions[idx2])
            
            text_pair = f"Query: {text2} [SEP] Candidate: {text1}"
            texts.append(text_pair)
            labels.append(0)  # Clean
            
            metadata.append({
                "sample_type": "negative",
                "query_text": text2,
                "candidate_text": text1,
                "pattern": "unrelated",
                "severity": 0.0,
                "category": "clean"
            })
        
        # Update statistics
        self.generation_stats['total_generated'] += num_samples
        self.generation_stats['positive_samples'] += num_positive
        self.generation_stats['negative_samples'] += num_negative
        
        # Create summary metadata
        summary_metadata = {
            "total_samples": num_samples,
            "positive_samples": num_positive,
            "negative_samples": num_negative,
            "positive_ratio": positive_ratio,
            "benchmark_questions_used": len(benchmark_questions),
            "pattern_distribution": self._calculate_pattern_distribution(metadata),
            "generation_stats": self.generation_stats.copy()
        }
        
        self.logger.info(f"Generated {len(texts)} training samples successfully")
        return texts, labels, summary_metadata
    
    def _get_benchmark_questions(self) -> List[str]:
        """Get benchmark questions from the database."""
        try:
            conn = self.database._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT question_text FROM benchmark_questions LIMIT 50000")
            questions = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            
            # Filter out very short or very long questions
            filtered_questions = [
                q for q in questions 
                if 10 <= len(q) <= 500 and q.strip()
            ]
            
            return filtered_questions
            
        except Exception as e:
            self.logger.error(f"Failed to get benchmark questions: {str(e)}")
            return self._get_fallback_questions()
    
    def _get_fallback_questions(self) -> List[str]:
        """Get fallback questions if database is not available."""
        return [
            "What is machine learning?",
            "How does neural network training work?",
            "What are the main types of supervised learning?",
            "Explain the concept of overfitting in machine learning.",
            "What is the difference between classification and regression?",
            "How do you evaluate a machine learning model?",
            "What is cross-validation?",
            "Explain the bias-variance tradeoff.",
            "What are ensemble methods in machine learning?",
            "How does gradient descent work?",
            "What is the purpose of regularization?",
            "Explain the concept of feature engineering.",
            "What are the advantages of deep learning?",
            "How do convolutional neural networks work?",
            "What is transfer learning?",
            "Explain the concept of attention mechanisms.",
            "What are generative adversarial networks?",
            "How do you handle imbalanced datasets?",
            "What is the curse of dimensionality?",
            "Explain the concept of reinforcement learning."
        ]
    
    def _apply_contamination_pattern(
        self,
        original: str,
        pattern_distribution: Optional[Dict[str, float]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Apply a contamination pattern to create a contaminated version."""
        # Select pattern based on distribution
        if pattern_distribution:
            pattern = np.random.choice(
                list(pattern_distribution.keys()),
                p=list(pattern_distribution.values())
            )
            # Find pattern object with error handling
            pattern_obj = None
            for p in self.contamination_patterns:
                if p.name == pattern:
                    pattern_obj = p
                    break
            
            if pattern_obj is None:
                self.logger.warning(f"Pattern '{pattern}' not found in contamination_patterns. Available patterns: {[p.name for p in self.contamination_patterns]}")
                pattern_obj = random.choice(self.contamination_patterns)
        else:
            pattern_obj = random.choice(self.contamination_patterns)
        
        # Update pattern usage statistics
        if pattern_obj.name not in self.generation_stats['pattern_usage']:
            self.generation_stats['pattern_usage'][pattern_obj.name] = 0
        self.generation_stats['pattern_usage'][pattern_obj.name] += 1
        
        # Apply the pattern
        contaminated = self._apply_specific_pattern(original, pattern_obj)
        
        pattern_info = {
            "pattern": pattern_obj.name,
            "description": pattern_obj.description,
            "severity": pattern_obj.severity,
            "category": pattern_obj.category
        }
        
        return contaminated, pattern_info
    
    def _apply_specific_pattern(self, text: str, pattern: ContaminationPattern) -> str:
        """Apply a specific contamination pattern to text."""
        if pattern.category == "paraphrase":
            return self._apply_paraphrase_pattern(text, pattern)
        elif pattern.category == "format":
            return self._apply_format_pattern(text, pattern)
        elif pattern.category == "context":
            return self._apply_context_pattern(text, pattern)
        elif pattern.category == "noise":
            return self._apply_noise_pattern(text, pattern)
        elif pattern.category == "translation":
            return self._apply_translation_pattern(text, pattern)
        else:
            return text
    
    def _apply_paraphrase_pattern(self, text: str, pattern: ContaminationPattern) -> str:
        """Apply paraphrasing patterns."""
        if pattern.name == "direct_paraphrase":
            # Apply template-based paraphrasing
            for template in self.paraphrase_templates:
                match = re.search(template["pattern"], text, re.IGNORECASE)
                if match:
                    replacement = random.choice(template["replacements"])
                    return replacement.format(match.group(1))
            
            # Fallback: simple word replacements
            replacements = {
                "what is": "can you tell me what",
                "how": "in what way",
                "why": "what is the reason",
                "which": "what",
                "where": "in what location",
                "when": "at what time"
            }
            
            result = text.lower()
            for old, new in replacements.items():
                result = result.replace(old, new)
            return result.capitalize()
        
        elif pattern.name == "structural_paraphrase":
            # Change sentence structure
            if text.endswith("?"):
                return f"Please explain: {text[:-1]}."
            else:
                return f"Can you tell me about {text.lower()}?"
        
        elif pattern.name == "question_reformulation":
            # Reformulate as different question type
            if "what" in text.lower():
                return text.replace("What", "Which").replace("what", "which")
            elif "how" in text.lower():
                return text.replace("How", "What is the method to").replace("how", "what is the method to")
            else:
                return f"I need to understand: {text}"
        
        return text
    
    def _apply_format_pattern(self, text: str, pattern: ContaminationPattern) -> str:
        """Apply format change patterns."""
        template = random.choice(self.format_templates)
        
        # Remove existing prefixes if any
        clean_text = text
        for prefix in ["Question:", "Q:", "Query:", "Problem:", "Task:"]:
            if clean_text.startswith(prefix):
                clean_text = clean_text[len(prefix):].strip()
        
        return f"{template['prefix']}{clean_text}{template['suffix']}"
    
    def _apply_context_pattern(self, text: str, pattern: ContaminationPattern) -> str:
        """Apply context addition patterns."""
        template = random.choice(self.context_templates)
        return template.format(text)
    
    def _apply_noise_pattern(self, text: str, pattern: ContaminationPattern) -> str:
        """Apply noise injection patterns."""
        result = text
        
        if pattern.name == "typo_injection":
            # Apply random typos
            for original, typo in self.noise_patterns["common_typos"]:
                if original in result.lower() and random.random() < 0.3:
                    result = result.replace(original, typo)
        
        elif pattern.name == "punctuation_changes":
            # Modify punctuation
            for original, modified in self.noise_patterns["punctuation_changes"]:
                if original in result and random.random() < 0.5:
                    result = result.replace(original, modified)
        
        elif pattern.name == "case_changes":
            # Random case changes
            if random.random() < 0.5:
                result = result.lower()
            elif random.random() < 0.3:
                result = result.upper()
        
        return result
    
    def _apply_translation_pattern(self, text: str, pattern: ContaminationPattern) -> str:
        """Apply translation artifact patterns."""
        # Simulate translation artifacts
        translation_changes = {
            "the": "a",
            "is": "was",
            "are": "were",
            "will": "would",
            "can": "could",
            "may": "might",
            "must": "should"
        }
        
        result = text
        for original, translated in translation_changes.items():
            if original in result.lower() and random.random() < 0.4:
                result = result.replace(original, translated)
        
        return result
    
    def _generate_unrelated_text(self, base_text: str) -> str:
        """Generate unrelated text for negative samples."""
        unrelated_topics = [
            "cooking and recipes",
            "travel destinations",
            "weather patterns",
            "sports and games",
            "music and entertainment",
            "health and fitness",
            "technology news",
            "historical events",
            "art and culture",
            "business and finance"
        ]
        
        topic = random.choice(unrelated_topics)
        
        templates = [
            f"What's the best way to learn about {topic}?",
            f"Can you recommend resources for {topic}?",
            f"I'm interested in {topic}, where should I start?",
            f"What are the latest trends in {topic}?",
            f"How has {topic} evolved over time?",
            f"What are the key concepts in {topic}?",
            f"Who are the experts in {topic}?",
            f"What tools are used for {topic}?",
            f"What are common misconceptions about {topic}?",
            f"How do beginners approach {topic}?"
        ]
        
        return random.choice(templates)
    
    def _calculate_pattern_distribution(self, metadata: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate the actual distribution of patterns used."""
        pattern_counts = {}
        total_positive = sum(1 for m in metadata if m["sample_type"] == "positive")
        
        for meta in metadata:
            if meta["sample_type"] == "positive":
                pattern = meta["pattern"]
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Convert to proportions
        return {
            pattern: count / total_positive
            for pattern, count in pattern_counts.items()
        }
    
    def save_training_data(
        self,
        texts: List[str],
        labels: List[int],
        metadata: Dict[str, Any],
        output_path: str
    ):
        """Save generated training data to files."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main training data
        training_data = {
            "texts": texts,
            "labels": labels,
            "metadata": metadata
        }
        
        with open(output_dir / "training_data.json", "w") as f:
            json.dump(training_data, f, indent=2)
        
        # Save separate files for easier loading
        with open(output_dir / "texts.json", "w") as f:
            json.dump(texts, f, indent=2)
        
        with open(output_dir / "labels.json", "w") as f:
            json.dump(labels, f, indent=2)
        
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Training data saved to: {output_dir}")
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about data generation."""
        return self.generation_stats.copy()
