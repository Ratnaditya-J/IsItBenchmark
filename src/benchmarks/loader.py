"""
Benchmark data loader for IsItBenchmark.

This module handles loading benchmark datasets from various sources
and formats into the database.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..detection.models import BenchmarkInfo, BenchmarkType
from .database import BenchmarkDatabase


class BenchmarkLoader:
    """
    Loader for benchmark datasets.
    
    Handles loading benchmark data from various sources and formats,
    including JSON files, CSV files, and online datasets.
    """
    
    def __init__(self, database: BenchmarkDatabase):
        """
        Initialize the benchmark loader.
        
        Args:
            database: BenchmarkDatabase instance
        """
        self.database = database
        self.logger = logging.getLogger(__name__)
    
    def load_sample_data(self):
        """Load sample benchmark data for demonstration."""
        self.logger.info("Loading sample benchmark data...")
        
        # Sample MMLU-style questions
        mmlu_info = BenchmarkInfo(
            name="MMLU-Sample",
            type=BenchmarkType.MULTIPLE_CHOICE,
            description="Sample questions from Massive Multitask Language Understanding",
            source_url="https://github.com/hendrycks/test",
            publication_date="2020-09-07",
            num_examples=100,
            languages=["en"],
            domains=["general", "academic"],
            license="MIT",
            citation="Hendrycks et al. (2020)"
        )
        
        mmlu_questions = [
            {
                "question_text": "What is the capital of France?",
                "answer": "Paris",
                "choices": ["London", "Berlin", "Paris", "Madrid"],
                "category": "geography",
                "difficulty": "easy",
                "metadata": {"subject": "world_geography", "level": "high_school"}
            },
            {
                "question_text": "Which of the following is a programming language?",
                "answer": "Python",
                "choices": ["HTML", "CSS", "Python", "JSON"],
                "category": "computer_science",
                "difficulty": "easy",
                "metadata": {"subject": "computer_science", "level": "high_school"}
            },
            {
                "question_text": "What is the square root of 16?",
                "answer": "4",
                "choices": ["2", "4", "8", "16"],
                "category": "mathematics",
                "difficulty": "easy",
                "metadata": {"subject": "elementary_mathematics", "level": "elementary"}
            }
        ]
        
        # Sample GSM8K-style questions
        gsm8k_info = BenchmarkInfo(
            name="GSM8K-Sample",
            type=BenchmarkType.MATH,
            description="Sample grade school math word problems",
            source_url="https://github.com/openai/grade-school-math",
            publication_date="2021-10-27",
            num_examples=50,
            languages=["en"],
            domains=["mathematics", "reasoning"],
            license="MIT",
            citation="Cobbe et al. (2021)"
        )
        
        gsm8k_questions = [
            {
                "question_text": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes 4 into muffins for her friends every day. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                "answer": "18",
                "choices": [],
                "category": "word_problems",
                "difficulty": "medium",
                "metadata": {"grade_level": "middle_school", "topic": "arithmetic"}
            },
            {
                "question_text": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
                "answer": "3",
                "choices": [],
                "category": "word_problems",
                "difficulty": "easy",
                "metadata": {"grade_level": "elementary", "topic": "fractions"}
            }
        ]
        
        # Sample HumanEval-style questions
        humaneval_info = BenchmarkInfo(
            name="HumanEval-Sample",
            type=BenchmarkType.CODE,
            description="Sample Python programming problems",
            source_url="https://github.com/openai/human-eval",
            publication_date="2021-07-07",
            num_examples=25,
            languages=["en"],
            domains=["programming", "python"],
            license="MIT",
            citation="Chen et al. (2021)"
        )
        
        humaneval_questions = [
            {
                "question_text": "Write a function that takes a list of numbers and returns the sum of all even numbers.",
                "answer": "def sum_even(numbers):\n    return sum(x for x in numbers if x % 2 == 0)",
                "choices": [],
                "category": "programming",
                "difficulty": "easy",
                "metadata": {"language": "python", "topic": "list_comprehension"}
            },
            {
                "question_text": "Implement a function to check if a string is a palindrome.",
                "answer": "def is_palindrome(s):\n    return s == s[::-1]",
                "choices": [],
                "category": "programming",
                "difficulty": "easy",
                "metadata": {"language": "python", "topic": "string_manipulation"}
            }
        ]
        
        # Sample HellaSwag-style questions
        hellaswag_info = BenchmarkInfo(
            name="HellaSwag-Sample",
            type=BenchmarkType.READING_COMPREHENSION,
            description="Sample commonsense reasoning questions",
            source_url="https://rowanzellers.com/hellaswag/",
            publication_date="2019-05-19",
            num_examples=30,
            languages=["en"],
            domains=["commonsense", "reasoning"],
            license="MIT",
            citation="Zellers et al. (2019)"
        )
        
        hellaswag_questions = [
            {
                "question_text": "A woman is outside with a bucket and a dog. The dog is running around in a circle. She",
                "answer": "throws a ball for the dog to fetch.",
                "choices": [
                    "throws a ball for the dog to fetch.",
                    "starts doing cartwheels.",
                    "begins to eat the bucket.",
                    "flies into the air."
                ],
                "category": "commonsense",
                "difficulty": "medium",
                "metadata": {"domain": "everyday_activities", "reasoning_type": "commonsense"}
            }
        ]
        
        try:
            # Add benchmarks to database
            self.database.add_benchmark(mmlu_info)
            self.database.add_questions("MMLU-Sample", mmlu_questions)
            
            self.database.add_benchmark(gsm8k_info)
            self.database.add_questions("GSM8K-Sample", gsm8k_questions)
            
            self.database.add_benchmark(humaneval_info)
            self.database.add_questions("HumanEval-Sample", humaneval_questions)
            
            self.database.add_benchmark(hellaswag_info)
            self.database.add_questions("HellaSwag-Sample", hellaswag_questions)
            
            self.logger.info("Sample benchmark data loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load sample data: {str(e)}")
            raise
    
    def load_from_json(self, file_path: str, benchmark_info: BenchmarkInfo):
        """
        Load benchmark data from JSON file.
        
        Args:
            file_path: Path to JSON file
            benchmark_info: Benchmark metadata
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Add benchmark info
            self.database.add_benchmark(benchmark_info)
            
            # Process questions
            questions = []
            if isinstance(data, list):
                questions = data
            elif isinstance(data, dict) and 'questions' in data:
                questions = data['questions']
            else:
                raise ValueError("Invalid JSON format")
            
            # Add questions to database
            self.database.add_questions(benchmark_info.name, questions)
            
            self.logger.info(f"Loaded {len(questions)} questions from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load from {file_path}: {str(e)}")
            raise
    
    def load_from_directory(self, directory_path: str):
        """
        Load all benchmark files from a directory.
        
        Args:
            directory_path: Path to directory containing benchmark files
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        loaded_count = 0
        for file_path in directory.glob("*.json"):
            try:
                # Try to infer benchmark info from filename or load from metadata
                benchmark_name = file_path.stem
                
                # Create default benchmark info (should be customized per dataset)
                benchmark_info = BenchmarkInfo(
                    name=benchmark_name,
                    type=BenchmarkType.MULTIPLE_CHOICE,  # Default type
                    description=f"Benchmark loaded from {file_path.name}",
                    source_url="",
                    publication_date="",
                    num_examples=0,
                    languages=["en"],
                    domains=["general"],
                    license="",
                    citation="",
                )
                
                self.load_from_json(str(file_path), benchmark_info)
                loaded_count += 1
                
            except Exception as e:
                self.logger.warning(f"Failed to load {file_path}: {str(e)}")
        
        self.logger.info(f"Loaded {loaded_count} benchmark files from {directory_path}")
    
    def get_load_status(self) -> Dict[str, Any]:
        """Get status of loaded benchmarks."""
        stats = self.database.get_statistics()
        benchmarks = self.database.get_all_benchmarks()
        
        return {
            "total_benchmarks": stats["total_benchmarks"],
            "total_questions": stats["total_questions"],
            "benchmarks": [
                {
                    "name": b.name,
                    "type": b.type.value,
                    "num_examples": b.num_examples,
                    "domains": b.domains,
                }
                for b in benchmarks
            ],
            "questions_by_type": stats["questions_by_type"],
        }
