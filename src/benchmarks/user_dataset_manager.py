"""
User dataset management for IsItBenchmark.

This module provides functionality for users to add their own benchmark datasets
in various formats (JSON, CSV, JSONL) with automatic format detection and validation.
"""

import json
import csv
import logging
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

from ..detection.models import BenchmarkInfo, BenchmarkType
from .database import BenchmarkDatabase


class DatasetFormat(Enum):
    """Supported dataset formats."""
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    TSV = "tsv"
    PARQUET = "parquet"


@dataclass
class DatasetValidationResult:
    """Result of dataset validation."""
    is_valid: bool
    format: Optional[DatasetFormat]
    num_questions: int
    sample_questions: List[Dict[str, Any]]
    errors: List[str]
    warnings: List[str]
    suggested_benchmark_info: Optional[BenchmarkInfo]


class UserDatasetManager:
    """
    Manager for user-provided benchmark datasets.
    
    Handles loading, validation, and integration of user datasets
    with automatic format detection and comprehensive error handling.
    """
    
    def __init__(self, database: BenchmarkDatabase):
        """
        Initialize the user dataset manager.
        
        Args:
            database: BenchmarkDatabase instance
        """
        self.database = database
        self.logger = logging.getLogger(__name__)
        
        # Required fields for questions
        self.required_fields = {"question_text"}
        self.optional_fields = {
            "answer", "choices", "category", "difficulty", 
            "metadata", "explanation", "source", "id"
        }
        
    def validate_dataset(self, file_path: str) -> DatasetValidationResult:
        """
        Validate a user dataset file.
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            DatasetValidationResult with validation details
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return DatasetValidationResult(
                is_valid=False,
                format=None,
                num_questions=0,
                sample_questions=[],
                errors=[f"File not found: {file_path}"],
                warnings=[],
                suggested_benchmark_info=None
            )
        
        # Detect format
        format_result = self._detect_format(file_path)
        if not format_result[0]:
            return DatasetValidationResult(
                is_valid=False,
                format=None,
                num_questions=0,
                sample_questions=[],
                errors=[f"Unsupported file format: {file_path.suffix}"],
                warnings=[],
                suggested_benchmark_info=None
            )
        
        detected_format = format_result[1]
        
        try:
            # Load and validate data
            questions = self._load_questions_by_format(file_path, detected_format)
            validation_errors = []
            validation_warnings = []
            
            if not questions:
                validation_errors.append("No questions found in dataset")
            
            # Validate question structure
            for i, question in enumerate(questions[:10]):  # Check first 10 questions
                if not isinstance(question, dict):
                    validation_errors.append(f"Question {i+1} is not a dictionary")
                    continue
                
                # Check required fields
                missing_fields = self.required_fields - set(question.keys())
                if missing_fields:
                    validation_errors.append(
                        f"Question {i+1} missing required fields: {missing_fields}"
                    )
                
                # Check for empty question text
                if not question.get("question_text", "").strip():
                    validation_errors.append(f"Question {i+1} has empty question_text")
                
                # Validate choices format if present
                if "choices" in question and question["choices"]:
                    if not isinstance(question["choices"], list):
                        validation_warnings.append(
                            f"Question {i+1} choices should be a list"
                        )
            
            # Generate suggested benchmark info
            suggested_info = self._generate_benchmark_info(file_path, questions)
            
            return DatasetValidationResult(
                is_valid=len(validation_errors) == 0,
                format=detected_format,
                num_questions=len(questions),
                sample_questions=questions[:3],  # First 3 questions as samples
                errors=validation_errors,
                warnings=validation_warnings,
                suggested_benchmark_info=suggested_info
            )
            
        except Exception as e:
            return DatasetValidationResult(
                is_valid=False,
                format=detected_format,
                num_questions=0,
                sample_questions=[],
                errors=[f"Error loading dataset: {str(e)}"],
                warnings=[],
                suggested_benchmark_info=None
            )
    
    def add_user_dataset(
        self, 
        file_path: str, 
        benchmark_info: Optional[BenchmarkInfo] = None,
        validate_first: bool = True
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Add a user dataset to the database.
        
        Args:
            file_path: Path to the dataset file
            benchmark_info: Benchmark metadata (auto-generated if None)
            validate_first: Whether to validate before adding
            
        Returns:
            Tuple of (success, message, stats)
        """
        try:
            # Validate dataset first
            if validate_first:
                validation = self.validate_dataset(file_path)
                if not validation.is_valid:
                    return False, f"Validation failed: {'; '.join(validation.errors)}", {}
                
                # Use suggested benchmark info if none provided
                if benchmark_info is None:
                    benchmark_info = validation.suggested_benchmark_info
            
            # Load questions
            file_path = Path(file_path)
            detected_format = self._detect_format(file_path)[1]
            questions = self._load_questions_by_format(file_path, detected_format)
            
            if not questions:
                return False, "No questions found in dataset", {}
            
            # Ensure benchmark info is provided
            if benchmark_info is None:
                benchmark_info = self._generate_benchmark_info(file_path, questions)
            
            # Update num_examples to actual count
            benchmark_info.num_examples = len(questions)
            
            # Add to database
            self.database.add_benchmark(benchmark_info)
            self.database.add_questions(benchmark_info.name, questions)
            
            stats = {
                "benchmark_name": benchmark_info.name,
                "num_questions": len(questions),
                "format": detected_format.value,
                "file_size": file_path.stat().st_size,
                "benchmark_type": benchmark_info.type.value
            }
            
            self.logger.info(f"Successfully added user dataset: {benchmark_info.name}")
            return True, f"Successfully added {len(questions)} questions", stats
            
        except Exception as e:
            error_msg = f"Failed to add dataset: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg, {}
    
    def list_user_datasets(self) -> List[Dict[str, Any]]:
        """
        List all datasets in the database with metadata.
        
        Returns:
            List of dataset information dictionaries
        """
        benchmarks = self.database.get_all_benchmarks()
        dataset_list = []
        
        for benchmark in benchmarks:
            # Get question count for this benchmark
            with self.database._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) FROM questions q
                    JOIN benchmarks b ON q.benchmark_id = b.id
                    WHERE b.name = ?
                """, (benchmark.name,))
                question_count = cursor.fetchone()[0]
            
            dataset_info = {
                "name": benchmark.name,
                "type": benchmark.type.value,
                "description": benchmark.description,
                "num_questions": question_count,
                "domains": benchmark.domains,
                "languages": benchmark.languages,
                "source_url": benchmark.source_url,
                "publication_date": benchmark.publication_date,
                "license": benchmark.license
            }
            dataset_list.append(dataset_info)
        
        return dataset_list
    
    def remove_dataset(self, benchmark_name: str) -> Tuple[bool, str]:
        """
        Remove a dataset from the database.
        
        Args:
            benchmark_name: Name of the benchmark to remove
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Check if benchmark exists
            benchmark_info = self.database.get_benchmark_info(benchmark_name)
            if not benchmark_info:
                return False, f"Benchmark '{benchmark_name}' not found"
            
            # Remove from database
            with self.database._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get benchmark ID
                cursor.execute("SELECT id FROM benchmarks WHERE name = ?", (benchmark_name,))
                result = cursor.fetchone()
                if not result:
                    return False, f"Benchmark '{benchmark_name}' not found"
                
                benchmark_id = result[0]
                
                # Delete questions first (foreign key constraint)
                cursor.execute("DELETE FROM questions WHERE benchmark_id = ?", (benchmark_id,))
                questions_deleted = cursor.rowcount
                
                # Delete benchmark
                cursor.execute("DELETE FROM benchmarks WHERE id = ?", (benchmark_id,))
                
                conn.commit()
            
            message = f"Removed benchmark '{benchmark_name}' ({questions_deleted} questions)"
            self.logger.info(message)
            return True, message
            
        except Exception as e:
            error_msg = f"Failed to remove dataset: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def _detect_format(self, file_path: Path) -> Tuple[bool, Optional[DatasetFormat]]:
        """Detect the format of a dataset file."""
        suffix = file_path.suffix.lower()
        
        format_mapping = {
            ".json": DatasetFormat.JSON,
            ".jsonl": DatasetFormat.JSONL,
            ".csv": DatasetFormat.CSV,
            ".tsv": DatasetFormat.TSV,
            ".parquet": DatasetFormat.PARQUET
        }
        
        if suffix in format_mapping:
            return True, format_mapping[suffix]
        
        return False, None
    
    def _load_questions_by_format(
        self, 
        file_path: Path, 
        format: DatasetFormat
    ) -> List[Dict[str, Any]]:
        """Load questions based on detected format."""
        
        if format == DatasetFormat.JSON:
            return self._load_json(file_path)
        elif format == DatasetFormat.JSONL:
            return self._load_jsonl(file_path)
        elif format == DatasetFormat.CSV:
            return self._load_csv(file_path)
        elif format == DatasetFormat.TSV:
            return self._load_csv(file_path, delimiter='\t')
        elif format == DatasetFormat.PARQUET:
            return self._load_parquet(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _load_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load questions from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Try common keys for question lists
            for key in ['questions', 'data', 'examples', 'items']:
                if key in data and isinstance(data[key], list):
                    return data[key]
            # If no list found, treat as single question
            return [data]
        else:
            raise ValueError("Invalid JSON structure")
    
    def _load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load questions from JSONL file."""
        questions = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    question = json.loads(line)
                    questions.append(question)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_num}: {e}")
        return questions
    
    def _load_csv(self, file_path: Path, delimiter: str = ',') -> List[Dict[str, Any]]:
        """Load questions from CSV/TSV file."""
        questions = []
        with open(file_path, 'r', encoding='utf-8') as f:
            # Try to detect if choices are in separate columns or JSON format
            reader = csv.DictReader(f, delimiter=delimiter)
            
            for row_num, row in enumerate(reader, 1):
                try:
                    question = {}
                    
                    # Handle different column name variations
                    for key, value in row.items():
                        if not value:  # Skip empty values
                            continue
                            
                        # Normalize column names
                        normalized_key = self._normalize_column_name(key)
                        
                        # Handle special fields
                        if normalized_key in ['choices', 'metadata'] and isinstance(value, str):
                            try:
                                # Try to parse as JSON
                                question[normalized_key] = json.loads(value)
                            except json.JSONDecodeError:
                                # If not JSON, split by delimiter for choices
                                if normalized_key == 'choices':
                                    question[normalized_key] = [c.strip() for c in value.split('|')]
                                else:
                                    question[normalized_key] = value
                        else:
                            question[normalized_key] = value
                    
                    if question:  # Only add non-empty questions
                        questions.append(question)
                        
                except Exception as e:
                    raise ValueError(f"Error processing row {row_num}: {e}")
        
        return questions
    
    def _load_parquet(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load questions from Parquet file."""
        try:
            df = pd.read_parquet(file_path)
            
            # Convert DataFrame to list of dictionaries
            questions = []
            for _, row in df.iterrows():
                question = {}
                for col, value in row.items():
                    if pd.notna(value):  # Skip NaN values
                        normalized_key = self._normalize_column_name(col)
                        
                        # Handle special fields
                        if normalized_key in ['choices', 'metadata'] and isinstance(value, str):
                            try:
                                question[normalized_key] = json.loads(value)
                            except json.JSONDecodeError:
                                if normalized_key == 'choices':
                                    question[normalized_key] = [c.strip() for c in value.split('|')]
                                else:
                                    question[normalized_key] = value
                        else:
                            question[normalized_key] = value
                
                if question:
                    questions.append(question)
            
            return questions
            
        except ImportError:
            raise ValueError("pandas is required to load Parquet files")
    
    def _normalize_column_name(self, column_name: str) -> str:
        """Normalize column names to standard field names."""
        column_name = column_name.lower().strip()
        
        # Mapping of common variations to standard names
        name_mapping = {
            'question': 'question_text',
            'text': 'question_text',
            'prompt': 'question_text',
            'query': 'question_text',
            'input': 'question_text',
            'correct_answer': 'answer',
            'target': 'answer',
            'label': 'answer',
            'ground_truth': 'answer',
            'options': 'choices',
            'alternatives': 'choices',
            'multiple_choice': 'choices',
            'type': 'category',
            'subject': 'category',
            'topic': 'category',
            'level': 'difficulty',
            'hard': 'difficulty',
        }
        
        return name_mapping.get(column_name, column_name)
    
    def _generate_benchmark_info(
        self, 
        file_path: Path, 
        questions: List[Dict[str, Any]]
    ) -> BenchmarkInfo:
        """Generate benchmark info from file and questions."""
        
        # Infer benchmark type from questions
        benchmark_type = self._infer_benchmark_type(questions)
        
        # Extract domains from categories if available
        domains = set()
        for question in questions[:100]:  # Sample first 100 questions
            if 'category' in question:
                domains.add(str(question['category']).lower())
        
        domains_list = list(domains)[:5] if domains else ["general"]
        
        return BenchmarkInfo(
            name=file_path.stem,
            type=benchmark_type,
            description=f"User-provided benchmark dataset from {file_path.name}",
            source_url="",
            publication_date="",
            num_examples=len(questions),
            languages=["en"],  # Default to English
            domains=domains_list,
            license="",
            citation=""
        )
    
    def _infer_benchmark_type(self, questions: List[Dict[str, Any]]) -> BenchmarkType:
        """Infer benchmark type from question structure."""
        if not questions:
            return BenchmarkType.MULTIPLE_CHOICE
        
        # Sample first few questions to infer type
        sample_questions = questions[:10]
        
        has_choices = any('choices' in q and q['choices'] for q in sample_questions)
        has_math_keywords = any(
            any(keyword in str(q.get('question_text', '')).lower() 
                for keyword in ['calculate', 'solve', 'equation', 'number', 'math'])
            for q in sample_questions
        )
        has_code_keywords = any(
            any(keyword in str(q.get('question_text', '')).lower() 
                for keyword in ['function', 'code', 'program', 'algorithm', 'python'])
            for q in sample_questions
        )
        
        # Infer type based on content
        if has_code_keywords:
            return BenchmarkType.CODE_GENERATION
        elif has_math_keywords and not has_choices:
            return BenchmarkType.MATH
        elif has_choices:
            return BenchmarkType.MULTIPLE_CHOICE
        else:
            return BenchmarkType.QUESTION_ANSWERING
