"""
Benchmark database management for storing and querying benchmark datasets.

This module provides the BenchmarkDatabase class for managing benchmark data,
including storage, retrieval, and efficient querying capabilities.
"""

import sqlite3
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import asdict

from ..detection.models import BenchmarkInfo, BenchmarkType


class BenchmarkDatabase:
    """
    Database manager for benchmark datasets.
    
    Provides efficient storage and retrieval of benchmark questions,
    metadata, and similarity search capabilities.
    """
    
    def __init__(self, database_path: Optional[str] = None):
        """
        Initialize the benchmark database.
        
        Args:
            database_path: Path to SQLite database file (uses default if None)
        """
        if database_path is None:
            # Use default path in data directory
            data_dir = Path(__file__).parent.parent.parent / "data"
            data_dir.mkdir(exist_ok=True)
            database_path = str(data_dir / "benchmarks.db")
        
        self.database_path = database_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self._init_database()
        self.logger.info(f"BenchmarkDatabase initialized: {database_path}")
    
    def _get_connection(self):
        """Get a database connection."""
        return sqlite3.connect(self.database_path)
    
    def _init_database(self):
        """Initialize database tables if they don't exist."""
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            
            # Create benchmarks table for metadata
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS benchmarks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    type TEXT NOT NULL,
                    description TEXT,
                    source_url TEXT,
                    publication_date TEXT,
                    num_examples INTEGER,
                    languages TEXT,  -- JSON array
                    domains TEXT,    -- JSON array
                    license TEXT,
                    citation TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create questions table for individual benchmark items
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    benchmark_id INTEGER,
                    question_text TEXT NOT NULL,
                    answer TEXT,
                    choices TEXT,     -- JSON array for multiple choice
                    category TEXT,
                    difficulty TEXT,
                    metadata TEXT,    -- JSON object for additional data
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (benchmark_id) REFERENCES benchmarks (id)
                )
            """)
            
            # Create indexes for efficient searching
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_questions_text ON questions(question_text)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_questions_benchmark ON questions(benchmark_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_benchmarks_name ON benchmarks(name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_benchmarks_type ON benchmarks(type)")
            
            conn.commit()
    
    def add_benchmark(self, benchmark_info: BenchmarkInfo) -> int:
        """
        Add a new benchmark to the database.
        
        Args:
            benchmark_info: Benchmark metadata
            
        Returns:
            Database ID of the inserted benchmark
        """
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO benchmarks 
                (name, type, description, source_url, publication_date, 
                 num_examples, languages, domains, license, citation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                benchmark_info.name,
                benchmark_info.type.value,
                benchmark_info.description,
                benchmark_info.source_url,
                benchmark_info.publication_date,
                benchmark_info.num_examples,
                json.dumps(benchmark_info.languages),
                json.dumps(benchmark_info.domains),
                benchmark_info.license,
                benchmark_info.citation,
            ))
            
            return cursor.lastrowid
    
    def add_questions(self, benchmark_name: str, questions: List[Dict[str, Any]]):
        """
        Add questions for a benchmark.
        
        Args:
            benchmark_name: Name of the benchmark
            questions: List of question dictionaries
        """
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            
            # Get benchmark ID
            cursor.execute("SELECT id FROM benchmarks WHERE name = ?", (benchmark_name,))
            result = cursor.fetchone()
            if not result:
                raise ValueError(f"Benchmark '{benchmark_name}' not found")
            
            benchmark_id = result[0]
            
            # Insert questions
            for question in questions:
                cursor.execute("""
                    INSERT INTO questions 
                    (benchmark_id, question_text, answer, choices, category, difficulty, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    benchmark_id,
                    question.get("question_text", ""),
                    question.get("answer", ""),
                    json.dumps(question.get("choices", [])),
                    question.get("category", ""),
                    question.get("difficulty", ""),
                    json.dumps(question.get("metadata", {})),
                ))
            
            conn.commit()
    
    def find_exact_matches(self, query: str) -> List[Dict[str, Any]]:
        """
        Find exact text matches in the database.
        
        Args:
            query: Query text to match
            
        Returns:
            List of exact matches with metadata
        """
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT q.question_text, b.name, b.type, b.source_url, 
                       b.publication_date, q.metadata
                FROM questions q
                JOIN benchmarks b ON q.benchmark_id = b.id
                WHERE q.question_text = ?
            """, (query,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "matched_text": row[0],
                    "benchmark_name": row[1],
                    "benchmark_type": row[2],
                    "source_url": row[3],
                    "publication_date": row[4],
                    "metadata": json.loads(row[5]) if row[5] else {},
                })
            
            return results
    
    def search_questions(
        self, 
        query: str, 
        benchmark_types: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search questions using SQL LIKE matching.
        
        Args:
            query: Search query
            benchmark_types: Filter by benchmark types
            limit: Maximum number of results
            
        Returns:
            List of matching questions with metadata
        """
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            
            base_query = """
                SELECT q.question_text, b.name, b.type, b.source_url, 
                       b.publication_date, q.metadata
                FROM questions q
                JOIN benchmarks b ON q.benchmark_id = b.id
                WHERE q.question_text LIKE ?
            """
            
            params = [f"%{query}%"]
            
            if benchmark_types:
                placeholders = ",".join("?" * len(benchmark_types))
                base_query += f" AND b.type IN ({placeholders})"
                params.extend(benchmark_types)
            
            base_query += " LIMIT ?"
            params.append(limit)
            
            cursor.execute(base_query, params)
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "matched_text": row[0],
                    "benchmark_name": row[1],
                    "benchmark_type": row[2],
                    "source_url": row[3],
                    "publication_date": row[4],
                    "metadata": json.loads(row[5]) if row[5] else {},
                })
            
            return results
    
    def get_all_benchmarks(self) -> List[BenchmarkInfo]:
        """Get list of all benchmarks in the database."""
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT name, type, description, source_url, publication_date,
                       num_examples, languages, domains, license, citation
                FROM benchmarks
                ORDER BY name
            """)
            
            benchmarks = []
            for row in cursor.fetchall():
                benchmark = BenchmarkInfo(
                    name=row[0],
                    type=BenchmarkType(row[1]),
                    description=row[2],
                    source_url=row[3],
                    publication_date=row[4],
                    num_examples=row[5],
                    languages=json.loads(row[6]) if row[6] else [],
                    domains=json.loads(row[7]) if row[7] else [],
                    license=row[8],
                    citation=row[9],
                )
                benchmarks.append(benchmark)
            
            return benchmarks
    
    def get_benchmark_info(self, benchmark_name: str) -> Optional[BenchmarkInfo]:
        """Get detailed information about a specific benchmark."""
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT name, type, description, source_url, publication_date,
                       num_examples, languages, domains, license, citation
                FROM benchmarks
                WHERE name = ?
            """, (benchmark_name,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return BenchmarkInfo(
                name=row[0],
                type=BenchmarkType(row[1]),
                description=row[2],
                source_url=row[3],
                publication_date=row[4],
                num_examples=row[5],
                languages=json.loads(row[6]) if row[6] else [],
                domains=json.loads(row[7]) if row[7] else [],
                license=row[8],
                citation=row[9],
            )
    
    def clear_all_data(self):
        """
        Clear all data from the database (benchmarks and questions).
        """
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM questions")
            cursor.execute("DELETE FROM benchmarks")
            conn.commit()
            self.logger.info("Cleared all data from database")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            
            # Count benchmarks
            cursor.execute("SELECT COUNT(*) FROM benchmarks")
            num_benchmarks = cursor.fetchone()[0]
            
            # Count questions
            cursor.execute("SELECT COUNT(*) FROM questions")
            num_questions = cursor.fetchone()[0]
            
            # Count by type
            cursor.execute("""
                SELECT b.type, COUNT(q.id) as question_count
                FROM benchmarks b
                LEFT JOIN questions q ON b.id = q.benchmark_id
                GROUP BY b.type
            """)
            
            type_counts = {}
            for row in cursor.fetchall():
                type_counts[row[0]] = row[1]
            
            return {
                "total_benchmarks": num_benchmarks,
                "total_questions": num_questions,
                "questions_by_type": type_counts,
                "database_path": self.database_path,
            }
    
    def update(self, force_refresh: bool = False) -> bool:
        """
        Update the database with latest benchmark data.
        
        Args:
            force_refresh: Whether to force a complete refresh
            
        Returns:
            True if update was successful
        """
        try:
            # This would typically download and update benchmark data
            # For now, we'll just ensure the database is properly initialized
            self._init_database()
            
            # In a real implementation, this would:
            # 1. Check for updates to benchmark datasets
            # 2. Download new data if available
            # 3. Update the database with new questions
            # 4. Update metadata and statistics
            
            self.logger.info("Database update completed successfully")
            return True
        except Exception as e:
            self.logger.error(f"Database update failed: {str(e)}")
            return False
