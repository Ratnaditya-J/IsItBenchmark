"""
API data models for IsItBenchmark.

This module defines Pydantic models for API request/response handling.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class AnalysisRequest(BaseModel):
    """Request model for benchmark analysis."""
    prompt: str = Field(..., description="Text prompt to analyze for benchmark contamination")
    similarity_threshold: Optional[float] = Field(
        default=0.7, 
        ge=0.0, 
        le=1.0, 
        description="Minimum similarity threshold for matches"
    )
    max_matches: Optional[int] = Field(
        default=10, 
        ge=1, 
        le=100, 
        description="Maximum number of matches to return"
    )
    enable_semantic: Optional[bool] = Field(
        default=True, 
        description="Enable semantic similarity matching"
    )
    enable_fuzzy: Optional[bool] = Field(
        default=True, 
        description="Enable fuzzy text matching"
    )
    
    @validator('prompt')
    def prompt_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Prompt cannot be empty')
        return v


class BenchmarkMatchResponse(BaseModel):
    """Response model for individual benchmark matches."""
    benchmark_name: str
    benchmark_type: str
    similarity_score: float
    exact_match: bool
    matched_text: str
    source_url: Optional[str] = None
    publication_date: Optional[str] = None
    dataset_version: Optional[str] = None


class AnalysisResponse(BaseModel):
    """Response model for benchmark analysis results."""
    input_prompt: str
    probability: float = Field(..., ge=0.0, le=1.0)
    confidence: str
    is_likely_benchmark: bool
    analysis_time_ms: float
    matches: List[BenchmarkMatchResponse]
    top_match: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any]


class BenchmarkInfoResponse(BaseModel):
    """Response model for benchmark information."""
    name: str
    type: str
    description: str
    source_url: str
    publication_date: str
    num_examples: int
    languages: List[str]
    domains: List[str]
    license: Optional[str] = None
    citation: Optional[str] = None


class DatabaseStatsResponse(BaseModel):
    """Response model for database statistics."""
    total_benchmarks: int
    total_questions: int
    questions_by_type: Dict[str, int]
    database_path: str


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str
    version: str
    uptime_seconds: float
    database_status: str
    semantic_model_status: str
    total_benchmarks: int
    total_questions: int
