"""
FastAPI server for IsItBenchmark API.

This module provides REST API endpoints for benchmark contamination detection
and a web interface for interactive analysis.
"""

import time
import logging
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

from .models import (
    AnalysisRequest, AnalysisResponse, BenchmarkMatchResponse,
    BenchmarkInfoResponse, DatabaseStatsResponse, HealthCheckResponse
)
from ..detection.detector import BenchmarkDetector
from ..utils.config import Config


# Global variables
app_start_time = time.time()
detector: Optional[BenchmarkDetector] = None
config: Optional[Config] = None


def create_app(config_path: Optional[str] = None) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured FastAPI application
    """
    global detector, config
    
    # Load configuration
    config = Config(config_path)
    
    # Initialize the detector
    detector = BenchmarkDetector(
        database_path=config.database.database_path,
        similarity_threshold=config.detection.similarity_threshold,
        enable_semantic_matching=config.detection.enable_semantic_matching,
        enable_fuzzy_matching=config.detection.enable_fuzzy_matching,
        max_matches=config.detection.max_matches,
    )
    
    # Create FastAPI app
    app = FastAPI(
        title="IsItBenchmark API",
        description="Benchmark Contamination Detection for AI/ML Systems",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Add CORS middleware
    if config.api.cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    return app


# Create the app instance
app = create_app()


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>IsItBenchmark - Benchmark Contamination Detection</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .header { text-align: center; margin-bottom: 30px; }
            .input-section { margin-bottom: 20px; }
            .results-section { margin-top: 20px; }
            textarea { width: 100%; height: 100px; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }
            button { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background-color: #0056b3; }
            .result { background-color: #f8f9fa; padding: 15px; border-radius: 4px; margin-bottom: 10px; }
            .probability { font-size: 1.2em; font-weight: bold; }
            .high-prob { color: #dc3545; }
            .medium-prob { color: #ffc107; }
            .low-prob { color: #28a745; }
            .match { background-color: #e9ecef; padding: 10px; margin: 5px 0; border-radius: 4px; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎯 IsItBenchmark</h1>
            <p>Benchmark Contamination Detection for AI/ML Systems</p>
        </div>
        
        <div class="input-section">
            <h3>Analyze a Prompt</h3>
            <textarea id="promptInput" placeholder="Enter a prompt to analyze for benchmark contamination..."></textarea>
            <br><br>
            <button onclick="analyzePrompt()">Analyze Prompt</button>
        </div>
        
        <div class="results-section" id="results" style="display: none;">
            <h3>Analysis Results</h3>
            <div id="resultContent"></div>
        </div>
        
        <script>
            async function analyzePrompt() {
                const prompt = document.getElementById('promptInput').value.trim();
                if (!prompt) {
                    alert('Please enter a prompt to analyze');
                    return;
                }
                
                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ prompt: prompt })
                    });
                    
                    const result = await response.json();
                    displayResults(result);
                } catch (error) {
                    alert('Error analyzing prompt: ' + error.message);
                }
            }
            
            function displayResults(result) {
                const resultsDiv = document.getElementById('results');
                const contentDiv = document.getElementById('resultContent');
                
                let probClass = 'low-prob';
                if (result.probability > 0.7) probClass = 'high-prob';
                else if (result.probability > 0.4) probClass = 'medium-prob';
                
                let html = `
                    <div class="result">
                        <div class="probability ${probClass}">
                            Benchmark Probability: ${(result.probability * 100).toFixed(1)}%
                        </div>
                        <p><strong>Confidence:</strong> ${result.confidence}</p>
                        <p><strong>Likely Benchmark:</strong> ${result.is_likely_benchmark ? 'Yes' : 'No'}</p>
                        <p><strong>Analysis Time:</strong> ${result.analysis_time_ms.toFixed(1)}ms</p>
                `;
                
                if (result.matches && result.matches.length > 0) {
                    html += '<h4>Matches Found:</h4>';
                    result.matches.forEach(match => {
                        html += `
                            <div class="match">
                                <strong>${match.benchmark_name}</strong> (${match.benchmark_type})
                                <br>Similarity: ${(match.similarity_score * 100).toFixed(1)}%
                                <br>Exact Match: ${match.exact_match ? 'Yes' : 'No'}
                                <br><em>"${match.matched_text}"</em>
                            </div>
                        `;
                    });
                } else {
                    html += '<p>No benchmark matches found.</p>';
                }
                
                html += '</div>';
                contentDiv.innerHTML = html;
                resultsDiv.style.display = 'block';
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_prompt(request: AnalysisRequest):
    """
    Analyze a prompt for benchmark contamination.
    
    Args:
        request: Analysis request with prompt and options
        
    Returns:
        Analysis results with probability and matches
    """
    if not detector:
        raise HTTPException(status_code=500, detail="Detector not initialized")
    
    try:
        # Perform analysis
        result = detector.analyze(
            request.prompt,
            similarity_threshold=request.similarity_threshold,
            max_matches=request.max_matches,
        )
        
        # Convert matches to response format
        matches = [
            BenchmarkMatchResponse(
                benchmark_name=match.benchmark_name,
                benchmark_type=match.benchmark_type.value,
                similarity_score=match.similarity_score,
                exact_match=match.exact_match,
                matched_text=match.matched_text,
                source_url=match.source_url,
                publication_date=match.publication_date,
                dataset_version=match.dataset_version,
            )
            for match in result.matches
        ]
        
        return AnalysisResponse(
            input_prompt=result.input_prompt,
            probability=result.probability,
            confidence=result.confidence.value,
            is_likely_benchmark=result.is_likely_benchmark,
            analysis_time_ms=result.analysis_time_ms,
            matches=matches,
            top_match=result.top_match.to_dict() if result.top_match else None,
            metadata=result.metadata,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/benchmarks", response_model=List[BenchmarkInfoResponse])
async def get_benchmarks():
    """Get list of all supported benchmarks."""
    if not detector:
        raise HTTPException(status_code=500, detail="Detector not initialized")
    
    try:
        benchmarks = detector.get_supported_benchmarks()
        return [
            BenchmarkInfoResponse(
                name=b.name,
                type=b.type.value,
                description=b.description,
                source_url=b.source_url,
                publication_date=b.publication_date,
                num_examples=b.num_examples,
                languages=b.languages,
                domains=b.domains,
                license=b.license,
                citation=b.citation,
            )
            for b in benchmarks
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get benchmarks: {str(e)}")


@app.get("/benchmarks/{benchmark_name}", response_model=BenchmarkInfoResponse)
async def get_benchmark_info(benchmark_name: str):
    """Get detailed information about a specific benchmark."""
    if not detector:
        raise HTTPException(status_code=500, detail="Detector not initialized")
    
    try:
        benchmark = detector.get_benchmark_info(benchmark_name)
        if not benchmark:
            raise HTTPException(status_code=404, detail=f"Benchmark '{benchmark_name}' not found")
        
        return BenchmarkInfoResponse(
            name=benchmark.name,
            type=benchmark.type.value,
            description=benchmark.description,
            source_url=benchmark.source_url,
            publication_date=benchmark.publication_date,
            num_examples=benchmark.num_examples,
            languages=benchmark.languages,
            domains=benchmark.domains,
            license=benchmark.license,
            citation=benchmark.citation,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get benchmark info: {str(e)}")


@app.get("/stats", response_model=DatabaseStatsResponse)
async def get_database_stats():
    """Get database statistics."""
    if not detector:
        raise HTTPException(status_code=500, detail="Detector not initialized")
    
    try:
        stats = detector.database.get_statistics()
        return DatabaseStatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    if not detector:
        raise HTTPException(status_code=500, detail="Detector not initialized")
    
    try:
        stats = detector.database.get_statistics()
        
        # Check semantic model status
        semantic_status = "available"
        if detector.semantic_matcher:
            model_info = detector.semantic_matcher.get_model_info()
            semantic_status = model_info.get("status", "unknown")
        else:
            semantic_status = "disabled"
        
        return HealthCheckResponse(
            status="healthy",
            version="0.1.0",
            uptime_seconds=time.time() - app_start_time,
            database_status="connected",
            semantic_model_status=semantic_status,
            total_benchmarks=stats["total_benchmarks"],
            total_questions=stats["total_questions"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/update-database")
async def update_database(force_refresh: bool = False):
    """Update the benchmark database."""
    if not detector:
        raise HTTPException(status_code=500, detail="Detector not initialized")
    
    try:
        success = detector.update_database(force_refresh=force_refresh)
        if success:
            return {"status": "success", "message": "Database updated successfully"}
        else:
            raise HTTPException(status_code=500, detail="Database update failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database update failed: {str(e)}")


def run_server(host: str = "localhost", port: int = 8000, debug: bool = False):
    """
    Run the FastAPI server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode
    """
    uvicorn.run(
        "src.api.server:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if not debug else "debug",
    )


if __name__ == "__main__":
    run_server()
