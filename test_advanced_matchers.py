#!/usr/bin/env python3
"""
Test script for advanced contamination detection matchers.

This script validates the N-gram overlap and membership inference matchers
to ensure they work correctly with the IsItBenchmark system.
"""

import sys
import os
import logging
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from similarity.matcher_factory import create_matcher, get_available_matchers
from similarity.base_matcher import MatcherType
from detection.benchmark_detector import BenchmarkDetector
from benchmarks.benchmark_loader import BenchmarkLoader

def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_matcher_factory():
    """Test that the matcher factory can create all matcher types."""
    print("\n=== Testing Matcher Factory ===")
    
    # Get available matchers
    available = get_available_matchers()
    print(f"Available matchers: {list(available.keys())}")
    
    # Test creating each matcher type
    for matcher_type in [MatcherType.SEMANTIC, MatcherType.LLM, MatcherType.NGRAM, MatcherType.MEMBERSHIP_INFERENCE]:
        try:
            print(f"\nTesting {matcher_type.value} matcher...")
            matcher = create_matcher(matcher_type.value)
            
            # Get matcher info
            info = matcher.get_matcher_info()
            print(f"  Type: {info['type']}")
            print(f"  Initialized: {info['is_initialized']}")
            print(f"  Capabilities: {info.get('capabilities', [])}")
            
            # Cleanup
            matcher.cleanup()
            print(f"  ✓ {matcher_type.value} matcher created and cleaned up successfully")
            
        except Exception as e:
            print(f"  ✗ Failed to create {matcher_type.value} matcher: {str(e)}")

def test_ngram_matcher():
    """Test N-gram overlap matcher specifically."""
    print("\n=== Testing N-gram Matcher ===")
    
    try:
        # Create N-gram matcher
        matcher = create_matcher("ngram")
        
        # Test similarity computation
        text1 = "What is the capital of France?"
        text2 = "What is the capital of France and Germany?"
        text3 = "How do you solve quadratic equations?"
        
        similarity1 = matcher.compute_similarity(text1, text2)
        similarity2 = matcher.compute_similarity(text1, text3)
        
        print(f"Similarity between similar texts: {similarity1:.3f}")
        print(f"Similarity between different texts: {similarity2:.3f}")
        
        # Test with benchmark-like data
        candidates = [
            {
                "question_text": "What is the capital of France?",
                "benchmark_name": "MMLU",
                "benchmark_type": "knowledge_evaluation",
                "source_url": "https://example.com/mmlu",
                "publication_date": "2020-09-07",
                "dataset_version": "1.0"
            },
            {
                "question_text": "How do you solve quadratic equations?",
                "benchmark_name": "GSM8K",
                "benchmark_type": "mathematical_reasoning",
                "source_url": "https://example.com/gsm8k",
                "publication_date": "2021-10-01",
                "dataset_version": "1.0"
            }
        ]
        
        matches = matcher.find_matches(
            query="What is the capital of France and Spain?",
            candidates=candidates,
            threshold=0.5,
            max_matches=5
        )
        
        print(f"Found {len(matches)} matches with N-gram matcher")
        for match in matches:
            print(f"  - {match.benchmark_name}: {match.confidence:.3f}")
        
        matcher.cleanup()
        print("✓ N-gram matcher test completed successfully")
        
    except Exception as e:
        print(f"✗ N-gram matcher test failed: {str(e)}")

def test_membership_inference_matcher():
    """Test membership inference matcher specifically."""
    print("\n=== Testing Membership Inference Matcher ===")
    
    try:
        # Create membership inference matcher
        matcher = create_matcher("membership_inference")
        
        # Test similarity computation
        text1 = "What is the capital of France?"
        text2 = "What is the capital of France and Germany?"
        text3 = "How do you solve quadratic equations?"
        
        similarity1 = matcher.compute_similarity(text1, text2)
        similarity2 = matcher.compute_similarity(text1, text3)
        
        print(f"Membership inference score for similar texts: {similarity1:.3f}")
        print(f"Membership inference score for different texts: {similarity2:.3f}")
        
        # Test with benchmark-like data
        candidates = [
            {
                "question_text": "What is the capital of France?",
                "benchmark_name": "MMLU",
                "benchmark_type": "knowledge_evaluation",
                "source_url": "https://example.com/mmlu",
                "publication_date": "2020-09-07",
                "dataset_version": "1.0"
            },
            {
                "question_text": "How do you solve quadratic equations?",
                "benchmark_name": "GSM8K",
                "benchmark_type": "mathematical_reasoning",
                "source_url": "https://example.com/gsm8k",
                "publication_date": "2021-10-01",
                "dataset_version": "1.0"
            }
        ]
        
        matches = matcher.find_matches(
            query="What is the capital of France?",
            candidates=candidates,
            threshold=0.5,
            max_matches=5
        )
        
        print(f"Found {len(matches)} matches with membership inference matcher")
        for match in matches:
            print(f"  - {match.benchmark_name}: {match.confidence:.3f}")
        
        matcher.cleanup()
        print("✓ Membership inference matcher test completed successfully")
        
    except Exception as e:
        print(f"✗ Membership inference matcher test failed: {str(e)}")

def test_detector_integration():
    """Test that the detector can use the new matchers."""
    print("\n=== Testing Detector Integration ===")
    
    # Test prompts
    test_prompts = [
        "What is the capital of France?",
        "Solve the equation 2x + 5 = 13",
        "Write a Python function to calculate factorial"
    ]
    
    # Test each new matcher type with the detector
    for matcher_type in ["ngram", "membership_inference"]:
        print(f"\nTesting detector with {matcher_type} matcher...")
        
        try:
            # Create detector configuration
            config = {
                "matcher_type": matcher_type,
                "scope": "all",
                "matcher_config": {}
            }
            
            # Initialize detector
            detector = BenchmarkDetector(config)
            
            # Test with a sample prompt
            prompt = test_prompts[0]
            result = detector.analyze_prompt(prompt)
            
            print(f"  Analysis completed for: '{prompt[:50]}...'")
            print(f"  Contamination probability: {result.contamination_probability:.3f}")
            print(f"  Matches found: {len(result.matches)}")
            print(f"  Matcher used: {result.analysis_metadata.get('matcher_type', 'unknown')}")
            
            print(f"  ✓ {matcher_type} detector integration successful")
            
        except Exception as e:
            print(f"  ✗ {matcher_type} detector integration failed: {str(e)}")

def main():
    """Run all tests."""
    print("Advanced Contamination Detection Matcher Tests")
    print("=" * 50)
    
    setup_logging()
    
    # Run tests
    test_matcher_factory()
    test_ngram_matcher()
    test_membership_inference_matcher()
    test_detector_integration()
    
    print("\n" + "=" * 50)
    print("Advanced matcher testing completed!")
    print("\nNext steps:")
    print("1. Run: python test_advanced_matchers.py")
    print("2. Test CLI: python main.py analyze 'What is the capital of France?' --matcher ngram")
    print("3. Test CLI: python main.py analyze 'Solve 2x + 5 = 13' --matcher membership_inference")

if __name__ == "__main__":
    main()
