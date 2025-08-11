#!/usr/bin/env python3
"""
Test script for advanced contamination detection methods.

Tests the N-gram overlap and membership inference matchers to ensure
they work correctly with the new base matcher interface.
"""

import sys
import os
import logging
from typing import List, Dict, Any

# Add src to path and handle imports properly
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

# Set up the package structure
os.chdir(project_root)

try:
    from src.similarity.matcher_factory import create_matcher
    from src.similarity.base_matcher import MatcherType
    from src.detection.models import BenchmarkType
except ImportError:
    # Fallback to direct imports
    sys.path.insert(0, os.path.join(src_path, 'similarity'))
    sys.path.insert(0, os.path.join(src_path, 'detection'))
    
    from matcher_factory import create_matcher
    from base_matcher import MatcherType
    from models import BenchmarkType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_candidates() -> List[Dict[str, Any]]:
    """Create test benchmark candidates for testing."""
    return [
        {
            "question_text": "What is the capital of France?",
            "benchmark_name": "MMLU",
            "benchmark_type": "knowledge_evaluation",
            "source_url": "https://example.com/mmlu",
            "publication_date": "2020-09-07",
            "dataset_version": "1.0"
        },
        {
            "question_text": "Solve the equation: 2x + 5 = 13",
            "benchmark_name": "GSM8K",
            "benchmark_type": "mathematical_reasoning",
            "source_url": "https://example.com/gsm8k",
            "publication_date": "2021-10-15",
            "dataset_version": "1.0"
        },
        {
            "question_text": "What are the ethical implications of artificial intelligence?",
            "benchmark_name": "AI_Ethics",
            "benchmark_type": "safety_evaluation",
            "source_url": "https://example.com/ai_ethics",
            "publication_date": "2023-01-01",
            "dataset_version": "2.0"
        },
        {
            "question_text": "The quick brown fox jumps over the lazy dog",
            "benchmark_name": "Text_Samples",
            "benchmark_type": "language_modeling",
            "source_url": "https://example.com/text_samples",
            "publication_date": "2022-06-01",
            "dataset_version": "1.5"
        }
    ]

def test_ngram_matcher():
    """Test the N-gram overlap matcher."""
    logger.info("=" * 60)
    logger.info("Testing N-gram Overlap Matcher")
    logger.info("=" * 60)
    
    try:
        # Create N-gram matcher
        matcher = create_matcher(MatcherType.NGRAM)
        logger.info(f"Created matcher: {matcher.__class__.__name__}")
        
        # Initialize matcher
        if not matcher.initialize():
            logger.error("Failed to initialize N-gram matcher")
            return False
        
        # Test candidates
        candidates = create_test_candidates()
        
        # Test queries with different levels of overlap
        test_queries = [
            "What is the capital of France?",  # Exact match
            "What is the capital of France and Germany?",  # Partial match
            "Solve this equation: 2x + 5 = 13",  # Similar but different
            "The quick brown fox jumps",  # Partial n-gram match
            "How do neural networks work?",  # No match expected
        ]
        
        for query in test_queries:
            logger.info(f"\nTesting query: '{query}'")
            
            # Find matches
            matches = matcher.find_matches(
                query=query,
                candidates=candidates,
                threshold=0.3,  # Lower threshold for testing
                max_matches=5
            )
            
            logger.info(f"Found {len(matches)} matches:")
            for i, match in enumerate(matches, 1):
                benchmark_info = match.benchmark_type or match.metadata.get('benchmark_name', 'Unknown')
                logger.info(f"  {i}. {benchmark_info} (score: {match.similarity_score:.3f})")
                logger.info(f"     Text: {match.text[:100]}...")
                if hasattr(match, 'metadata') and match.metadata:
                    logger.info(f"     Method: {match.metadata.get('detection_method', 'unknown')}")
        
        # Test compute_similarity method
        logger.info(f"\nTesting compute_similarity method:")
        text1 = "What is the capital of France?"
        text2 = "What is the capital of France and Germany?"
        similarity = matcher.compute_similarity(text1, text2)
        logger.info(f"Similarity between texts: {similarity:.3f}")
        
        logger.info("N-gram matcher test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"N-gram matcher test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_membership_inference_matcher():
    """Test the membership inference matcher."""
    logger.info("=" * 60)
    logger.info("Testing Membership Inference Matcher")
    logger.info("=" * 60)
    
    try:
        # Create membership inference matcher
        matcher = create_matcher(MatcherType.MEMBERSHIP_INFERENCE)
        logger.info(f"Created matcher: {matcher.__class__.__name__}")
        
        # Initialize matcher
        if not matcher.initialize():
            logger.error("Failed to initialize membership inference matcher")
            return False
        
        # Test candidates
        candidates = create_test_candidates()
        
        # Test queries
        test_queries = [
            "What is the capital of France?",  # Should match benchmark question
            "Solve the equation: 2x + 5 = 13",  # Mathematical question
            "What are the ethical implications of AI?",  # Ethics question
            "How do I bake a chocolate cake?",  # Unrelated question
        ]
        
        for query in test_queries:
            logger.info(f"\nTesting query: '{query}'")
            
            # Find matches
            matches = matcher.find_matches(
                query=query,
                candidates=candidates,
                threshold=0.1,  # Lower threshold for testing
                max_matches=5
            )
            
            logger.info(f"Found {len(matches)} matches:")
            for i, match in enumerate(matches, 1):
                benchmark_info = match.benchmark_type or match.metadata.get('benchmark_name', 'Unknown')
                logger.info(f"  {i}. {benchmark_info} (score: {match.similarity_score:.3f})")
                logger.info(f"     Text: {match.text[:100]}...")
                if hasattr(match, 'metadata') and match.metadata:
                    logger.info(f"     Method: {match.metadata.get('detection_method', 'unknown')}")
                    logger.info(f"     Query perplexity: {match.metadata.get('query_perplexity', 'N/A')}")
                    logger.info(f"     Membership prob: {match.metadata.get('membership_probability', 'N/A')}")
        
        # Test compute_similarity method
        logger.info(f"\nTesting compute_similarity method:")
        text1 = "What is the capital of France?"
        text2 = "What is the capital of Germany?"
        similarity = matcher.compute_similarity(text1, text2)
        logger.info(f"Similarity between texts: {similarity:.3f}")
        
        logger.info("Membership inference matcher test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Membership inference matcher test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_matcher_factory_integration():
    """Test that the matcher factory correctly creates advanced matchers."""
    logger.info("=" * 60)
    logger.info("Testing Matcher Factory Integration")
    logger.info("=" * 60)
    
    try:
        try:
            from src.similarity.matcher_factory import get_available_matchers
        except ImportError:
            from matcher_factory import get_available_matchers
        
        # Get available matchers
        available = get_available_matchers()
        logger.info("Available matchers:")
        for matcher_type, info in available.items():
            logger.info(f"  - {matcher_type}: {info.get('description', 'No description')}")
        
        # Test creating each advanced matcher type
        advanced_types = [MatcherType.NGRAM, MatcherType.MEMBERSHIP_INFERENCE]
        
        for matcher_type in advanced_types:
            logger.info(f"\nTesting creation of {matcher_type.value} matcher...")
            matcher = create_matcher(matcher_type)
            logger.info(f"  Created: {matcher.__class__.__name__}")
            
            # Test initialization
            if matcher.initialize():
                logger.info(f"  Initialized successfully")
                
                # Get matcher info
                info = matcher.get_matcher_info()
                logger.info(f"  Matcher info: {info}")
                
                # Cleanup
                matcher.cleanup()
                logger.info(f"  Cleaned up successfully")
            else:
                logger.warning(f"  Failed to initialize {matcher_type.value} matcher")
        
        logger.info("Matcher factory integration test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Matcher factory integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all advanced detection tests."""
    logger.info("Starting Advanced Contamination Detection Tests")
    logger.info("=" * 80)
    
    results = []
    
    # Test N-gram matcher
    results.append(("N-gram Matcher", test_ngram_matcher()))
    
    # Test membership inference matcher
    results.append(("Membership Inference Matcher", test_membership_inference_matcher()))
    
    # Test matcher factory integration
    results.append(("Matcher Factory Integration", test_matcher_factory_integration()))
    
    # Print summary
    logger.info("=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    
    all_passed = True
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        logger.info(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 All advanced detection tests PASSED!")
        return 0
    else:
        logger.error("\n❌ Some tests FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
