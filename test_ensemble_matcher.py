#!/usr/bin/env python3
"""
Test script for the Ensemble Matcher.

Tests the multi-method ensemble detection system that combines
semantic, LLM, n-gram, and membership inference matchers.
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
    from src.similarity.ensemble_matcher import EnsembleConfig
    from src.detection.models import BenchmarkType
except ImportError:
    # Fallback to direct imports
    sys.path.insert(0, os.path.join(src_path, 'similarity'))
    sys.path.insert(0, os.path.join(src_path, 'detection'))
    from matcher_factory import create_matcher
    from base_matcher import MatcherType
    from ensemble_matcher import EnsembleConfig
    from models import BenchmarkType

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_candidates() -> List[Dict[str, Any]]:
    """Create test benchmark candidates."""
    return [
        {
            'benchmark_name': 'MMLU',
            'question': 'What is the capital of France?',
            'answer': 'Paris',
            'category': 'geography',
            'benchmark_type': BenchmarkType.LANGUAGE_UNDERSTANDING
        },
        {
            'benchmark_name': 'GSM8K',
            'question': 'If John has 5 apples and gives away 2, how many does he have left?',
            'answer': '3',
            'category': 'math',
            'benchmark_type': BenchmarkType.MATHEMATICAL_REASONING
        },
        {
            'benchmark_name': 'AI_Ethics',
            'question': 'What are the key principles of AI ethics?',
            'answer': 'Fairness, accountability, transparency, and human autonomy',
            'category': 'ethics',
            'benchmark_type': BenchmarkType.SAFETY_EVALUATION
        },
        {
            'benchmark_name': 'Text_Samples',
            'question': 'How do I bake a chocolate cake?',
            'answer': 'Mix ingredients, bake at 350°F for 30 minutes',
            'category': 'cooking',
            'benchmark_type': BenchmarkType.COMMON_SENSE
        }
    ]

def test_ensemble_matcher():
    """Test the ensemble matcher functionality."""
    logger.info("============================================================")
    logger.info("Testing Ensemble Matcher")
    logger.info("============================================================")
    
    try:
        # Create ensemble matcher with custom configuration
        ensemble_config = {
            'semantic_weight': 0.4,
            'llm_weight': 0.3,
            'ngram_weight': 0.2,
            'membership_inference_weight': 0.1,
            'high_confidence_threshold': 0.8,
            'medium_confidence_threshold': 0.6,
            'combination_method': 'weighted_average'
        }
        
        matcher = create_matcher(MatcherType.ENSEMBLE, ensemble_config)
        logger.info(f"Created matcher: {matcher.__class__.__name__}")
        
        # Test candidates
        candidates = create_test_candidates()
        
        # Test queries
        test_queries = [
            "What is the capital of France?",
            "If I have 10 cookies and eat 3, how many are left?",
            "How do I bake a chocolate cake?",
            "What are the ethical considerations in AI development?"
        ]
        
        for query in test_queries:
            logger.info(f"\nTesting query: '{query}'")
            
            matches = matcher.find_matches(
                query=query,
                candidates=candidates,
                max_matches=5
            )
            
            logger.info(f"Found {len(matches)} ensemble matches:")
            for i, match in enumerate(matches, 1):
                logger.info(f"  {i}. {match.benchmark_name} (score: {match.score:.3f})")
                logger.info(f"     Text: {match.text[:80]}...")
                logger.info(f"     Method: {match.method}")
                logger.info(f"     Confidence: {match.confidence_level}")
                logger.info(f"     Agreement: {match.agreement_score:.3f}")
                logger.info(f"     Individual scores: {match.individual_scores}")
                logger.info(f"     Method contributions: {match.method_contributions}")
        
        # Test compute_similarity method
        logger.info("\nTesting compute_similarity method:")
        text1 = "What is the capital of France?"
        text2 = "Paris is the capital of France."
        similarity = matcher.compute_similarity(text1, text2)
        logger.info(f"Ensemble similarity between texts: {similarity:.3f}")
        
        # Test matcher info
        logger.info("\nTesting get_info method:")
        info = matcher.get_info()
        logger.info(f"Matcher info: {info}")
        
        # Clean up
        matcher.cleanup()
        logger.info("Ensemble matcher test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Ensemble matcher test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ensemble_configurations():
    """Test different ensemble configurations."""
    logger.info("============================================================")
    logger.info("Testing Different Ensemble Configurations")
    logger.info("============================================================")
    
    configurations = [
        {
            'name': 'Balanced',
            'config': {
                'semantic_weight': 0.25,
                'llm_weight': 0.25,
                'ngram_weight': 0.25,
                'membership_inference_weight': 0.25,
                'combination_method': 'weighted_average'
            }
        },
        {
            'name': 'Semantic-Heavy',
            'config': {
                'semantic_weight': 0.5,
                'llm_weight': 0.2,
                'ngram_weight': 0.2,
                'membership_inference_weight': 0.1,
                'combination_method': 'weighted_average'
            }
        },
        {
            'name': 'Voting Method',
            'config': {
                'semantic_weight': 0.25,
                'llm_weight': 0.25,
                'ngram_weight': 0.25,
                'membership_inference_weight': 0.25,
                'combination_method': 'voting'
            }
        },
        {
            'name': 'Max Method',
            'config': {
                'semantic_weight': 0.25,
                'llm_weight': 0.25,
                'ngram_weight': 0.25,
                'membership_inference_weight': 0.25,
                'combination_method': 'max'
            }
        }
    ]
    
    candidates = create_test_candidates()
    test_query = "What is the capital of France?"
    
    results = {}
    
    for config_info in configurations:
        config_name = config_info['name']
        config = config_info['config']
        
        logger.info(f"\nTesting {config_name} configuration...")
        
        try:
            matcher = create_matcher(MatcherType.ENSEMBLE, config)
            matches = matcher.find_matches(test_query, candidates, max_matches=3)
            
            results[config_name] = {
                'matches': len(matches),
                'top_score': matches[0].score if matches else 0.0,
                'top_match': matches[0].benchmark_name if matches else 'None'
            }
            
            logger.info(f"  Results: {len(matches)} matches, top: {matches[0].benchmark_name if matches else 'None'} ({matches[0].score:.3f})" if matches else "  No matches found")
            
            matcher.cleanup()
            
        except Exception as e:
            logger.error(f"  Failed: {e}")
            results[config_name] = {'error': str(e)}
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("CONFIGURATION COMPARISON SUMMARY")
    logger.info("="*60)
    for config_name, result in results.items():
        if 'error' in result:
            logger.info(f"{config_name}: ERROR - {result['error']}")
        else:
            logger.info(f"{config_name}: {result['matches']} matches, top: {result['top_match']} ({result['top_score']:.3f})")
    
    return True

def main():
    """Main test function."""
    logger.info("Starting Ensemble Matcher Tests")
    logger.info("=" * 80)
    
    test_results = {
        'ensemble_matcher': False,
        'ensemble_configurations': False
    }
    
    # Test ensemble matcher
    test_results['ensemble_matcher'] = test_ensemble_matcher()
    
    # Test different configurations
    test_results['ensemble_configurations'] = test_ensemble_configurations()
    
    # Summary
    logger.info("=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Ensemble Matcher: {'PASSED' if test_results['ensemble_matcher'] else 'FAILED'}")
    logger.info(f"Ensemble Configurations: {'PASSED' if test_results['ensemble_configurations'] else 'FAILED'}")
    
    if all(test_results.values()):
        logger.info("\n🎉 All ensemble matcher tests PASSED!")
        return 0
    else:
        logger.error("\n❌ Some ensemble matcher tests FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
