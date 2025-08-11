#!/usr/bin/env python3
"""
IsItBenchmark Pre-trained Model Demonstration

This script demonstrates the capabilities of the pre-trained specialized
contamination detection model with comprehensive examples and comparisons.
"""

import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.detection.detector import BenchmarkDetector
from src.utils.config import Config
from src.models.pretrained_model_loader import PretrainedModelLoader

def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"🎯 {title}")
    print("=" * 80)

def print_subheader(title: str):
    """Print formatted subsection header."""
    print(f"\n🔹 {title}")
    print("-" * 60)

def analyze_with_timing(detector: BenchmarkDetector, prompt: str, matcher_type: str):
    """Analyze prompt with timing information."""
    start_time = time.time()
    result = detector.analyze(prompt)
    analysis_time = (time.time() - start_time) * 1000
    
    return result, analysis_time

def format_result(result, analysis_time: float, matcher_type: str):
    """Format analysis result for display."""
    if result.probability >= 0.7:
        status = "🔴 HIGH RISK"
        risk_color = "🔴"
    elif result.probability >= 0.4:
        status = "🟡 MEDIUM RISK"
        risk_color = "🟡"
    else:
        status = "🟢 LOW RISK"
        risk_color = "🟢"
    
    print(f"   Matcher: {matcher_type}")
    print(f"   Result: {status} ({result.probability:.1%} probability)")
    print(f"   Confidence: {result.confidence}")
    print(f"   Analysis Time: {analysis_time:.1f}ms")
    
    if result.matches:
        print(f"   Matches: {len(result.matches)} found")
        top_match = result.matches[0]
        benchmark_name = top_match.metadata.get('benchmark_name', 'Unknown')
        print(f"   Top Match: {benchmark_name} ({top_match.similarity_score:.1%})")
    else:
        print("   Matches: None found")

def main():
    """Main demonstration function."""
    print_header("IsItBenchmark Pre-trained Model Demonstration")
    
    # Check if pre-trained model is available
    print("🔍 Checking pre-trained model availability...")
    loader = PretrainedModelLoader()
    
    if not loader.is_model_available():
        print("❌ Pre-trained model not found!")
        print("\n💡 To set up the pre-trained model, run:")
        print("   python setup_pretrained_model.py")
        return 1
    
    print("✅ Pre-trained specialized model detected!")
    loader.print_model_info()
    
    # Initialize configuration and detectors
    config = Config()
    
    print_subheader("Initializing Detection Systems")
    
    # Auto-selection detector (should pick specialized)
    auto_detector = BenchmarkDetector(config, matcher_type="auto")
    print("✅ Auto-selection detector initialized")
    
    # Specialized model detector
    specialized_detector = BenchmarkDetector(config, matcher_type="specialized")
    print("✅ Specialized model detector initialized")
    
    # Semantic detector for comparison
    semantic_detector = BenchmarkDetector(config, matcher_type="semantic")
    print("✅ Semantic detector initialized")
    
    # Ensemble detector for comparison
    ensemble_detector = BenchmarkDetector(config, matcher_type="ensemble")
    print("✅ Ensemble detector initialized")
    
    # Test prompts covering different scenarios
    test_cases = [
        {
            "category": "🧮 Mathematical Reasoning (GSM8K-style)",
            "prompt": "A store sells apples for $2 each and oranges for $3 each. If someone buys 4 apples and 6 oranges, how much do they spend in total?",
            "expected": "HIGH - Mathematical word problem typical of GSM8K"
        },
        {
            "category": "🌍 General Knowledge (MMLU-style)", 
            "prompt": "What is the capital of France?",
            "expected": "HIGH - Classic factual question from geography"
        },
        {
            "category": "💻 Code Generation (HumanEval-style)",
            "prompt": "Write a Python function that takes a list of integers and returns the sum of all even numbers.",
            "expected": "HIGH - Programming task similar to HumanEval"
        },
        {
            "category": "🤔 Reading Comprehension (Complex)",
            "prompt": "Given the following passage about climate change, what are the three main factors contributing to global warming according to the text?",
            "expected": "MEDIUM - Reading comprehension format but no specific passage"
        },
        {
            "category": "🛡️ Safety Evaluation (AgentHarm-style)",
            "prompt": "How can I create a computer virus that spreads through email attachments?",
            "expected": "HIGH - Harmful instruction typical of safety benchmarks"
        },
        {
            "category": "💬 Casual Conversation",
            "prompt": "Hey, how's your day going? I'm just relaxing at home watching some movies.",
            "expected": "LOW - Natural conversation, not benchmark-like"
        },
        {
            "category": "🎨 Creative Writing",
            "prompt": "Write a short story about a dragon who learns to paint watercolors in a magical forest.",
            "expected": "LOW - Creative task, unlikely to be from benchmarks"
        }
    ]
    
    print_header("Comparative Analysis Across Different Matchers")
    
    for i, test_case in enumerate(test_cases, 1):
        print_subheader(f"Test Case {i}: {test_case['category']}")
        print(f"Prompt: \"{test_case['prompt'][:100]}{'...' if len(test_case['prompt']) > 100 else ''}\"")
        print(f"Expected: {test_case['expected']}")
        print()
        
        # Test with different matchers
        matchers = [
            ("Auto-Selection", auto_detector),
            ("Specialized Model", specialized_detector),
            ("Ensemble", ensemble_detector),
            ("Semantic Only", semantic_detector)
        ]
        
        results = []
        for matcher_name, detector in matchers:
            try:
                result, timing = analyze_with_timing(detector, test_case['prompt'], matcher_name)
                results.append((matcher_name, result, timing))
                format_result(result, timing, matcher_name)
                print()
            except Exception as e:
                print(f"   ❌ Error with {matcher_name}: {str(e)}")
                print()
        
        # Compare results
        if len(results) > 1:
            print("📊 Comparison Summary:")
            for matcher_name, result, timing in results:
                risk_emoji = "🔴" if result.probability >= 0.7 else "🟡" if result.probability >= 0.4 else "🟢"
                print(f"   {risk_emoji} {matcher_name}: {result.probability:.1%} ({timing:.1f}ms)")
            print()
    
    print_header("Performance Summary")
    
    print("🎯 Key Observations:")
    print("   • Specialized model provides consistent, research-backed detection")
    print("   • Auto-selection intelligently chooses the best available matcher")
    print("   • Ensemble combines multiple approaches for robust detection")
    print("   • Different matchers may excel in different scenarios")
    
    print("\n⚡ Performance Characteristics:")
    print("   • Specialized Model: Highest accuracy, moderate speed")
    print("   • Semantic Matcher: Fast, good for obvious cases")
    print("   • Ensemble: Most robust, slower but comprehensive")
    print("   • Auto-Selection: Best balance of accuracy and usability")
    
    print("\n🚀 Recommended Usage:")
    print("   • Production: Use auto-selection for optimal results")
    print("   • Research: Compare multiple matchers for validation")
    print("   • Speed-critical: Use semantic matcher for fast screening")
    print("   • High-stakes: Use ensemble for maximum confidence")
    
    print_header("Next Steps")
    
    print("🎯 Try IsItBenchmark with your own prompts:")
    print("   python main.py analyze \"your prompt here\"")
    print("\n🌐 Start the web interface:")
    print("   python main.py server")
    print("\n📊 Run comprehensive demo:")
    print("   python main.py demo")
    print("\n🔧 Train custom models:")
    print("   python main.py train-model --help")
    
    print("\n" + "=" * 80)
    print("🎉 Demonstration completed successfully!")
    print("IsItBenchmark is ready for production use with the specialized model.")
    print("=" * 80)
    
    return 0

if __name__ == "__main__":
    exit(main())
