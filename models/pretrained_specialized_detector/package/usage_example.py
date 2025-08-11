#!/usr/bin/env python3
"""
Example usage of the IsItBenchmark Specialized Contamination Detector
"""

from src.training.specialized_matcher import SpecializedContaminationMatcher

def main():
    # Initialize the specialized matcher
    matcher = SpecializedContaminationMatcher({
        "model_path": "models/pretrained_specialized_detector",
        "threshold": 0.8
    })
    matcher.initialize()
    
    # Test prompts
    test_prompts = [
        "What is the capital of France?",
        "Solve this math problem: 2 + 2 = ?",
        "The following are multiple choice questions (with answers) about machine learning.",
        "Question: Which of the following is NOT a supervised learning algorithm?"
    ]
    
    print("🔍 Testing Contamination Detection:")
    print("=" * 50)
    
    for prompt in test_prompts:
        # Note: In real usage, you'd provide actual benchmark candidates
        # This is just for demonstration
        result = matcher.find_matches(prompt, [])
        
        print(f"Prompt: {prompt[:50]}...")
        print(f"Contamination Score: {result.get('contamination_score', 0):.3f}")
        print(f"Is Contaminated: {'Yes' if result.get('contamination_score', 0) > 0.8 else 'No'}")
        print("-" * 30)

if __name__ == "__main__":
    main()
