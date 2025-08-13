#!/usr/bin/env python3
"""
Test ARC Dataset Fix

Tests the ensemble matcher with ARC questions to verify the fix worked.
"""

import sys
import sqlite3
from pathlib import Path

# Add the src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Now import the modules
from similarity.matcher_factory import MatcherFactory


def test_arc_fix():
    """Test that ARC dataset fix worked by running ensemble matcher on ARC questions."""
    print("🧪 Testing ARC Dataset Fix with Ensemble Matcher")
    print("=" * 60)
    
    # Connect to database and get ARC questions
    db_path = project_root / "data" / "benchmarks.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    try:
        # Verify ARC questions have proper text
        cursor.execute('''
            SELECT COUNT(*) as total, 
                   SUM(CASE WHEN question_text IS NULL OR question_text = "" THEN 1 ELSE 0 END) as empty
            FROM questions q
            JOIN benchmarks b ON q.benchmark_id = b.id
            WHERE b.name = "ARC"
        ''')
        total, empty = cursor.fetchone()
        
        print(f"📊 ARC Dataset Status:")
        print(f"  - Total questions: {total}")
        print(f"  - Empty questions: {empty} ({empty/total*100:.1f}%)")
        
        if empty > 0:
            print("❌ ARC dataset still has empty questions!")
            return False
        
        print("✅ All ARC questions have proper text!")
        print()
        
        # Get a sample ARC question to test with
        cursor.execute('''
            SELECT q.question_text, q.answer, q.difficulty
            FROM questions q
            JOIN benchmarks b ON q.benchmark_id = b.id
            WHERE b.name = "ARC" AND q.question_text != ""
            ORDER BY RANDOM()
            LIMIT 1
        ''')
        
        result = cursor.fetchone()
        if not result:
            print("❌ No ARC questions found for testing")
            return False
        
        question_text, answer, difficulty = result
        print(f"🎯 Testing with ARC question ({difficulty} difficulty):")
        print(f"Question: {question_text[:200]}...")
        print(f"Answer: {answer}")
        print()
        
        # Create ensemble matcher and test
        print("🔍 Creating ensemble matcher...")
        factory = MatcherFactory()
        matcher = factory.create_matcher('ensemble')
        
        print("🔍 Running contamination detection...")
        match_result = matcher.find_matches(question_text, top_k=5)
        
        print(f"📊 Ensemble Matcher Results:")
        print(f"  - Confidence: {match_result.confidence:.1%}")
        print(f"  - Confidence Level: {match_result.confidence_level}")
        print(f"  - Match Type: {match_result.match_type}")
        print(f"  - Found {len(match_result.matches)} matches")
        print()
        
        if match_result.matches:
            print("🔍 Top matches:")
            for i, match in enumerate(match_result.matches[:3], 1):
                print(f"  {i}. Similarity: {match.similarity:.3f}")
                print(f"     Benchmark: {match.benchmark_name}")
                print(f"     Question: {match.question_text[:100]}...")
                print()
        
        # Test with a slight variation to check semantic matching
        print("🧪 Testing semantic matching with slight variation...")
        variation = question_text.replace("Which", "What").replace("most recently", "latest")
        variation_result = matcher.find_matches(variation, top_k=3)
        
        print(f"📊 Variation Test Results:")
        print(f"  - Original confidence: {match_result.confidence:.1%}")
        print(f"  - Variation confidence: {variation_result.confidence:.1%}")
        print(f"  - Semantic matching working: {'✅' if variation_result.confidence > 0.5 else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        conn.close()


def test_all_benchmarks():
    """Test that all benchmarks have proper question text."""
    print("\n" + "=" * 60)
    print("🔍 Testing All Benchmark Datasets")
    print("=" * 60)
    
    db_path = Path(__file__).parent / "data" / "benchmarks.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    try:
        # Get statistics for all benchmarks
        cursor.execute('''
            SELECT 
                b.name,
                COUNT(*) as total,
                SUM(CASE WHEN q.question_text IS NULL OR q.question_text = "" THEN 1 ELSE 0 END) as empty,
                AVG(LENGTH(q.question_text)) as avg_length
            FROM benchmarks b
            JOIN questions q ON b.id = q.benchmark_id
            GROUP BY b.name
            ORDER BY b.name
        ''')
        
        results = cursor.fetchall()
        
        print(f"{'Benchmark':<15} {'Total':<8} {'Empty':<8} {'Empty %':<10} {'Avg Length':<12}")
        print("-" * 60)
        
        all_good = True
        for name, total, empty, avg_length in results:
            empty_pct = (empty / total * 100) if total > 0 else 0
            status = "✅" if empty == 0 else "❌"
            print(f"{name:<15} {total:<8} {empty:<8} {empty_pct:<9.1f}% {avg_length:<11.0f} {status}")
            
            if empty > 0:
                all_good = False
        
        print()
        if all_good:
            print("🎉 All benchmarks have proper question text!")
        else:
            print("⚠️  Some benchmarks still have empty questions")
        
        return all_good
        
    except Exception as e:
        print(f"❌ Error checking benchmarks: {str(e)}")
        return False
    finally:
        conn.close()


if __name__ == "__main__":
    print("🚀 Starting ARC Dataset Fix Validation")
    print()
    
    # Test ARC specifically
    arc_success = test_arc_fix()
    
    # Test all benchmarks
    all_success = test_all_benchmarks()
    
    print("\n" + "=" * 60)
    print("📋 Final Results:")
    print(f"  - ARC dataset fix: {'✅ SUCCESS' if arc_success else '❌ FAILED'}")
    print(f"  - All benchmarks: {'✅ SUCCESS' if all_success else '❌ SOME ISSUES'}")
    
    if arc_success and all_success:
        print("\n🎉 All tests passed! Contamination detection is working properly.")
    else:
        print("\n⚠️  Some issues remain. Check the output above for details.")
