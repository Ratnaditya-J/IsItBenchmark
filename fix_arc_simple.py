#!/usr/bin/env python3
"""
Simple ARC Dataset Fix Script

Directly fixes the ARC dataset by re-downloading and updating the database
without relying on complex import structures.
"""

import sqlite3
import json
from pathlib import Path

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    print("❌ HuggingFace datasets library not available")
    DATASETS_AVAILABLE = False
    exit(1)


def fix_arc_dataset():
    """Fix ARC dataset by re-downloading with correct question_text mapping."""
    print("🔧 Fixing ARC Dataset - Re-downloading with correct question_text mapping")
    
    # Database path
    db_path = Path(__file__).parent / "data" / "benchmarks.db"
    
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return False
    
    # Connect to database
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    try:
        # Find ARC benchmark_id
        cursor.execute('SELECT id FROM benchmarks WHERE name = "ARC"')
        result = cursor.fetchone()
        if not result:
            print("❌ ARC benchmark not found in database")
            return False
        
        benchmark_id = result[0]
        print(f"📍 Found ARC benchmark with ID: {benchmark_id}")
        
        # Delete existing ARC questions
        cursor.execute('DELETE FROM questions WHERE benchmark_id = ?', (benchmark_id,))
        deleted_count = cursor.rowcount
        print(f"🗑️  Deleted {deleted_count} existing ARC questions")
        
        # Load ARC dataset (both easy and challenge)
        print("📥 Downloading ARC-Easy dataset...")
        dataset_easy = load_dataset("ai2_arc", "ARC-Easy")
        
        print("📥 Downloading ARC-Challenge dataset...")
        dataset_challenge = load_dataset("ai2_arc", "ARC-Challenge")
        
        questions = []
        
        # Process ARC-Easy
        print("🔄 Processing ARC-Easy questions...")
        for item in dataset_easy["validation"]:
            question_text = item["question"]
            choices = item["choices"]
            correct_answer = item["answerKey"]
            
            # Format as multiple choice question
            formatted_question = f"{question_text}\n\nChoices:"
            for i, choice in enumerate(choices["text"]):
                label = choices["label"][i]
                formatted_question += f"\n{label}. {choice}"
            
            questions.append({
                "question_text": formatted_question,  # ✅ Using correct key
                "answer": correct_answer,
                "category": "science_reasoning",
                "difficulty": "easy",
                "metadata": {
                    "choices": choices,
                    "source": "arc",
                    "subset": "easy",
                    "split": "validation"
                }
            })
        
        # Process ARC-Challenge
        print("🔄 Processing ARC-Challenge questions...")
        for item in dataset_challenge["validation"]:
            question_text = item["question"]
            choices = item["choices"]
            correct_answer = item["answerKey"]
            
            # Format as multiple choice question
            formatted_question = f"{question_text}\n\nChoices:"
            for i, choice in enumerate(choices["text"]):
                label = choices["label"][i]
                formatted_question += f"\n{label}. {choice}"
            
            questions.append({
                "question_text": formatted_question,  # ✅ Using correct key
                "answer": correct_answer,
                "category": "science_reasoning",
                "difficulty": "hard",
                "metadata": {
                    "choices": choices,
                    "source": "arc",
                    "subset": "challenge",
                    "split": "validation"
                }
            })
        
        # Insert questions into database
        print(f"💾 Inserting {len(questions)} ARC questions into database...")
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
        
        # Commit changes
        conn.commit()
        
        # Verify the fix worked
        cursor.execute('''
            SELECT COUNT(*) as total, 
                   SUM(CASE WHEN question_text IS NULL OR question_text = "" THEN 1 ELSE 0 END) as empty
            FROM questions
            WHERE benchmark_id = ?
        ''', (benchmark_id,))
        total, empty = cursor.fetchone()
        
        print(f"📊 ARC Statistics: {total} total questions, {empty} empty ({empty/total*100:.1f}%)")
        
        if empty == 0:
            print("🎉 SUCCESS: All ARC questions now have proper question text!")
            
            # Show sample questions
            cursor.execute('''
                SELECT question_text, answer
                FROM questions
                WHERE benchmark_id = ? AND question_text != ""
                LIMIT 3
            ''', (benchmark_id,))
            
            samples = cursor.fetchall()
            print("📝 Sample ARC questions:")
            for i, (question, answer) in enumerate(samples, 1):
                print(f"  {i}. {question[:100]}... (Answer: {answer})")
            
            return True
        else:
            print(f"⚠️  Still have {empty} empty questions")
            return False
            
    except Exception as e:
        print(f"❌ Error fixing ARC dataset: {str(e)}")
        return False
    finally:
        conn.close()


if __name__ == "__main__":
    success = fix_arc_dataset()
    if success:
        print("✅ ARC dataset fix completed successfully!")
    else:
        print("❌ ARC dataset fix failed!")
