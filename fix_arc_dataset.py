#!/usr/bin/env python3
"""
Fix ARC Dataset Script

Re-downloads the ARC dataset with the corrected question_text field mapping.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from benchmarks.database import BenchmarkDatabase
from benchmarks.dataset_downloader import DatasetDownloader


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Re-download ARC dataset with fixed question_text mapping."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🔧 Fixing ARC Dataset - Re-downloading with correct question_text mapping")
    
    # Initialize database and downloader
    db_path = Path(__file__).parent / "data" / "benchmarks.db"
    database = BenchmarkDatabase(str(db_path))
    downloader = DatasetDownloader(database)
    
    # Re-download ARC dataset
    logger.info("📥 Re-downloading ARC dataset...")
    success = downloader.download_arc()
    
    if success:
        logger.info("✅ ARC dataset fixed successfully!")
        
        # Verify the fix worked
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check ARC questions now have proper text
        cursor.execute('''
            SELECT COUNT(*) as total, 
                   SUM(CASE WHEN question_text IS NULL OR question_text = "" THEN 1 ELSE 0 END) as empty
            FROM questions q
            JOIN benchmarks b ON q.benchmark_id = b.id
            WHERE b.name = "ARC"
        ''')
        total, empty = cursor.fetchone()
        
        logger.info(f"📊 ARC Statistics: {total} total questions, {empty} empty ({empty/total*100:.1f}%)")
        
        if empty == 0:
            logger.info("🎉 SUCCESS: All ARC questions now have proper question text!")
        else:
            logger.warning(f"⚠️  Still have {empty} empty questions")
            
        # Show sample questions
        cursor.execute('''
            SELECT question_text, answer
            FROM questions q
            JOIN benchmarks b ON q.benchmark_id = b.id
            WHERE b.name = "ARC" AND question_text != ""
            LIMIT 3
        ''')
        
        samples = cursor.fetchall()
        logger.info("📝 Sample ARC questions:")
        for i, (question, answer) in enumerate(samples, 1):
            logger.info(f"  {i}. {question[:100]}... (Answer: {answer})")
        
        conn.close()
    else:
        logger.error("❌ ARC dataset re-download failed")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
