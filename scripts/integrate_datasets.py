#!/usr/bin/env python3
"""
Dataset Integration Script for IsItBenchmark

This script downloads and integrates real benchmark datasets to replace
the mock data with actual benchmark questions from popular datasets.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmarks.database import BenchmarkDatabase
from benchmarks.dataset_downloader import DatasetDownloader


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('dataset_integration.log')
        ]
    )


def main():
    """Main integration function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting IsItBenchmark Dataset Integration")
    
    # Initialize database and downloader
    db_path = Path(__file__).parent.parent / "data" / "benchmarks.db"
    database = BenchmarkDatabase(str(db_path))
    downloader = DatasetDownloader(database)
    
    # Check if datasets library is available
    try:
        import datasets
        logger.info("✅ HuggingFace datasets library available")
    except ImportError:
        logger.warning("⚠️  HuggingFace datasets library not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
        logger.info("✅ HuggingFace datasets library installed")
    
    # Clear existing mock data
    logger.info("🗑️  Clearing existing mock data...")
    database.clear_all_data()
    
    # Download all popular benchmarks
    logger.info("📥 Downloading popular benchmark datasets...")
    results = downloader.download_all_popular_benchmarks()
    
    # Report results
    logger.info("\n" + "="*50)
    logger.info("📊 DATASET INTEGRATION RESULTS")
    logger.info("="*50)
    
    successful = []
    failed = []
    
    for benchmark, success in results.items():
        if success:
            successful.append(benchmark)
            logger.info(f"✅ {benchmark}")
        else:
            failed.append(benchmark)
            logger.info(f"❌ {benchmark}")
    
    logger.info(f"\n📈 Summary: {len(successful)} successful, {len(failed)} failed")
    
    if successful:
        logger.info(f"✅ Successfully integrated: {', '.join(successful)}")
    
    if failed:
        logger.info(f"❌ Failed to integrate: {', '.join(failed)}")
    
    # Get database statistics
    stats = database.get_statistics()
    logger.info(f"\n📊 Database Statistics:")
    logger.info(f"   Total benchmarks: {stats['total_benchmarks']}")
    logger.info(f"   Total questions: {stats['total_questions']}")
    
    if stats['total_questions'] > 0:
        logger.info("\n🎉 Dataset integration completed successfully!")
        logger.info("   IsItBenchmark now has real benchmark data for detection.")
    else:
        logger.error("\n💥 Dataset integration failed!")
        logger.error("   No questions were successfully loaded.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
