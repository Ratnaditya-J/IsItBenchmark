#!/usr/bin/env python3
"""
IsItBenchmark - Benchmark Contamination Detection Tool

Main entry point for the IsItBenchmark system. Provides both CLI and web
interfaces for detecting benchmark contamination in AI/ML prompts.

Usage:
    python main.py --help                    # Show help
    python main.py analyze "prompt text"     # Analyze a single prompt
    python main.py server                    # Start web server
    python main.py demo                      # Run demonstration
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.detection.detector import BenchmarkDetector
from src.api.server import run_server
from src.utils.config import Config
from src.benchmarks.database import BenchmarkDatabase
from src.benchmarks.user_dataset_manager import UserDatasetManager
from src.training.train_specialized_model import train_specialized_model
from src.models.pretrained_model_loader import PretrainedModelLoader


def setup_logging(debug: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def analyze_prompt(prompt: str, config_path: str = None, debug: bool = False, matcher: str = "auto", scope: str = "all"):
    """
    Analyze a prompt for benchmark contamination.
    
    Args:
        prompt: Text prompt to analyze
        config_path: Path to configuration file
        debug: Enable debug logging
        matcher: Matcher type to use (auto, specialized, semantic, llm, ensemble)
        scope: Benchmark scope (all or safety)
    """
    # Auto-select best available matcher
    if matcher == "auto":
        pretrained_loader = PretrainedModelLoader()
        if pretrained_loader.is_model_available():
            matcher = "specialized"
            print("🤖 Auto-selected: Pre-trained specialized model detected")
        else:
            matcher = "ensemble"
            print("⚡ Auto-selected: Ensemble matcher (specialized model not available)")
    setup_logging(debug)
    
    print("🎯 IsItBenchmark - Benchmark Contamination Detection")
    print("=" * 60)
    print(f"Analyzing: {prompt}")
    print(f"Scope: {'Safety benchmarks only' if scope == 'safety' else 'All benchmarks'}")
    print("-" * 60)
    
    # Load configuration
    config = Config(config_path)
    
    # Configure detector with selected matcher and scope
    detector_config = {
        "matcher_type": matcher,
        "similarity_threshold": config.detection.similarity_threshold,
        "max_matches": config.detection.max_matches,
        "scope": scope,
    }
    
    # Add LLM-specific configuration if needed
    if matcher == "llm":
        detector_config["matcher_config"] = {
            "model_name": "microsoft/DialoGPT-medium",
            "device": "auto",
            "temperature": 0.1,
        }
    
    # Initialize detector
    detector = BenchmarkDetector(detector_config)
    
    # Analyze the prompt
    result = detector.analyze(prompt)
    
    # Display results
    print(f"\n📊 Analysis Result:")
    print(f"   Query: {result.input_prompt}")
    print(f"   Probability: {result.probability:.1%}")
    print(f"   Confidence: {result.confidence}")
    print(f"   Matcher: {matcher.upper()}")
    print(f"Likely Benchmark: {'Yes' if result.is_likely_benchmark else 'No'}")
    print(f"Analysis Time: {result.analysis_time_ms:.1f}ms")
    
    if result.matches:
        print(f"\n🎯 MATCHES FOUND ({len(result.matches)}):")
        for i, match in enumerate(result.matches, 1):
            # Handle both MatchResult (with metadata) and BenchmarkMatch (with direct attributes)
            if hasattr(match, 'metadata'):
                benchmark_name = match.metadata.get('benchmark_name', 'Unknown')
                benchmark_type = match.metadata.get('benchmark_type', 'Unknown')
                source_url = match.metadata.get('source_url')
                text = match.text
            else:
                # BenchmarkMatch object
                benchmark_name = getattr(match, 'benchmark_name', 'Unknown')
                benchmark_type = getattr(match, 'benchmark_type', 'Unknown')
                source_url = getattr(match, 'source_url', None)
                text = getattr(match, 'matched_text', 'No text available')
            
            print(f"\n{i}. {benchmark_name} ({benchmark_type})")
            print(f"   Similarity: {match.similarity_score:.1%}")
            print(f"   Exact Match: {'Yes' if match.exact_match else 'No'}")
            print(f"   Text: \"{text[:100]}{'...' if len(text) > 100 else ''}\"")
            if source_url:
                print(f"   Source: {source_url}")
    else:
        print("\n✅ No benchmark matches found.")
    
    if result.matches:
        print(f"\n🏆 TOP MATCH:")
        top = result.matches[0]  # First match is highest scoring
        # Handle both MatchResult (with metadata) and BenchmarkMatch (with direct attributes)
        if hasattr(top, 'metadata'):
            benchmark_name = top.metadata.get('benchmark_name', 'Unknown')
            benchmark_type = top.metadata.get('benchmark_type', 'Unknown')
        else:
            # BenchmarkMatch object
            benchmark_name = getattr(top, 'benchmark_name', 'Unknown')
            benchmark_type = getattr(top, 'benchmark_type', 'Unknown')
        
        print(f"   Benchmark: {benchmark_name}")
        print(f"   Confidence: {top.similarity_score:.1%}")
        print(f"   Type: {benchmark_type}")


def run_demo(config_path: str = None, debug: bool = False, matcher: str = "semantic", scope: str = "all"):
    """
    Run a demonstration of the benchmark detection system.
    
    Args:
        config_path: Path to configuration file
        debug: Enable debug logging
        matcher: Matcher type to use (semantic or llm)
        scope: Benchmark scope (all or safety)
    """
    setup_logging(debug)
    
    print("🎯 IsItBenchmark - Demo Mode")
    print("=" * 60)
    print(f"Scope: {'Safety benchmarks only' if scope == 'safety' else 'All benchmarks'}")
    print("-" * 60)
    
    # Sample prompts for demonstration
    demo_prompts = [
        # Likely benchmark questions
        "What is the capital of France?",
        "Solve: 2 + 2 = ?",
        "Translate 'Hello, world!' to Spanish.",
        
        # Unlikely benchmark questions
        "What did I have for breakfast today?",
        "Please write a poem about my cat named Whiskers.",
        "How do I configure my specific router model XYZ-123?",
        
        # Borderline cases
        "What is the square root of 16?",
        "Name three programming languages.",
        "What is machine learning?",
    ]
    
    # Load configuration
    config = Config(config_path)
    
    # Configure detector with selected matcher and scope
    detector_config = {
        "matcher_type": matcher,
        "similarity_threshold": config.detection.similarity_threshold,
        "max_matches": config.detection.max_matches,
        "scope": scope,
    }
    
    # Add LLM-specific configuration if needed
    if matcher == "llm":
        detector_config["matcher_config"] = {
            "model_name": "microsoft/DialoGPT-medium",
            "device": "auto",
            "temperature": 0.1,
        }
    
    # Initialize detector
    detector = BenchmarkDetector(detector_config)
    
    print(f"\n🧪 Running demo with {len(demo_prompts)} sample prompts...")
    print("-" * 60)
    
    for i, prompt in enumerate(demo_prompts, 1):
        print(f"\n{i}. Analyzing: \"{prompt}\"")
        
        result = detector.analyze(prompt)
        
        # Color coding based on probability
        if result.probability > 0.7:
            status = "🔴 HIGH RISK"
        elif result.probability > 0.4:
            status = "🟡 MEDIUM RISK"
        else:
            status = "🟢 LOW RISK"
        
        print(f"   Result: {status} ({result.probability:.1%} probability)")
        print(f"   Confidence: {result.confidence}")
        
        if result.matches:
            print(f"   Matches: {len(result.matches)} found")
            top_match = result.matches[0]  # First match is highest scoring
            # Handle both MatchResult (with metadata) and BenchmarkMatch (with direct attributes)
            if hasattr(top_match, 'metadata'):
                benchmark_name = top_match.metadata.get('benchmark_name', 'Unknown')
            else:
                # BenchmarkMatch object
                benchmark_name = getattr(top_match, 'benchmark_name', 'Unknown')
            print(f"   Top Match: {benchmark_name} ({top_match.similarity_score:.1%})")
        else:
            print("   Matches: None found")
    
    print(f"\n✅ Demo completed! Analyzed {len(demo_prompts)} prompts.")
    print("\nTo analyze your own prompts:")
    print("  python main.py analyze \"your prompt here\"")
    print("\nTo start the web interface:")
    print("  python main.py server")


def start_server(host: str = "localhost", port: int = 8000, debug: bool = False, config_path: str = None, default_matcher: str = "semantic"):
    """
    Start the web server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode
        config_path: Path to configuration file
        default_matcher: Default matcher type to use (semantic or llm)
    """
    setup_logging(debug)
    
    print("🎯 IsItBenchmark - Web Server")
    print("=" * 60)
    print(f"Starting server at http://{host}:{port}")
    print("Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        run_server(host=host, port=port, debug=debug, default_matcher=default_matcher)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {str(e)}")
        sys.exit(1)


def add_dataset(file_path: str, name: str = None, description: str = None, debug: bool = False):
    """
    Add a user dataset to the benchmark database.
    
    Args:
        file_path: Path to the dataset file
        name: Custom name for the benchmark (optional)
        description: Custom description (optional)
        debug: Enable debug logging
    """
    setup_logging(debug)
    
    print("📁 IsItBenchmark - Add User Dataset")
    print("=" * 60)
    print(f"Dataset file: {file_path}")
    print("-" * 60)
    
    try:
        # Initialize database and manager
        database = BenchmarkDatabase()
        manager = UserDatasetManager(database)
        
        # Validate dataset first
        print("🔍 Validating dataset...")
        validation = manager.validate_dataset(file_path)
        
        if not validation.is_valid:
            print("❌ Dataset validation failed:")
            for error in validation.errors:
                print(f"   • {error}")
            if validation.warnings:
                print("⚠️  Warnings:")
                for warning in validation.warnings:
                    print(f"   • {warning}")
            sys.exit(1)
        
        print(f"✅ Dataset validation passed!")
        print(f"   Format: {validation.format.value.upper()}")
        print(f"   Questions: {validation.num_questions}")
        
        if validation.warnings:
            print("⚠️  Warnings:")
            for warning in validation.warnings:
                print(f"   • {warning}")
        
        # Show sample questions
        if validation.sample_questions:
            print("\n📋 Sample questions:")
            for i, question in enumerate(validation.sample_questions, 1):
                text = question.get('question_text', 'No text')[:80]
                print(f"   {i}. {text}{'...' if len(text) == 80 else ''}")
        
        # Customize benchmark info if provided
        benchmark_info = validation.suggested_benchmark_info
        if name:
            benchmark_info.name = name
        if description:
            benchmark_info.description = description
        
        print(f"\n📊 Benchmark info:")
        print(f"   Name: {benchmark_info.name}")
        print(f"   Type: {benchmark_info.type.value}")
        print(f"   Description: {benchmark_info.description}")
        print(f"   Domains: {', '.join(benchmark_info.domains)}")
        
        # Confirm addition
        response = input("\n❓ Add this dataset to the benchmark database? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("❌ Dataset addition cancelled.")
            return
        
        # Add dataset
        print("\n📥 Adding dataset to database...")
        success, message, stats = manager.add_user_dataset(
            file_path, benchmark_info, validate_first=False
        )
        
        if success:
            print(f"✅ {message}")
            print(f"   Benchmark: {stats['benchmark_name']}")
            print(f"   Questions: {stats['num_questions']}")
            print(f"   Format: {stats['format']}")
            print(f"   File size: {stats['file_size']} bytes")
        else:
            print(f"❌ {message}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error adding dataset: {str(e)}")
        if debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def list_datasets(debug: bool = False):
    """
    List all datasets in the benchmark database.
    
    Args:
        debug: Enable debug logging
    """
    setup_logging(debug)
    
    print("📚 IsItBenchmark - Dataset List")
    print("=" * 60)
    
    try:
        # Initialize database and manager
        database = BenchmarkDatabase()
        manager = UserDatasetManager(database)
        
        # Get dataset list
        datasets = manager.list_user_datasets()
        
        if not datasets:
            print("📭 No datasets found in the database.")
            print("\n💡 Add datasets using: python main.py add-dataset <file_path>")
            return
        
        print(f"Found {len(datasets)} datasets:\n")
        
        # Display datasets
        for i, dataset in enumerate(datasets, 1):
            print(f"{i}. {dataset['name']}")
            print(f"   Type: {dataset['type']}")
            print(f"   Questions: {dataset['num_questions']}")
            print(f"   Domains: {', '.join(dataset['domains'])}")
            if dataset['description']:
                desc = dataset['description'][:100]
                print(f"   Description: {desc}{'...' if len(dataset['description']) > 100 else ''}")
            if dataset['source_url']:
                print(f"   Source: {dataset['source_url']}")
            print()
        
        # Show summary statistics
        total_questions = sum(d['num_questions'] for d in datasets)
        types = set(d['type'] for d in datasets)
        print(f"📊 Summary:")
        print(f"   Total datasets: {len(datasets)}")
        print(f"   Total questions: {total_questions:,}")
        print(f"   Types: {', '.join(sorted(types))}")
        
    except Exception as e:
        print(f"❌ Error listing datasets: {str(e)}")
        if debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def remove_dataset(name: str, debug: bool = False):
    """
    Remove a dataset from the benchmark database.
    
    Args:
        name: Name of the benchmark to remove
        debug: Enable debug logging
    """
    setup_logging(debug)
    
    print("🗑️  IsItBenchmark - Remove Dataset")
    print("=" * 60)
    print(f"Dataset: {name}")
    print("-" * 60)
    
    try:
        # Initialize database and manager
        database = BenchmarkDatabase()
        manager = UserDatasetManager(database)
        
        # Check if dataset exists
        datasets = manager.list_user_datasets()
        dataset = next((d for d in datasets if d['name'] == name), None)
        
        if not dataset:
            print(f"❌ Dataset '{name}' not found.")
            print("\n📚 Available datasets:")
            for d in datasets:
                print(f"   • {d['name']}")
            sys.exit(1)
        
        # Show dataset info
        print(f"📊 Dataset info:")
        print(f"   Name: {dataset['name']}")
        print(f"   Type: {dataset['type']}")
        print(f"   Questions: {dataset['num_questions']}")
        print(f"   Domains: {', '.join(dataset['domains'])}")
        
        # Confirm removal
        response = input(f"\n❓ Are you sure you want to remove '{name}'? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("❌ Dataset removal cancelled.")
            return
        
        # Remove dataset
        print("\n🗑️  Removing dataset...")
        success, message = manager.remove_dataset(name)
        
        if success:
            print(f"✅ {message}")
        else:
            print(f"❌ {message}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error removing dataset: {str(e)}")
        if debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def validate_dataset(file_path: str, debug: bool = False):
    """
    Validate a dataset file without adding it to the database.
    
    Args:
        file_path: Path to the dataset file
        debug: Enable debug logging
    """
    setup_logging(debug)
    
    print("🔍 IsItBenchmark - Validate Dataset")
    print("=" * 60)
    print(f"Dataset file: {file_path}")
    print("-" * 60)
    
    try:
        # Initialize database and manager
        database = BenchmarkDatabase()
        manager = UserDatasetManager(database)
        
        # Validate dataset
        print("🔍 Validating dataset...")
        validation = manager.validate_dataset(file_path)
        
        # Show results
        if validation.is_valid:
            print("✅ Dataset validation passed!")
        else:
            print("❌ Dataset validation failed!")
        
        print(f"\n📊 Validation results:")
        if validation.format:
            print(f"   Format: {validation.format.value.upper()}")
        print(f"   Questions: {validation.num_questions}")
        print(f"   Valid: {'Yes' if validation.is_valid else 'No'}")
        
        if validation.errors:
            print(f"\n❌ Errors ({len(validation.errors)}):")
            for error in validation.errors:
                print(f"   • {error}")
        
        if validation.warnings:
            print(f"\n⚠️  Warnings ({len(validation.warnings)}):")
            for warning in validation.warnings:
                print(f"   • {warning}")
        
        # Show sample questions
        if validation.sample_questions:
            print(f"\n📋 Sample questions ({len(validation.sample_questions)}):")
            for i, question in enumerate(validation.sample_questions, 1):
                text = question.get('question_text', 'No text')[:100]
                print(f"   {i}. {text}{'...' if len(text) == 100 else ''}")
                if 'choices' in question and question['choices']:
                    choices = question['choices'][:3]  # Show first 3 choices
                    print(f"      Choices: {', '.join(str(c) for c in choices)}{'...' if len(question['choices']) > 3 else ''}")
        
        # Show suggested benchmark info
        if validation.suggested_benchmark_info:
            info = validation.suggested_benchmark_info
            print(f"\n📊 Suggested benchmark info:")
            print(f"   Name: {info.name}")
            print(f"   Type: {info.type.value}")
            print(f"   Description: {info.description}")
            print(f"   Domains: {', '.join(info.domains)}")
            print(f"   Languages: {', '.join(info.languages)}")
        
        if not validation.is_valid:
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error validating dataset: {str(e)}")
        if debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def train_model(
    model_name: str = "microsoft/DialoGPT-medium",
    num_samples: int = 10000,
    positive_ratio: float = 0.5,
    data_path: str = None,
    output_dir: str = "models/specialized_contamination_detector",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    use_wandb: bool = False,
    wandb_project: str = "isitbenchmark-contamination",
    debug: bool = False
):
    """
    Train specialized contamination detection model (Phase 2).
    
    Args:
        model_name: Base model name for fine-tuning
        num_samples: Number of training samples to generate
        positive_ratio: Ratio of positive (contaminated) samples
        data_path: Path to existing training data (optional)
        output_dir: Output directory for trained model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        max_length: Maximum sequence length
        use_wandb: Use Weights & Biases for experiment tracking
        wandb_project: Weights & Biases project name
        debug: Enable debug logging
    """
    setup_logging(debug)
    
    print("🚀 IsItBenchmark Phase 2: Specialized Model Training")
    print("=" * 60)
    print(f"Base Model: {model_name}")
    print(f"Training Samples: {num_samples:,}")
    print(f"Positive Ratio: {positive_ratio:.1%}")
    print(f"Output Directory: {output_dir}")
    print(f"Training Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    if use_wandb:
        print(f"W&B Project: {wandb_project}")
    print("-" * 60)
    
    try:
        # Train the specialized model
        training_metadata = train_specialized_model(
            model_name=model_name,
            num_samples=num_samples,
            positive_ratio=positive_ratio,
            data_path=data_path,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_length=max_length,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            log_level="DEBUG" if debug else "INFO"
        )
        
        print("\n" + "=" * 80)
        print("🎉 PHASE 2 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"✅ Model saved to: {output_dir}")
        print(f"✅ Training accuracy: {training_metadata['evaluation_results']['eval_accuracy']:.3f}")
        print(f"✅ Training F1 score: {training_metadata['evaluation_results']['eval_f1']:.3f}")
        print(f"✅ Model parameters: {training_metadata['model_info']['num_parameters']:,}")
        print(f"✅ Training time: {training_metadata['training_results']['training_time']:.1f}s")
        print("\n🔬 The first specialized contamination detection model is now ready!")
        print("\n📋 Next Steps:")
        print("   1. Test the model: python main.py analyze 'your prompt' --matcher specialized")
        print("   2. Compare with other matchers for accuracy evaluation")
        print("   3. Fine-tune hyperparameters if needed")
        print("   4. Deploy for production use")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        if debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="IsItBenchmark - Benchmark Contamination Detection Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py analyze "What is the capital of France?"
  python main.py server --port 8080
  python main.py demo --debug
  python main.py add-dataset my_benchmark.json --name "My Custom Benchmark"
  python main.py list-datasets
  python main.py validate-dataset my_benchmark.csv
  python main.py remove-dataset "My Custom Benchmark"
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a prompt')
    analyze_parser.add_argument('prompt', help='Prompt text to analyze')
    analyze_parser.add_argument('--config', help='Configuration file path')
    analyze_parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    analyze_parser.add_argument('--matcher', default='auto', type=str, choices=['auto', 'specialized', 'semantic', 'llm', 'ngram', 'membership_inference', 'ensemble'], help='Matcher type to use (auto=best available, specialized=pre-trained model)')
    analyze_parser.add_argument('--scope', default='all', type=str, choices=['all', 'safety'], help='Benchmark scope: all benchmarks or safety benchmarks only')
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Start web server')
    server_parser.add_argument('--host', default='localhost', help='Host to bind to')
    server_parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    server_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    server_parser.add_argument('--config', help='Configuration file path')
    server_parser.add_argument('--default-matcher', default='auto', type=str, choices=['auto', 'specialized', 'semantic', 'llm', 'ngram', 'membership_inference', 'ensemble'], help='Default matcher type to use (auto=best available)')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demonstration')
    demo_parser.add_argument('--config', help='Configuration file path')
    demo_parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    demo_parser.add_argument('--matcher', default='auto', type=str, choices=['auto', 'specialized', 'semantic', 'llm', 'ngram', 'membership_inference', 'ensemble'], help='Matcher type to use (auto=best available)')
    demo_parser.add_argument('--scope', default='all', type=str, choices=['all', 'safety'], help='Benchmark scope: all benchmarks or safety benchmarks only')
    
    # Dataset management commands
    add_dataset_parser = subparsers.add_parser('add-dataset', help='Add a user dataset to the benchmark database')
    add_dataset_parser.add_argument('file_path', help='Path to the dataset file')
    add_dataset_parser.add_argument('--name', help='Custom name for the benchmark')
    add_dataset_parser.add_argument('--description', help='Custom description for the benchmark')
    add_dataset_parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    list_datasets_parser = subparsers.add_parser('list-datasets', help='List all datasets in the benchmark database')
    list_datasets_parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    remove_dataset_parser = subparsers.add_parser('remove-dataset', help='Remove a dataset from the benchmark database')
    remove_dataset_parser.add_argument('name', help='Name of the benchmark to remove')
    remove_dataset_parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    validate_dataset_parser = subparsers.add_parser('validate-dataset', help='Validate a dataset file without adding it')
    validate_dataset_parser.add_argument('file_path', help='Path to the dataset file to validate')
    validate_dataset_parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    # Phase 2: Model training command
    train_parser = subparsers.add_parser('train-model', help='Train specialized contamination detection model (Phase 2)')
    train_parser.add_argument('--model-name', default='microsoft/DialoGPT-medium', help='Base model name for fine-tuning')
    train_parser.add_argument('--num-samples', type=int, default=10000, help='Number of training samples to generate')
    train_parser.add_argument('--positive-ratio', type=float, default=0.5, help='Ratio of positive (contaminated) samples')
    train_parser.add_argument('--data-path', help='Path to existing training data (optional)')
    train_parser.add_argument('--output-dir', default='models/specialized_contamination_detector', help='Output directory for trained model')
    train_parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=8, help='Training batch size')
    train_parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    train_parser.add_argument('--max-length', type=int, default=512, help='Maximum sequence length')
    train_parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases for experiment tracking')
    train_parser.add_argument('--wandb-project', default='isitbenchmark-contamination', help='Weights & Biases project name')
    train_parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'analyze':
            analyze_prompt(args.prompt, args.config, args.debug, args.matcher, args.scope)
        elif args.command == 'server':
            start_server(args.host, args.port, args.debug, args.config, args.default_matcher)
        elif args.command == 'demo':
            run_demo(args.config, args.debug, args.matcher, args.scope)
        elif args.command == 'add-dataset':
            add_dataset(args.file_path, args.name, args.description, args.debug)
        elif args.command == 'list-datasets':
            list_datasets(args.debug)
        elif args.command == 'remove-dataset':
            remove_dataset(args.name, args.debug)
        elif args.command == 'validate-dataset':
            validate_dataset(args.file_path, args.debug)
        elif args.command == 'train-model':
            train_model(
                model_name=args.model_name,
                num_samples=args.num_samples,
                positive_ratio=args.positive_ratio,
                data_path=args.data_path,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                max_length=args.max_length,
                use_wandb=args.use_wandb,
                wandb_project=args.wandb_project,
                debug=args.debug
            )
        else:
            parser.print_help()
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        if args.debug if hasattr(args, 'debug') else False:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
