#!/usr/bin/env python3
"""
Setup Pre-trained Specialized Model for IsItBenchmark

This script sets up the pre-trained specialized contamination detection model
for out-of-the-box use with IsItBenchmark. It generates the model using all
integrated benchmark datasets and makes it immediately available.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def main():
    """Main setup function."""
    print("🚀 IsItBenchmark Pre-trained Model Setup")
    print("=" * 60)
    print("Setting up specialized contamination detection model...")
    print("This will create a production-ready model using all 49,159 benchmark questions.")
    print("-" * 60)
    
    # Check if model already exists
    project_root = Path(__file__).parent
    model_dir = project_root / "models" / "pretrained_specialized_detector"
    
    if model_dir.exists() and (model_dir / "pytorch_model.bin").exists():
        print("✅ Pre-trained model already exists!")
        print(f"📁 Location: {model_dir}")
        
        # Check model status
        try:
            sys.path.append(str(project_root / "src"))
            from src.models.pretrained_model_loader import PretrainedModelLoader
            
            loader = PretrainedModelLoader()
            if loader.validate_model_integrity():
                print("✅ Model integrity validated")
                loader.print_model_info()
                print("\n🎯 Ready to use! Try:")
                print("  python main.py analyze \"What is the capital of France?\"")
                return 0
            else:
                print("⚠️  Model integrity check failed, regenerating...")
        except Exception as e:
            print(f"⚠️  Could not validate existing model: {e}")
            print("Regenerating model...")
    
    # Generate the pre-trained model
    print("\n🔧 Generating pre-trained specialized model...")
    print("This may take 30-60 minutes depending on your hardware.")
    print("The model will be trained on:")
    print("  • 49,159 benchmark questions across 9 datasets")
    print("  • 14 contamination patterns from research literature")
    print("  • Comprehensive training data generation")
    
    # Confirm with user
    response = input("\n❓ Continue with model generation? [Y/n]: ")
    if response.lower() in ['n', 'no']:
        print("❌ Setup cancelled.")
        return 1
    
    print("\n🚀 Starting model generation...")
    start_time = time.time()
    
    try:
        # Run the model generation script
        script_path = project_root / "scripts" / "generate_pretrained_model.py"
        result = subprocess.run([
            sys.executable, str(script_path)
        ], capture_output=True, text=True, cwd=str(project_root))
        
        if result.returncode == 0:
            elapsed_time = time.time() - start_time
            print(f"\n🎉 SUCCESS! Pre-trained model generated in {elapsed_time/60:.1f} minutes")
            print("=" * 60)
            
            # Show model info
            try:
                sys.path.append(str(project_root / "src"))
                from src.models.pretrained_model_loader import PretrainedModelLoader
                
                loader = PretrainedModelLoader()
                loader.print_model_info()
                
                print("\n🎯 Ready to use! Try these commands:")
                print("  python main.py analyze \"What is the capital of France?\"")
                print("  python main.py demo")
                print("  python main.py server")
                print("\n💡 The system will automatically use the pre-trained model when available!")
                
            except Exception as e:
                print(f"⚠️  Model generated but could not load info: {e}")
                print("✅ Model should still be functional")
            
            return 0
            
        else:
            print(f"\n❌ Model generation failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return 1
            
    except Exception as e:
        print(f"\n❌ Error during model generation: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
