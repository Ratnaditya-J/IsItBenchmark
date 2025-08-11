#!/usr/bin/env python3
"""
Weights & Biases Setup Script for IsItBenchmark
Comprehensive setup and validation for W&B integration.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "config"))

try:
    import wandb
    from wandb_config import IsItBenchmarkWandbConfig
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required packages: pip install wandb matplotlib seaborn")
    sys.exit(1)


def check_wandb_installation():
    """Check if W&B is properly installed and configured."""
    print("🔍 Checking W&B installation...")
    
    try:
        import wandb
        print(f"✅ W&B version: {wandb.__version__}")
        return True
    except ImportError:
        print("❌ W&B not installed. Run: pip install wandb")
        return False


def setup_wandb_auth():
    """Setup W&B authentication."""
    print("\n🔐 Setting up W&B authentication...")
    
    # Check if already logged in
    try:
        api = wandb.Api()
        user = api.viewer
        print(f"✅ Already logged in as: {user.username}")
        return True
    except Exception:
        print("⚠️  Not logged in to W&B")
        
        print("\nPlease choose an authentication method:")
        print("1. Login via browser (recommended)")
        print("2. Use API key")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            try:
                wandb.login()
                print("✅ Successfully logged in via browser")
                return True
            except Exception as e:
                print(f"❌ Browser login failed: {e}")
                return False
        
        elif choice == "2":
            api_key = input("Enter your W&B API key: ").strip()
            if api_key:
                try:
                    wandb.login(key=api_key)
                    print("✅ Successfully logged in with API key")
                    return True
                except Exception as e:
                    print(f"❌ API key login failed: {e}")
                    return False
            else:
                print("❌ No API key provided")
                return False
        
        else:
            print("❌ Invalid choice")
            return False


def validate_wandb_config():
    """Validate the W&B configuration."""
    print("\n🧪 Validating W&B configuration...")
    
    try:
        # Test configuration initialization
        config = IsItBenchmarkWandbConfig.IDEAL_CONFIG
        print(f"✅ Configuration loaded with {len(config)} parameters")
        
        # Test key configuration values
        required_keys = [
            "project_phase", "experiment_type", "base_model",
            "learning_rate", "batch_size", "num_epochs"
        ]
        
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            print(f"❌ Missing configuration keys: {missing_keys}")
            return False
        
        print("✅ All required configuration keys present")
        
        # Validate hyperparameter sweep config
        sweep_config_path = project_root / "config" / "wandb_sweep_config.yaml"
        if sweep_config_path.exists():
            print("✅ Hyperparameter sweep configuration found")
        else:
            print("⚠️  Hyperparameter sweep configuration not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


def test_wandb_integration():
    """Test W&B integration with a minimal example."""
    print("\n🧪 Testing W&B integration...")
    
    try:
        # Initialize a test run
        test_run = wandb.init(
            project="isitbenchmark-test",
            name="integration-test",
            config={"test": True},
            mode="offline"  # Don't sync to avoid cluttering the project
        )
        
        # Log some test metrics
        for i in range(5):
            wandb.log({
                "test_metric": i * 0.1,
                "step": i
            })
        
        # Test artifact creation
        artifact = wandb.Artifact("test-artifact", type="test")
        
        # Create a temporary test file
        test_file = "test_artifact.txt"
        with open(test_file, "w") as f:
            f.write("Test artifact content")
        
        artifact.add_file(test_file)
        test_run.log_artifact(artifact)
        
        # Clean up
        os.remove(test_file)
        wandb.finish()
        
        print("✅ W&B integration test successful")
        return True
        
    except Exception as e:
        print(f"❌ W&B integration test failed: {e}")
        return False


def create_project_setup():
    """Create W&B project with proper setup."""
    print("\n🚀 Setting up IsItBenchmark W&B project...")
    
    try:
        # Initialize with comprehensive config
        run = IsItBenchmarkWandbConfig.initialize_wandb(
            run_name="project-setup",
            config_override={"setup": True, "test_run": True},
            tags=["setup", "initialization"]
        )
        
        # Log project information
        project_info = {
            "project_name": "IsItBenchmark",
            "phase": "Phase 2 - Specialized Model Training",
            "description": "First specialized contamination detection model for benchmark analysis",
            "research_focus": "Benchmark contamination detection",
            "innovation": "First-mover advantage in specialized contamination detection",
            "datasets": 9,
            "benchmark_questions": 49159,
            "detection_methods": 5
        }
        
        wandb.log({"project_info": project_info})
        
        # Log research milestone
        IsItBenchmarkWandbConfig.log_research_milestone(
            milestone_name="W&B Integration Complete",
            description="Comprehensive Weights & Biases integration for specialized model training",
            achievements=[
                "Implemented comprehensive W&B configuration",
                "Created hyperparameter sweep configuration", 
                "Integrated contamination-specific metrics and visualizations",
                "Setup model artifact tracking and versioning"
            ],
            metrics={
                "configuration_parameters": len(IsItBenchmarkWandbConfig.IDEAL_CONFIG),
                "tracked_metrics": 25,
                "visualization_types": 4
            },
            next_steps=[
                "Run hyperparameter optimization sweeps",
                "Track model performance across different architectures",
                "Monitor training progress and resource utilization"
            ]
        )
        
        wandb.finish()
        print("✅ Project setup complete")
        return True
        
    except Exception as e:
        print(f"❌ Project setup failed: {e}")
        return False


def generate_usage_examples():
    """Generate usage examples for W&B integration."""
    print("\n📝 Generating usage examples...")
    
    examples = {
        "basic_training": {
            "command": "python main.py train-model --use-wandb --num-samples 1000 --epochs 3",
            "description": "Basic training with W&B logging"
        },
        "hyperparameter_sweep": {
            "command": "wandb sweep config/wandb_sweep_config.yaml",
            "description": "Initialize hyperparameter sweep"
        },
        "sweep_agent": {
            "command": "wandb agent <sweep-id>",
            "description": "Run sweep agent for optimization"
        },
        "custom_config": {
            "command": "python main.py train-model --use-wandb --wandb-project my-project --model-name gpt2-medium",
            "description": "Training with custom W&B project and model"
        }
    }
    
    examples_file = project_root / "docs" / "wandb_usage_examples.json"
    with open(examples_file, "w") as f:
        json.dump(examples, f, indent=2)
    
    print(f"✅ Usage examples saved to: {examples_file}")
    return True


def main():
    """Main setup function."""
    print("="*60)
    print("🎯 IsItBenchmark W&B Setup & Validation")
    print("="*60)
    
    success_steps = []
    
    # Step 1: Check installation
    if check_wandb_installation():
        success_steps.append("installation")
    else:
        print("\n❌ Setup failed at installation check")
        return False
    
    # Step 2: Setup authentication
    if setup_wandb_auth():
        success_steps.append("authentication")
    else:
        print("\n❌ Setup failed at authentication")
        return False
    
    # Step 3: Validate configuration
    if validate_wandb_config():
        success_steps.append("configuration")
    else:
        print("\n❌ Setup failed at configuration validation")
        return False
    
    # Step 4: Test integration
    if test_wandb_integration():
        success_steps.append("integration_test")
    else:
        print("\n❌ Setup failed at integration test")
        return False
    
    # Step 5: Create project setup
    if create_project_setup():
        success_steps.append("project_setup")
    else:
        print("\n❌ Setup failed at project setup")
        return False
    
    # Step 6: Generate examples
    if generate_usage_examples():
        success_steps.append("examples")
    else:
        print("\n⚠️  Failed to generate usage examples (non-critical)")
    
    # Final summary
    print("\n" + "="*60)
    print("🎉 W&B SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"✅ Completed steps: {', '.join(success_steps)}")
    print("\n📋 Next Steps:")
    print("1. Run training with W&B: python main.py train-model --use-wandb")
    print("2. Start hyperparameter sweep: wandb sweep config/wandb_sweep_config.yaml")
    print("3. Monitor experiments at: https://wandb.ai/")
    print("4. View comprehensive metrics and visualizations")
    print("\n🚀 Ready for specialized model training with full W&B integration!")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
