# Specialized Contamination Detection Model Training

## Overview

IsItBenchmark Phase 2 introduces specialized contamination detection model training using GPT-OSS and other transformer models. This creates the first specialized model for benchmark contamination detection, providing a significant first-mover advantage in this emerging field.

## Key Features

- **First-of-its-Kind**: No existing specialized models for benchmark contamination detection
- **Research-Based**: Implements contamination patterns from academic literature
- **Flexible Training**: Supports various base models and training configurations
- **Comprehensive Evaluation**: Built-in validation and metrics tracking
- **Production Ready**: Optimized for real-world deployment

## Training Command

```bash
python main.py train-model [OPTIONS]
```

### Basic Usage

```bash
# Quick training with default settings
python main.py train-model --num-samples 1000 --epochs 3

# Advanced training with custom configuration
python main.py train-model \
    --model-name "microsoft/DialoGPT-medium" \
    --num-samples 5000 \
    --positive-ratio 0.6 \
    --epochs 5 \
    --batch-size 16 \
    --learning-rate 2e-5 \
    --max-length 512 \
    --output-dir "models/my_contamination_detector"
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model-name` | Base model for fine-tuning | `microsoft/DialoGPT-medium` |
| `--num-samples` | Number of training samples to generate | `1000` |
| `--positive-ratio` | Ratio of contaminated samples | `0.5` |
| `--data-path` | Path to existing training data (optional) | `None` |
| `--output-dir` | Output directory for trained model | `models/specialized_contamination_detector` |
| `--epochs` | Number of training epochs | `3` |
| `--batch-size` | Training batch size | `8` |
| `--learning-rate` | Learning rate | `2e-5` |
| `--max-length` | Maximum sequence length | `512` |
| `--use-wandb` | Enable Weights & Biases tracking | `False` |
| `--wandb-project` | W&B project name | `isitbenchmark-contamination` |
| `--debug` | Enable debug logging | `False` |

## Contamination Patterns

The training system implements 14 research-based contamination patterns:

### Paraphrasing Patterns
- **Direct Paraphrase**: Synonym replacement and rewording
- **Structural Paraphrase**: Sentence structure changes
- **Question Reformulation**: Different question phrasing

### Format Patterns
- **Format Change**: Multiple choice to open-ended conversion
- **Multiple Choice Removal**: Answer option removal
- **Prompt Wrapping**: Additional context wrapping

### Context Patterns
- **Context Addition**: Relevant background information
- **Domain Shift**: Cross-domain knowledge transfer
- **Persona Injection**: Role-based perspective changes

### Noise Patterns
- **Typo Injection**: Realistic typing errors
- **Punctuation Changes**: Punctuation modifications
- **Case Changes**: Capitalization variations
- **Translation Artifacts**: Cross-language artifacts
- **Grammar Shifts**: Grammatical style changes

## Training Data Generation

### Automatic Generation
The system automatically generates training data from benchmark questions:

```python
# Data generation process
1. Load benchmark questions from database
2. Apply contamination patterns to create positive samples
3. Generate clean variations for negative samples
4. Balance dataset according to positive_ratio
5. Split into training and validation sets
```

### Custom Training Data
You can provide your own training data:

```bash
python main.py train-model --data-path "data/my_training_data.json"
```

Expected format:
```json
{
  "texts": ["sample text 1", "sample text 2", ...],
  "labels": [0, 1, 0, 1, ...],
  "metadata": {
    "description": "Custom contamination detection dataset",
    "source": "manual_annotation"
  }
}
```

## Model Architecture

### Base Models Supported
- **GPT-2 Family**: `gpt2`, `gpt2-medium`, `gpt2-large`
- **DialoGPT**: `microsoft/DialoGPT-medium`, `microsoft/DialoGPT-large`
- **DistilGPT-2**: `distilgpt2`
- **Custom Models**: Any HuggingFace transformer model

### Classification Head
- Binary classification (clean vs. contaminated)
- Sequence classification with attention pooling
- Dropout regularization for robustness

## Training Process

### 1. Data Preparation
```
Training Samples: 4000 (80%)
Validation Samples: 1000 (20%)
Positive Samples: 2500 (50%)
Negative Samples: 2500 (50%)
```

### 2. Model Training
```
Optimizer: AdamW
Learning Rate: 2e-5 with warmup
Weight Decay: 0.01
Batch Size: 8-16 (depending on GPU memory)
Max Sequence Length: 512 tokens
```

### 3. Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: Contamination detection precision
- **Recall**: Contamination detection recall
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

## Using Trained Models

### Integration with Detection Pipeline

```python
from src.detection.detector import BenchmarkDetector

# Initialize detector with specialized model
detector = BenchmarkDetector(
    matcher_type="specialized",
    config={
        "model_path": "models/specialized_contamination_detector",
        "threshold": 0.7
    }
)

# Analyze prompt for contamination
result = detector.analyze_prompt("Your prompt here")
print(f"Contamination probability: {result.contamination_score}")
```

### CLI Usage with Specialized Model

```bash
# Use specialized model for detection
python main.py analyze \
    --matcher specialized \
    --model-path models/specialized_contamination_detector \
    --prompt "What is the capital of France?"
```

## Performance Optimization

### Training Optimization
- **Mixed Precision**: Automatic FP16 training on GPU
- **Gradient Accumulation**: Handle large effective batch sizes
- **Learning Rate Scheduling**: Warmup and decay strategies
- **Early Stopping**: Prevent overfitting

### Inference Optimization
- **Model Quantization**: Reduce model size for deployment
- **Batch Processing**: Efficient multi-sample analysis
- **Caching**: Cache embeddings for repeated analysis

## Monitoring and Logging

### Weights & Biases Integration
```bash
# Enable W&B tracking
python main.py train-model --use-wandb --wandb-project "my-project"
```

### Training Logs
- Real-time training progress
- Validation metrics tracking
- Loss curves and learning rate schedules
- Model checkpointing

### Output Structure
```
models/specialized_contamination_detector/
├── pytorch_model.bin          # Trained model weights
├── config.json               # Model configuration
├── tokenizer.json            # Tokenizer configuration
├── training_metadata.json    # Training statistics
└── logs/                     # Training logs
    ├── events.out.tfevents   # TensorBoard logs
    └── training.log          # Text logs
```

## Advanced Usage

### Custom Contamination Patterns

```python
from src.training.data_generator import ContaminationPattern

# Define custom pattern
custom_pattern = ContaminationPattern(
    name="domain_specific_contamination",
    description="Domain-specific contamination pattern",
    severity=0.8,
    category="custom"
)

# Add to generator
generator.contamination_patterns.append(custom_pattern)
```

### Multi-GPU Training

```bash
# Use multiple GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py train-model \
    --num-samples 10000 \
    --batch-size 32 \
    --epochs 5
```

### Hyperparameter Tuning

```bash
# Grid search example
for lr in 1e-5 2e-5 5e-5; do
    for bs in 8 16 32; do
        python main.py train-model \
            --learning-rate $lr \
            --batch-size $bs \
            --output-dir "models/hp_search_lr${lr}_bs${bs}"
    done
done
```

## Research Applications

### Academic Research
- First specialized model for benchmark contamination detection
- Novel contamination pattern implementation
- Comprehensive evaluation framework
- Publication-ready results and methodology

### Industry Applications
- LLM evaluation and validation
- Training data quality assessment
- Benchmark integrity verification
- AI safety and robustness testing

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   ```bash
   # Reduce batch size
   python main.py train-model --batch-size 4
   ```

2. **Slow Training**
   ```bash
   # Enable mixed precision
   python main.py train-model --fp16
   ```

3. **Poor Performance**
   ```bash
   # Increase training data
   python main.py train-model --num-samples 5000
   ```

### Debug Mode
```bash
# Enable detailed logging
python main.py train-model --debug
```

## Future Enhancements

### Planned Features
- Multi-label contamination classification
- Contamination severity scoring
- Cross-lingual contamination detection
- Real-time contamination monitoring
- Automated retraining pipelines

### Research Directions
- Ensemble methods combining multiple detection approaches
- Adversarial training for robustness
- Few-shot contamination detection
- Explainable contamination analysis

## Citation

If you use IsItBenchmark's specialized contamination detection in your research, please cite:

```bibtex
@software{isitbenchmark2025,
  title={IsItBenchmark: Specialized Contamination Detection for LLM Evaluation},
  author={IsItBenchmark Team},
  year={2025},
  url={https://github.com/your-repo/IsItBenchmark}
}
```

## Support

For questions, issues, or contributions:
- GitHub Issues: [Report bugs and feature requests]
- Documentation: [Comprehensive guides and API reference]
- Community: [Join discussions and get help]
