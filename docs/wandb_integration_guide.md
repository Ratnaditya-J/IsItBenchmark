# Weights & Biases Integration Guide for IsItBenchmark

## Overview

IsItBenchmark now features comprehensive Weights & Biases (W&B) integration for tracking specialized contamination detection model training. This integration provides detailed experiment tracking, hyperparameter optimization, and research milestone documentation.

## 🎯 Ideal W&B Configuration

### Project Configuration
- **Project Name**: `isitbenchmark-contamination-detection`
- **Experiment Type**: `contamination_detection`
- **Model Architecture**: `transformer_classification`
- **Research Focus**: First specialized contamination detection model

### Key Tracking Parameters

#### Model Configuration
```yaml
base_model: "microsoft/DialoGPT-medium"
model_type: "specialized_contamination_detector"
num_labels: 2
max_sequence_length: 512
dropout_rate: 0.1
```

#### Training Hyperparameters
```yaml
learning_rate: 2e-5
batch_size: 16
num_epochs: 5
warmup_steps: 500
weight_decay: 0.01
gradient_accumulation_steps: 1
max_grad_norm: 1.0
```

#### Data Configuration
```yaml
train_samples: 10000
validation_split: 0.2
positive_ratio: 0.5
data_augmentation: true
contamination_patterns: 14
```

#### Research Metrics
```yaml
benchmark_datasets: 9
total_benchmark_questions: 49159
detection_methods: ["semantic", "llm", "ngram", "membership_inference", "specialized"]
research_focus: "first_specialized_contamination_model"
```

## 🚀 Quick Start

### 1. Setup W&B Integration

```bash
# Run the comprehensive setup script
python scripts/setup_wandb.py

# Or manual setup
pip install wandb
wandb login
```

### 2. Basic Training with W&B

```bash
# Train with W&B logging
python main.py train-model \
    --use-wandb \
    --num-samples 10000 \
    --epochs 5 \
    --batch-size 16

# Custom W&B project
python main.py train-model \
    --use-wandb \
    --wandb-project "my-contamination-project" \
    --model-name "gpt2-medium"
```

### 3. Hyperparameter Optimization

```bash
# Initialize sweep
wandb sweep config/wandb_sweep_config.yaml

# Run sweep agent
wandb agent <sweep-id>
```

## 📊 Comprehensive Metrics Tracking

### Training Metrics
- **Loss Tracking**: Train/validation loss per epoch
- **Accuracy Metrics**: Overall accuracy, precision, recall, F1-score
- **Learning Rate**: Dynamic learning rate tracking
- **Resource Usage**: GPU utilization, memory usage

### Contamination-Specific Metrics
- **Contamination Detection**: True/false positives/negatives
- **Sensitivity/Specificity**: Contamination detection rates
- **Probability Analysis**: Distribution of contamination probabilities
- **Pattern Analysis**: Detection of different contamination patterns

### Research Metrics
- **Model Comparison**: Performance across different architectures
- **Benchmark Coverage**: Analysis across 9 benchmark datasets
- **Innovation Tracking**: First-mover advantage metrics
- **Milestone Documentation**: Research achievements and next steps

## 🎨 Advanced Visualizations

### 1. Confusion Matrix
- Heatmap visualization of contamination detection performance
- True/false positive/negative analysis
- Per-class performance breakdown

### 2. Probability Distributions
- Contamination probability histograms by true label
- Scatter plots of probability vs. true labels
- Probability separation analysis

### 3. Contamination Patterns
- Bar charts of detected contamination pattern types
- Pattern frequency analysis
- Pattern-specific performance metrics

### 4. Benchmark Coverage
- Coverage percentage across different benchmark datasets
- Performance comparison by benchmark type
- Dataset-specific contamination rates

## 🔧 Configuration Examples

### Basic Configuration
```python
from config.wandb_config import IsItBenchmarkWandbConfig

# Initialize with default config
run = IsItBenchmarkWandbConfig.initialize_wandb(
    run_name="contamination-detector-basic",
    tags=["basic_training", "phase_2"]
)
```

### Advanced Configuration
```python
# Custom configuration override
config_override = {
    "base_model": "gpt2-large",
    "train_samples": 50000,
    "num_epochs": 10,
    "batch_size": 32,
    "learning_rate": 1e-5
}

run = IsItBenchmarkWandbConfig.initialize_wandb(
    run_name="contamination-detector-advanced",
    config_override=config_override,
    tags=["advanced_training", "large_model", "phase_2"]
)
```

### Research Milestone Logging
```python
IsItBenchmarkWandbConfig.log_research_milestone(
    milestone_name="Specialized Model Training Complete",
    description="Successfully trained first specialized contamination detection model",
    achievements=[
        "Achieved 92% accuracy on contamination detection",
        "Outperformed general-purpose methods by 7%",
        "Implemented 14 contamination patterns from literature"
    ],
    metrics={
        "accuracy": 0.92,
        "f1_score": 0.91,
        "model_parameters": 354825216
    },
    next_steps=[
        "Implement ensemble methods",
        "Expand to additional benchmarks",
        "Optimize for production deployment"
    ]
)
```

## 📈 Hyperparameter Optimization

### Sweep Configuration
The project includes a comprehensive sweep configuration in `config/wandb_sweep_config.yaml`:

```yaml
method: bayes
metric:
  name: val/f1_score
  goal: maximize

parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 1e-6
    max: 1e-3
  
  batch_size:
    values: [8, 16, 32, 64]
  
  num_epochs:
    values: [3, 5, 7, 10]
  
  model_name:
    values: 
      - "microsoft/DialoGPT-medium"
      - "gpt2-medium"
      - "distilgpt2"
```

### Running Sweeps
```bash
# Initialize sweep
wandb sweep config/wandb_sweep_config.yaml

# Run multiple agents in parallel
wandb agent <sweep-id> &
wandb agent <sweep-id> &
wandb agent <sweep-id> &
```

## 🏆 Research Integration

### Academic Tracking
- **Publication Preparation**: Track metrics for research papers
- **Benchmark Creation**: Document new contamination detection benchmarks
- **Innovation Documentation**: First-mover advantage in specialized models
- **Collaboration Metrics**: Track academic partnerships and validations

### Performance Comparisons
- **Baseline Comparisons**: Semantic vs. LLM vs. Specialized methods
- **Architecture Analysis**: Performance across different model sizes
- **Efficiency Metrics**: Speed vs. accuracy trade-offs
- **Scalability Analysis**: Performance with increasing dataset sizes

## 🔍 Monitoring and Alerts

### Real-time Monitoring
- **Training Progress**: Live loss and accuracy tracking
- **Resource Utilization**: GPU/CPU usage monitoring
- **Error Detection**: Automatic failure alerts
- **Performance Degradation**: Early warning systems

### Custom Alerts
```python
# Set up custom alerts for key metrics
wandb.alert(
    title="High Accuracy Achieved",
    text=f"Model achieved {accuracy:.3f} accuracy, exceeding target of 0.90",
    level=wandb.AlertLevel.INFO
)
```

## 📋 Best Practices

### 1. Experiment Organization
- Use consistent naming conventions
- Tag experiments appropriately
- Document experiment objectives
- Track hyperparameter rationale

### 2. Metric Selection
- Focus on contamination-specific metrics
- Track both accuracy and efficiency
- Monitor overfitting indicators
- Compare against baselines

### 3. Artifact Management
- Version control model checkpoints
- Save training configurations
- Document data preprocessing steps
- Track model performance over time

### 4. Collaboration
- Share experiment results with team
- Document findings and insights
- Create reproducible experiments
- Maintain experiment history

## 🚀 Production Integration

### Model Deployment Tracking
```python
# Track model deployment metrics
wandb.log({
    "deployment/latency_ms": inference_time,
    "deployment/throughput_qps": queries_per_second,
    "deployment/accuracy": production_accuracy,
    "deployment/error_rate": error_rate
})
```

### A/B Testing Integration
```python
# Track A/B test results
wandb.log({
    "ab_test/variant": "specialized_model",
    "ab_test/conversion_rate": conversion_rate,
    "ab_test/user_satisfaction": satisfaction_score
})
```

## 🎯 Success Metrics

### Phase 2 Targets
- **Accuracy**: > 90% contamination detection accuracy
- **F1-Score**: > 0.90 for balanced performance
- **Speed**: < 200ms inference time per query
- **Coverage**: Support for all 9 benchmark datasets

### Research Impact
- **Innovation**: First specialized contamination detection model
- **Publications**: Target 2-3 research papers
- **Adoption**: Industry and academic usage
- **Benchmarks**: New evaluation standards

## 📚 Additional Resources

- [W&B Documentation](https://docs.wandb.ai/)
- [Hyperparameter Tuning Guide](https://docs.wandb.ai/guides/sweeps)
- [Model Registry](https://docs.wandb.ai/guides/models)
- [Experiment Tracking Best Practices](https://docs.wandb.ai/guides/track)

## 🆘 Troubleshooting

### Common Issues
1. **Authentication Errors**: Run `wandb login` and verify API key
2. **Slow Logging**: Reduce logging frequency or use offline mode
3. **Memory Issues**: Decrease batch size or use gradient accumulation
4. **Network Issues**: Use offline mode and sync later

### Support
- Check logs in `training.log`
- Review W&B run pages for detailed metrics
- Use `wandb status` to check connection
- Contact support via W&B community forums
