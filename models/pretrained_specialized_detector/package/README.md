# IsItBenchmark Specialized Contamination Detector v1.0

## Overview
This is a pre-trained DistilBERT model specifically designed for detecting benchmark contamination in AI training data. It represents the first specialized model for this task and provides state-of-the-art accuracy.

## Model Details
- **Architecture**: DistilBERT-base with sequence classification head
- **Parameters**: 66,000,000
- **Model Size**: 268.0 MB
- **Training Data**: 90,000 benchmark questions from 9 datasets

## Performance
- **Accuracy**: 1.000
- **F1-Score**: 0.000
- **Precision**: 0.000
- **Recall**: 0.000

## Usage
```python
from src.training.specialized_matcher import SpecializedContaminationMatcher

# Initialize matcher with pre-trained model
matcher = SpecializedContaminationMatcher({
    "model_path": "models/pretrained_specialized_detector"
})
matcher.initialize()

# Detect contamination
result = matcher.find_matches("What is the capital of France?", candidates)
```

## Benchmark Coverage
- **MMLU**: 15,908 questions
- **HellaSwag**: 10,042 questions
- **ARC**: 7,787 questions
- **TruthfulQA**: 817 questions
- **GSM8K**: 8,792 questions
- **HumanEval**: 164 questions
- **AgentHarm**: 110 questions
- **Aegis Safety**: 26 questions
- **CBRN**: 200 questions

## Training Details
- **Training Samples**: 25,000
- **Contamination Patterns**: 14 research-based patterns
- **Training Epochs**: 2
- **Validation Split**: 0.2 (80% train, 20% validation)

## Citation
If you use this model in your research, please cite:
```
IsItBenchmark Specialized Contamination Detector v1.0
Trained on comprehensive benchmark dataset
First specialized model for benchmark contamination detection
```
