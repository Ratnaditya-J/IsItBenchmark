# IsItBenchmark 🎯

**Benchmark Contamination Detection for AI/ML Systems**

IsItBenchmark is a powerful tool that analyzes prompts and provides probability scores for whether they originate from known benchmark datasets. This addresses the critical need for data integrity in AI research and development.

## 🚀 Project Overview

### Problem Statement
- **Data Contamination:** LLMs often trained on benchmark datasets, leading to inflated performance scores
- **Research Integrity:** Need to ensure fair evaluation and reproducible results
- **Evaluation Fraud:** Difficulty detecting when models have seen test data during training

### Solution
IsItBenchmark provides:
- **🤖 Pre-trained Specialized Model:** First-of-its-kind contamination detection model trained on 49,159+ benchmark questions
- **⚡ Auto-Selection:** Intelligent matcher selection for optimal accuracy
- **🎯 Probability-based detection** of benchmark contamination
- **🔍 Multi-modal analysis** across text, code, math, and reasoning benchmarks
- **🔄 Fuzzy matching** for paraphrased or modified benchmark questions
- **📊 Comprehensive database** of known benchmarks with temporal tracking

## 🤖 Pre-trained Specialized Model

**World's First Specialized Contamination Detection Model** - IsItBenchmark includes a pre-trained transformer-based model specifically fine-tuned for benchmark contamination detection.

### Key Features
- **🎯 High Accuracy:** Trained on 49,159 questions across 9 benchmark datasets
- **🚀 Out-of-the-box:** Ready to use immediately after setup
- **⚡ Auto-detection:** Automatically selected when available
- **🔬 Research-backed:** Implements 14+ contamination patterns from literature
- **🛡️ Safety Coverage:** Includes trust & safety benchmark detection

### Quick Setup
```bash
# Generate the pre-trained model (one-time setup)
python setup_pretrained_model.py

# Use immediately with auto-selection
python main.py analyze "What is the capital of France?"
```

### Benchmark Coverage
- **Traditional Benchmarks:** MMLU, GSM8K, HellaSwag, ARC, TruthfulQA, HumanEval
- **Safety Benchmarks:** AgentHarm, Aegis Safety, CBRN Safety
- **Total Questions:** 49,159+ across 9 datasets

## 🎯 Key Use Cases

### 🔬 Academic & Research
- **Paper Review Process:** Verify evaluation integrity in publications
- **Dataset Curation:** Clean training data before model training
- **Benchmark Creation:** Ensure new benchmarks don't overlap with existing ones
- **Meta-Analysis:** Study benchmark contamination across published models

### 🏢 Enterprise & Industry
- **Model Evaluation:** Validate models haven't seen test data during training
- **Vendor Assessment:** Evaluate AI vendor performance claims
- **Compliance:** Meet regulatory requirements for fair AI evaluation
- **Internal Auditing:** Ensure proper train/test splits in development

### 🛡️ AI Safety & Governance
- **Model Certification:** Third-party validation of AI system capabilities
- **Regulatory Compliance:** Government verification of AI evaluation claims
- **Standardization:** Industry-wide fair comparison across models
- **Red Team Operations:** Security research for evaluation shortcuts

### 📚 Educational
- **Academic Integrity:** Detect benchmark problems in student assignments
- **Course Design:** Ensure homework doesn't overlap with standard benchmarks
- **Assessment Tools:** Create fair evaluation frameworks for AI courses

## 🔧 Technical Architecture

### Core Components
```
IsItBenchmark/
├── src/
│   ├── detection/          # Core detection algorithms
│   ├── benchmarks/         # Benchmark database management
│   ├── similarity/         # Fuzzy matching and similarity analysis
│   ├── api/               # REST API and web interface
│   └── utils/             # Utilities and helpers
├── data/
│   ├── benchmarks/        # Benchmark datasets
│   ├── embeddings/        # Pre-computed embeddings
│   └── models/            # Trained detection models
├── tests/                 # Comprehensive test suite
└── docs/                  # Documentation and examples
```

### Key Features
- **Multi-Modal Detection:** Text, code, math, reasoning benchmarks
- **Similarity Analysis:** Find near-duplicates and variations
- **Confidence Scoring:** Probability-based output with explanations
- **Temporal Tracking:** When benchmarks were published vs. model training dates
- **Cross-Language Detection:** Translated versions of benchmarks
- **API Integration:** Easy integration into existing ML pipelines

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/Ratnaditya-J/IsItBenchmark.git
cd IsItBenchmark
pip install -r requirements.txt
```

### Quick Start with Pre-trained Model

**🚀 One-time setup (generates pre-trained model):**
```bash
python setup_pretrained_model.py
```

**⚡ Immediate use with auto-selection:**
```bash
# Auto-selects best available matcher (specialized model if available)
python main.py analyze "What is the capital of France?"

# Run demo with multiple test cases
python main.py demo

# Start web interface
python main.py server
```

**🎯 Manual matcher selection:**
```bash
# Use pre-trained specialized model
python main.py analyze "Your prompt" --matcher specialized

# Use ensemble of multiple matchers
python main.py analyze "Your prompt" --matcher ensemble

# Use semantic similarity only
python main.py analyze "Your prompt" --matcher semantic
```

### Python API Usage
```python
from src.detection.detector import BenchmarkDetector
from src.utils.config import Config

# Auto-select best matcher (uses specialized model if available)
config = Config()
detector = BenchmarkDetector(config, matcher_type="auto")
result = detector.analyze("What is the capital of France?")

print(f"Benchmark probability: {result.probability:.2f}")
print(f"Matches found: {len(result.matches)}")
print(f"Confidence: {result.confidence}")
```

### Advanced Model Training 🔬

**Train custom specialized models:**
```bash
# Train with custom parameters
python main.py train-model --num-samples 10000 --epochs 5 --use-wandb

# Train with existing data
python main.py train-model --data-path my_training_data.json
```

### REST API Usage

**Start the server:**
```bash
# Auto-selects best matcher as default
python main.py server --port 8000

# Specify default matcher
python main.py server --default-matcher specialized
```

**Analyze prompts:**
```bash
# Use default matcher (auto-selected)
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?"}'

# Specify matcher type
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the capital of France?",
    "matcher_type": "specialized",
    "scope": "all"
  }'
```

**Response format:**
```json
{
  "probability": 0.85,
  "confidence": "high",
  "matches": [
    {
      "benchmark_name": "MMLU",
      "similarity_score": 0.92,
      "question_id": "geography_001"
    }
  ],
  "analysis_time_ms": 245,
  "matcher_used": "specialized"
}
```

## 📊 Supported Benchmarks

### Current Coverage
- **Language Understanding:** MMLU, HellaSwag, ARC, WinoGrande
- **Mathematical Reasoning:** GSM8K, MATH, DROP
- **Code Generation:** HumanEval, MBPP, CodeContests
- **Common Sense:** CommonsenseQA, PIQA, OpenBookQA
- **Reading Comprehension:** SQuAD, QuAC, CoQA

### Planned Additions
- **Multilingual:** XNLI, XQuAD, MLQA
- **Specialized:** BioASQ, SciFact, LegalBench
- **Multimodal:** VQA, COCO Captions, TextVQA

## 🔒 Privacy & Security

- **Local Processing:** All analysis performed locally by default
- **No Data Storage:** Prompts not stored unless explicitly configured
- **Secure API:** Optional authentication and rate limiting
- **Audit Logging:** Comprehensive logging for compliance requirements

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone and setup development environment
git clone https://github.com/Ratnaditya-J/IsItBenchmark.git
cd IsItBenchmark
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Start development server
python -m src.api.server --dev
```

## 📈 Roadmap

### Phase 1: Core Detection (Current)
- [x] Basic benchmark database
- [x] Text similarity detection
- [x] Probability scoring
- [x] REST API

### Phase 2: Advanced Features
- [ ] Fuzzy matching algorithms
- [ ] Multi-modal detection
- [ ] Temporal analysis
- [ ] Web interface

### Phase 3: Enterprise Features
- [ ] Batch processing
- [ ] Custom benchmark integration
- [ ] Advanced analytics
- [ ] Enterprise authentication

### Phase 4: Ecosystem Integration
- [ ] MLOps platform integrations
- [ ] CI/CD pipeline plugins
- [ ] Academic publishing tools
- [ ] Regulatory compliance features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Benchmark datasets from the AI research community
- Open source ML libraries and tools
- Contributors and early adopters

## 📞 Contact

- **GitHub:** [Ratnaditya-J/IsItBenchmark](https://github.com/Ratnaditya-J/IsItBenchmark)
- **Issues:** [GitHub Issues](https://github.com/Ratnaditya-J/IsItBenchmark/issues)
- **Discussions:** [GitHub Discussions](https://github.com/Ratnaditya-J/IsItBenchmark/discussions)

---

**IsItBenchmark** - Ensuring integrity in AI evaluation, one prompt at a time. 🎯✨
