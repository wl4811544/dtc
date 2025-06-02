# Anomaly-Aware Knowledge Tracing

This project implements an anomaly detection system for knowledge tracing to improve prediction accuracy by identifying and handling abnormal student response patterns.

## Overview

The system consists of three main components:

1. **Anomaly Generator**: Creates synthetic anomalies in student response sequences for training
2. **Causal Anomaly Detector**: Detects abnormal response patterns while respecting temporal causality
3. **Anomaly-Aware DTransformer**: Modified knowledge tracing model that adjusts predictions based on detected anomalies

## Key Features

- **Strict Causality**: All models respect temporal order - only past information is used for predictions
- **Multiple Anomaly Strategies**: Consecutive flips, pattern anomalies, random bursts, difficulty-based anomalies
- **Comprehensive Evaluation**: Multiple metrics including recall, precision, F1, AUC-ROC
- **Modular Design**: Easy to extend and customize components

## Installation

1. Clone the DTransformer repository:
```bash
git clone https://github.com/yxonic/DTransformer.git
cd DTransformer
```

2. Clone this repository into the DTransformer directory:
```bash
git clone <this-repo-url> anomaly_aware_kt
```

3. Install dependencies:
```bash
pip install -r anomaly_aware_kt/requirements.txt
pip install -e .  # Install DTransformer
```

## Quick Start

### One-Command Full Pipeline

Run the complete training and evaluation pipeline:

```bash
python anomaly_aware_kt/scripts/full_pipeline.py \
    --dataset assist17 \
    --with_pid \
    --use_cl \
    --proj \
    --device cuda
```

### Step-by-Step Approach

1. **Train Baseline Model** (optional if you have one):
```bash
python scripts/train.py -m DTransformer -d assist17 -p -cl --proj -o output/baseline
```

2. **Train Anomaly Detector**:
```bash
python anomaly_aware_kt/scripts/train_detector.py \
    --dataset assist17 \
    --epochs 30 \
    --optimize_for f1_score \
    --anomaly_ratio 0.1
```

3. **Train Anomaly-Aware Model**:
```bash
python anomaly_aware_kt/scripts/train_kt_model.py \
    --dataset assist17 \
    --detector_path output/detector/best_model.pt \
    --anomaly_weight 0.5
```

4. **Evaluate Performance**:
```bash
python anomaly_aware_kt/scripts/evaluate.py \
    --dataset assist17 \
    --baseline_model output/baseline/best_model.pt \
    --anomaly_model output/anomaly_aware/best_model.pt
```

## Configuration

Create a YAML configuration file for easier experiment management:

```yaml
# config/experiment.yaml
dataset: assist17
with_pid: true
device: cuda

# Model parameters
d_model: 128
n_heads: 8
n_know: 16
n_layers: 3

# Anomaly detection
detector_d_model: 128
anomaly_ratio: 0.1
anomaly_weight: 0.5
optimize_for: f1_score

# Training
batch_size: 32
learning_rate: 0.001
kt_epochs: 100
detector_epochs: 30
use_cl: true
proj: true
```

Then run with:
```bash
python anomaly_aware_kt/scripts/full_pipeline.py --config config/experiment.yaml
```

## 📚 Documentation

### Core Documentation
- **[异常分类器设计原理详解](docs/anomaly_classifier_design_principles.md)** - 异常分类器的设计原理和与基线模型的关系
- **[因果关系设计决策](docs/causality_debate_and_design_decisions.md)** - 深入讨论因果关系的学术辩论与技术选择
- **[完整系统指南](docs/complete_guide.md)** - 系统使用的详细指南
- **[课程学习研究设计](docs/curriculum_learning_research_design.md)** - 课程学习的理论基础
- **[第一阶段参数说明](docs/stage1_parameters.md)** - 基线模型的参数配置
- **[第四阶段评估指南](docs/stage4_evaluation_guide.md)** - 系统评估方法

### Quick Links
- 🚀 [快速开始](#quick-start) - 立即开始使用
- 🧠 [异常分类器原理](docs/anomaly_classifier_design_principles.md) - 理解核心设计
- 🤔 [因果关系讨论](docs/causality_debate_and_design_decisions.md) - 深入理解设计决策
- 🎓 [课程学习策略](docs/curriculum_learning_research_design.md) - 了解训练策略
- 🔧 [故障排除](#troubleshooting) - 解决常见问题

## Key Concepts

### Anomaly Generation Strategies

1. **Consecutive Flip**: Simulates sustained abnormal behavior (e.g., cheating)
2. **Pattern Anomaly**: Creates unnatural patterns (all correct/wrong answers)
3. **Random Burst**: Short periods of random responses
4. **Difficulty-Based**: Violates normal learning patterns (failing easy questions, passing hard ones)

### Causality Preservation

The anomaly detector uses:
- **Causal attention masks**: Can only attend to previous positions
- **Historical statistics**: Features computed only from past responses
- **One-directional convolutions**: Ensures no future information leakage

### Performance Metrics

- **检出率 (Recall)**: Fraction of anomalies detected
- **检出准确率 (Precision)**: Accuracy of anomaly predictions
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Overall classification performance

## Expected Results

- Anomaly detector AUC-ROC > 0.85
- Knowledge tracing AUC improvement ~1%
- Reduced impact of abnormal responses on knowledge state estimation

## Troubleshooting

### Low Anomaly Detection Performance
- Increase `anomaly_ratio` to generate more training examples
- Try different optimization targets (`recall` for fewer missed anomalies)
- Increase detector model capacity (`detector_d_model`, `detector_n_layers`)

### No Improvement in KT Performance
- Adjust `anomaly_weight` (typically 0.3-0.7 works well)
- Ensure detector is well-trained before training KT model
- Try different anomaly generation strategies

### GPU Memory Issues
- Reduce `batch_size`
- Use gradient accumulation
- Reduce model size parameters

## Project Structure

```
anomaly_aware_kt/
├── anomaly_kt/
│   ├── __init__.py
│   ├── generator.py      # Anomaly generation
│   ├── detector.py       # Causal anomaly detector
│   ├── model.py         # Anomaly-aware DTransformer
│   ├── trainer.py       # Training utilities
│   └── evaluator.py     # Evaluation metrics
├── scripts/
│   ├── full_pipeline.py # Complete training pipeline
│   └── ...             # Other scripts
├── tests/              # Unit tests
├── configs/            # Configuration files
└── README.md          # This file
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{yin2023tracing,
  title={Tracing Knowledge Instead of Patterns: Stable Knowledge Tracing with Diagnostic Transformer},
  author={Yin, Yu and others},
  booktitle={Proceedings of the ACM Web Conference 2023},
  year={2023}
}
```

## License

This project extends DTransformer and follows its license terms.

## Acknowledgments

- Based on the DTransformer implementation
- Inspired by anomaly detection research in educational data mining

---

## 🚀 Example Commands

### Basic Training
```bash
python anomaly_aware_kt/scripts/full_pipeline.py \
    --dataset assist17 \
    --skip_baseline \
    --baseline_path output/baseline/model-048-0.7410.pt \
    --device cuda --with_pid --use_cl --proj \
    --n_know 32 --batch_size 16 --test_batch_size 32
```

### Enhanced Training with Optimized Parameters
```bash
python anomaly_aware_kt/scripts/full_pipeline.py \
    --dataset assist17 \
    --skip_baseline \
    --baseline_path output/baseline/model-048-0.7410.pt \
    --device cuda --with_pid --use_cl --proj \
    --n_know 32 --batch_size 16 --test_batch_size 32 \
    --anomaly_ratio 0.25 --optimize_for recall --detector_lr 0.0005
```

### Using Pre-trained Detector
```bash
python anomaly_aware_kt/scripts/full_pipeline.py \
    --dataset assist17 \
    --skip_baseline \
    --baseline_path output/baseline/model-048-0.7410.pt \
    --skip_detector \
    --detector_path output/detector/best_model.pt \
    --device cuda --with_pid --use_cl --proj \
    --n_know 32 --batch_size 16 --test_batch_size 32 \
    --anomaly_weight 0.5
```
