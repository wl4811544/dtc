# 异常感知知识追踪项目结构

## 项目目录结构

```
anomaly_aware_kt/
├── README.md                          # 项目说明文档
├── requirements.txt                   # 依赖包列表
├── setup.py                          # 安装配置
│
├── anomaly_kt/                       # 主要代码包
│   ├── __init__.py
│   ├── generator.py                  # 异常数据生成器
│   ├── detector.py                   # 因果异常检测器
│   ├── model.py                      # 异常感知的DTransformer
│   ├── trainer.py                    # 训练器
│   ├── evaluator.py                  # 评估器
│   └── utils.py                      # 工具函数
│
├── scripts/                          # 可执行脚本
│   ├── train_detector.py             # 训练异常检测器
│   ├── train_kt_model.py             # 训练知识追踪模型
│   ├── evaluate.py                   # 评估脚本
│   └── full_pipeline.py              # 完整流程脚本
│
├── configs/                          # 配置文件
│   ├── detector_config.yaml          # 检测器配置
│   ├── kt_model_config.yaml          # KT模型配置
│   └── training_config.yaml          # 训练配置
│
├── notebooks/                        # Jupyter笔记本
│   ├── data_analysis.ipynb           # 数据分析
│   ├── anomaly_visualization.ipynb   # 异常可视化
│   └── results_analysis.ipynb        # 结果分析
│
├── tests/                            # 测试代码
│   ├── test_causality.py             # 因果性测试
│   ├── test_detector.py              # 检测器测试
│   └── test_integration.py           # 集成测试
│
└── output/                           # 输出目录（自动创建）
    ├── models/                       # 保存的模型
    ├── logs/                         # 训练日志
    └── results/                      # 评估结果
```

## 核心模块说明

### 1. **anomaly_kt/generator.py** - 异常数据生成器
- 多种异常生成策略（连续翻转、模式异常、随机突发、难度异常）
- 灵活的异常比例控制

### 2. **anomaly_kt/detector.py** - 因果异常检测器
- 严格遵守时序因果性（只看历史，不看未来）
- 因果自注意力机制
- 历史窗口统计特征

### 3. **anomaly_kt/model.py** - 异常感知的DTransformer
- 基于原始DTransformer的扩展
- 集成异常检测结果
- 动态调整知识状态更新

### 4. **anomaly_kt/trainer.py** - 高级训练器
- 支持多指标优化（F1、AUC、Recall、Precision）
- 自动类别平衡
- 早停和学习率调度

### 5. **anomaly_kt/evaluator.py** - 综合评估器
- 基础指标：检出率、检出准确率
- 高级指标：AUC-ROC、分布分离度
- 可视化功能

## 使用流程

### 第一步：训练异常检测器
```bash
python scripts/train_detector.py \
    --dataset assist17 \
    --epochs 30 \
    --optimize_for f1_score \
    --anomaly_ratio 0.1
```

### 第二步：训练异常感知的知识追踪模型
```bash
python scripts/train_kt_model.py \
    --dataset assist17 \
    --detector_path output/models/best_detector.pt \
    --epochs 100 \
    --anomaly_weight 0.5
```

### 第三步：评估性能提升
```bash
python scripts/evaluate.py \
    --dataset assist17 \
    --baseline_model path/to/baseline.pt \
    --anomaly_model output/models/best_kt_model.pt
```

### 一键运行完整流程
```bash
python scripts/full_pipeline.py --dataset assist17 --config configs/training_config.yaml
```

## 配置文件示例

**configs/training_config.yaml:**
```yaml
# 数据配置
dataset: assist17
data_dir: data
with_pid: true

# 异常检测器配置
detector:
  d_model: 128
  n_heads: 8
  n_layers: 2
  dropout: 0.1
  window_size: 10
  
# 训练配置
training:
  batch_size: 32
  epochs: 30
  learning_rate: 0.001
  anomaly_ratio: 0.1
  optimize_for: f1_score
  
# 知识追踪模型配置
kt_model:
  n_know: 16
  n_layers: 3
  anomaly_weight: 0.5
```

## 关键特性

1. **因果性保证**：所有模型严格遵守时序因果性
2. **模块化设计**：各组件独立，易于扩展
3. **完整的评估体系**：多维度评估指标
4. **可视化支持**：训练曲线、分数分布、ROC曲线
5. **配置驱动**：通过配置文件控制实验

## 安装和依赖

```bash
# 克隆原始DTransformer项目
git clone https://github.com/yxonic/DTransformer.git

# 安装依赖
pip install -r requirements.txt

# 安装本项目
pip install -e .
```

## 预期结果

- 异常检测器AUC-ROC > 0.85
- 知识追踪准确率提升 ~1%
- 减少异常数据对知识状态估计的干扰