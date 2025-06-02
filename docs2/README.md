# Anomaly-Aware Knowledge Tracing v2

基于DTransformer的异常感知知识追踪项目重构版本

## 📋 项目概述

本项目旨在通过异常检测技术增强知识追踪模型的性能。项目分为四个主要阶段：

1. **Stage 1: 基线模型训练** - 训练标准DTransformer知识追踪模型
2. **Stage 2: 异常分类器训练** - 训练异常检测器识别学习行为异常
3. **Stage 3: 异常感知知识追踪** - 训练融合异常信息的知识追踪模型
4. **Stage 4: 性能评估与对比** - 评估异常感知模型相对于基线的提升

## 🏗️ 项目结构

```
anomaly_kt_v2/
├── __init__.py                 # 项目初始化
├── configs/                    # 配置文件
│   ├── __init__.py            # 配置管理模块
│   └── assist17_baseline.yaml # ASSIST17基线配置
├── core/                      # 核心模块
│   ├── __init__.py           # 核心模块导出
│   └── common.py             # 通用工具函数
├── stages/                   # 训练阶段
│   ├── __init__.py          # 阶段模块导出
│   └── stage1_baseline.py   # 第一阶段：基线训练
└── scripts/                 # 训练脚本
    ├── __init__.py         # 脚本模块
    └── run_stage1_baseline.py # 第一阶段训练脚本
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.9+
- CUDA (推荐)

### 安装依赖

```bash
# 安装基础依赖
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn
pip install pyyaml tomlkit tqdm

# 确保DTransformer项目在同级目录
```

### 重要说明

⚠️ **依赖关系**: 本项目完全基于原始的DTransformer代码，不依赖任何`anomaly_kt`模块。所有训练逻辑都直接使用DTransformer的原生实现。

### 第一阶段：基线模型训练

#### 基础模型训练（推荐开始）

```bash
# 基础模型：d_model=128, n_heads=8, 参数量~1.2M
python scripts/run_stage1_baseline.py \
    --dataset assist17 \
    --model_type basic \
    --auto_config \
    --device cuda
```

#### 扩展模型训练（追求更高性能）

```bash
# 扩展模型：d_model=256, n_heads=16, 参数量~4.8M
python scripts/run_stage1_baseline.py \
    --dataset assist17 \
    --model_type extended \
    --auto_config \
    --device cuda
```

#### 使用自定义配置

```bash
python scripts/run_stage1_baseline.py \
    --dataset assist17 \
    --model_type basic \
    --config configs/assist17_baseline.yaml \
    --device cuda
```

#### 完全自定义参数

```bash
python scripts/run_stage1_baseline.py \
    --dataset assist17 \
    --model_type basic \
    --d_model 256 \
    --n_heads 16 \
    --n_layers 3 \
    --kt_epochs 100 \
    --learning_rate 0.001 \
    --device cuda
```

## 🔄 模型类型对比

### 基础模型 vs 扩展模型

| 特性 | 基础模型 (basic) | 扩展模型 (extended) |
|------|------------------|---------------------|
| **模型维度** | d_model=128, n_heads=8 | d_model=256, n_heads=16 |
| **参数量** | ~1.2M | ~4.8M |
| **训练时间** | 30-45分钟 | 60-90分钟 |
| **GPU内存** | 较少 | 较多 |
| **预期AUC** | 0.74-0.76 | 0.76-0.78 |
| **适用场景** | 快速验证、资源受限 | 追求最佳性能 |

### 选择建议

- **🚀 快速开始**: 使用基础模型验证流程
- **📈 性能对比**: 训练两种模型进行对比
- **💰 资源受限**: 选择基础模型
- **🎯 追求极致**: 选择扩展模型

## 📊 支持的数据集

- **ASSIST09**: 小规模数据集，适合快速实验
- **ASSIST17**: 大规模数据集，推荐用于完整实验
- **Algebra05**: 代数学习数据集
- **Statics**: 静力学数据集

## ⚙️ 配置系统

### 配置文件格式

配置文件使用YAML格式，支持以下参数类别：

```yaml
# 基本参数
dataset: assist17
device: cuda
with_pid: true

# 数据参数
batch_size: 16
test_batch_size: 32

# 模型参数
d_model: 128
n_heads: 8
n_layers: 3
dropout: 0.2

# 训练参数
kt_epochs: 100
learning_rate: 0.001
patience: 10
```

### 参数优先级

1. **命令行参数** (最高优先级)
2. **配置文件参数**
3. **默认值** (最低优先级)

## 📈 训练监控

### 输出目录结构

```
output/stage1_assist17_20240101_120000/
├── config.yaml              # 保存的训练配置
├── baseline/                 # 基线模型目录
│   ├── best_model.pt        # 最佳模型权重
│   ├── training_log.txt     # 训练日志
│   └── metrics.json         # 训练指标
└── plots/                   # 训练曲线图
    ├── loss_curve.png
    └── auc_curve.png
```

### 关键指标

- **AUC**: 知识追踪的主要评估指标
- **ACC**: 预测准确率
- **Loss**: 训练损失
- **Early Stopping**: 基于验证集AUC的早停机制

## 🔧 开发指南

### 添加新数据集

1. 在`data/datasets.toml`中添加数据集配置
2. 创建对应的配置文件`configs/{dataset}_baseline.yaml`
3. 在脚本中添加数据集选项

### 自定义模型参数

修改配置文件或使用命令行参数覆盖默认设置：

```bash
# 增加模型容量
python scripts/run_stage1_baseline.py \
    --dataset assist17 \
    --d_model 512 \
    --n_heads 16 \
    --n_layers 6 \
    --device cuda
```

### 调试模式

```bash
# 使用小批次快速调试
python scripts/run_stage1_baseline.py \
    --dataset assist17 \
    --batch_size 4 \
    --kt_epochs 5 \
    --device cpu
```

## 📝 实验记录

### 推荐的实验流程

1. **基线实验**: 使用默认配置建立性能基线
2. **参数调优**: 系统性调整关键参数
3. **消融研究**: 验证各组件的贡献
4. **最终评估**: 在测试集上评估最佳配置

### 实验配置示例

```yaml
# 保守配置（快速验证）
d_model: 128
n_heads: 8
kt_epochs: 50

# 标准配置（平衡性能）
d_model: 256
n_heads: 16
kt_epochs: 100

# 高性能配置（追求最佳效果）
d_model: 512
n_heads: 16
kt_epochs: 200
```

## 🐛 常见问题

### CUDA内存不足

```bash
# 减少批次大小
--batch_size 8 --test_batch_size 16

# 或使用CPU
--device cpu
```

### 训练收敛慢

```bash
# 调整学习率
--learning_rate 0.005

# 增加模型容量
--d_model 256 --n_heads 16
```

### 配置文件未找到

确保配置文件路径正确，或使用`--auto_config`自动检测。

## 📚 参考文献

1. DTransformer: 原始知识追踪模型
2. Anomaly Detection in Educational Data Mining
3. Contrastive Learning for Knowledge Tracing

## 🤝 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 发起Pull Request

## 📄 许可证

本项目采用MIT许可证。详见LICENSE文件。
