# Stage 1: 基线模型训练详细指南

## 📋 概述

第一阶段的目标是训练一个标准的DTransformer知识追踪模型，作为后续异常感知训练的基线。这个阶段建立了性能基准，为后续阶段的改进提供对比基础。

## 🎯 训练目标

1. **建立性能基线**: 获得标准知识追踪模型的性能指标
2. **验证数据质量**: 确保数据集和预处理流程正确
3. **优化超参数**: 找到适合数据集的最佳模型配置
4. **保存模型权重**: 为第二阶段提供预训练模型

## 🏗️ 模型架构

### DTransformer核心组件

```python
DTransformer(
    n_questions=问题总数,      # 数据集中的问题数量
    n_pid=问题ID总数,          # 问题ID数量（可选）
    d_model=模型维度,          # 隐藏层维度
    n_heads=注意力头数,        # 多头注意力头数
    n_know=知识概念数,         # 知识概念数量
    n_layers=网络层数,         # Transformer层数
    dropout=丢弃率,            # 防止过拟合
    lambda_cl=对比学习权重,    # 对比学习损失权重
    proj=使用投影层,           # 是否使用投影层
    hard_neg=困难负样本,       # 是否使用困难负样本
    window=注意力窗口          # 注意力窗口大小
)
```

### 关键特性

- **因果注意力**: 确保只使用历史信息进行预测
- **位置编码**: 捕获序列中的时间信息
- **多头注意力**: 学习不同类型的依赖关系
- **对比学习**: 增强表征学习能力

## ⚙️ 配置参数详解

### 模型参数

| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `d_model` | 128 | 模型隐藏维度 | 128-512，更大的数据集使用更大的维度 |
| `n_heads` | 8 | 注意力头数 | 必须是d_model的因子 |
| `n_layers` | 3 | Transformer层数 | 3-6层，过深可能过拟合 |
| `dropout` | 0.2 | 丢弃率 | 0.1-0.3，大数据集可以更小 |
| `n_know` | 16 | 知识概念数 | 根据数据集特性调整 |

### 训练参数

| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `kt_epochs` | 100 | 最大训练轮数 | 根据数据集大小调整 |
| `learning_rate` | 0.001 | 学习率 | 0.0005-0.005 |
| `patience` | 10 | 早停耐心值 | 5-15，小数据集用更小值 |
| `batch_size` | 16 | 训练批次大小 | 根据GPU内存调整 |

### 对比学习参数

| 参数 | 默认值 | 说明 | 使用场景 |
|------|--------|------|----------|
| `use_cl` | false | 是否使用对比学习 | 大数据集推荐开启 |
| `lambda_cl` | 0.1 | 对比学习权重 | 0.05-0.2 |
| `proj` | false | 使用投影层 | 对比学习时推荐开启 |
| `hard_neg` | false | 困难负样本 | 高级优化选项 |

## 📊 数据集配置

### ASSIST17配置示例

```yaml
# ASSIST17 基线训练配置
dataset: assist17
data_dir: data
device: cuda
with_pid: true  # ASSIST17有问题ID

# 数据加载参数
batch_size: 16          # 保守起点
test_batch_size: 32     # 测试时可以更大

# 模型架构参数
d_model: 128            # 中等维度
n_heads: 8              # d_model的因子
n_know: 16              # 知识概念数
n_layers: 3             # 3层Transformer
dropout: 0.2            # 适中的dropout

# 训练参数
kt_epochs: 100          # 最大训练轮数
learning_rate: 0.001    # 标准学习率
patience: 10            # 早停耐心值
use_cl: true            # 大数据集使用对比学习
```

### 不同数据集的推荐配置

#### ASSIST09 (小数据集)
```yaml
d_model: 64
n_heads: 4
n_layers: 2
batch_size: 32
use_cl: false
```

#### ASSIST17 (大数据集)
```yaml
d_model: 256
n_heads: 16
n_layers: 4
batch_size: 16
use_cl: true
```

## 🚀 训练流程

### 1. 环境准备

```bash
# 确保在正确的虚拟环境中
source venv/bin/activate

# 检查CUDA可用性
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. 数据验证

```bash
# 检查数据文件是否存在
ls data/assist17/
# 应该看到: train.txt, test.txt, datasets.toml
```

### 3. 配置选择

#### 选项A: 自动配置（推荐）
```bash
python scripts/run_stage1_baseline.py \
    --dataset assist17 \
    --auto_config \
    --device cuda
```

#### 选项B: 自定义配置
```bash
python scripts/run_stage1_baseline.py \
    --dataset assist17 \
    --config configs/my_config.yaml \
    --device cuda
```

#### 选项C: 命令行参数
```bash
python scripts/run_stage1_baseline.py \
    --dataset assist17 \
    --d_model 256 \
    --n_heads 16 \
    --kt_epochs 100 \
    --device cuda
```

### 4. 训练监控

训练过程中会显示：
```
Epoch  10 | Train Loss: 0.4523 | Val AUC: 0.7234 | Val ACC: 0.6891
Epoch  20 | Train Loss: 0.4156 | Val AUC: 0.7456 | Val ACC: 0.7023
...
Early stopping triggered at epoch 45
Best AUC: 0.7623
```

### 5. 结果验证

```bash
# 检查输出目录
ls output/stage1_assist17_20240101_120000/
# 应该看到: config.yaml, baseline/

# 检查模型文件
ls output/stage1_assist17_20240101_120000/baseline/
# 应该看到: best_model.pt, training_log.txt
```

## 📈 性能基准

### ASSIST17预期性能

| 配置 | AUC | ACC | 训练时间 |
|------|-----|-----|----------|
| 保守配置 | 0.74-0.76 | 0.68-0.70 | 30-45分钟 |
| 标准配置 | 0.76-0.78 | 0.70-0.72 | 60-90分钟 |
| 高性能配置 | 0.78-0.80 | 0.72-0.74 | 120-180分钟 |

### ASSIST09预期性能

| 配置 | AUC | ACC | 训练时间 |
|------|-----|-----|----------|
| 标准配置 | 0.82-0.85 | 0.75-0.78 | 10-20分钟 |

## 🔧 调优指南

### 性能不佳的解决方案

#### AUC < 0.70
1. 增加模型容量：`d_model=256, n_heads=16`
2. 延长训练：`kt_epochs=200, patience=15`
3. 调整学习率：`learning_rate=0.0005`

#### 训练不收敛
1. 降低学习率：`learning_rate=0.0005`
2. 增加正则化：`dropout=0.3`
3. 检查数据质量

#### 过拟合现象
1. 增加dropout：`dropout=0.3`
2. 减少模型容量：`d_model=128`
3. 早停：`patience=5`

### 内存优化

#### GPU内存不足
```bash
# 减少批次大小
--batch_size 8 --test_batch_size 16

# 减少模型大小
--d_model 128 --n_layers 2
```

#### CPU训练
```bash
# 使用CPU（较慢但稳定）
--device cpu --batch_size 32
```

## 📊 输出分析

### 训练日志解读

```
Epoch  45 | Train Loss: 0.3892 | Val AUC: 0.7623 | Val ACC: 0.7156
Early stopping triggered at epoch 45
Best AUC: 0.7623 achieved at epoch 35
```

- **Train Loss**: 训练损失，应该逐渐下降
- **Val AUC**: 验证集AUC，主要评估指标
- **Val ACC**: 验证集准确率，辅助指标
- **Early Stopping**: 防止过拟合的机制

### 模型文件说明

- `best_model.pt`: 验证集AUC最高的模型权重
- `config.yaml`: 完整的训练配置
- `training_log.txt`: 详细的训练日志

## 🔄 下一步

训练完成后，可以进行：

1. **模型评估**: 在测试集上评估性能
2. **第二阶段**: 使用此模型训练异常分类器
3. **参数调优**: 尝试不同配置提升性能

### 第二阶段命令示例

```bash
python scripts/run_stage2_anomaly_classifier.py \
    --dataset assist17 \
    --baseline_model_path output/stage1_assist17_20240101_120000/baseline/best_model.pt \
    --device cuda
```

## ❓ 常见问题

### Q: 训练时间太长怎么办？
A: 可以减少`kt_epochs`或使用更小的模型配置进行快速验证。

### Q: AUC不如预期怎么办？
A: 检查数据预处理、增加模型容量、调整学习率。

### Q: 如何选择最佳配置？
A: 从保守配置开始，逐步增加模型容量，观察性能变化。

### Q: 可以中断训练吗？
A: 可以，模型会保存最佳检查点，可以从中断处继续。
