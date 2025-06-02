# 配置系统详细指南

## 📋 概述

Anomaly-Aware Knowledge Tracing v2 采用灵活的配置系统，支持多种配置方式，确保实验的可重现性和参数管理的便利性。

## 🏗️ 配置系统架构

### 配置优先级

1. **命令行参数** (最高优先级)
2. **配置文件参数**
3. **默认值** (最低优先级)

### 配置文件格式

使用YAML格式，支持注释和层次结构：

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

## 📁 配置文件管理

### 预定义配置文件

#### `assist17_baseline.yaml`
```yaml
# ASSIST17 基线训练配置
# 针对大规模数据集的保守起点设置

# ==================== 基本参数 ====================
dataset: assist17
data_dir: data
output_dir: null  # 如果为null，将自动生成时间戳目录
device: cuda
with_pid: true  # assist17有问题ID

# ==================== 数据加载参数 ====================
batch_size: 16          # 保守起点，避免内存问题
test_batch_size: 32     # 测试时可以用更大batch

# ==================== 模型架构参数 ====================
# 保守的起点设置 - 可以稳定训练并观察效果
d_model: 128            # 中等维度，平衡性能和效率
n_heads: 8              # d_model的因子
n_know: 16              # 知识概念数，可以后续调整
n_layers: 3             # 3层Transformer
dropout: 0.2            # 适中的dropout
lambda_cl: 0.1          # 对比学习权重
proj: true              # 使用投影层
hard_neg: false         # 起点不用困难负样本
window: 1               # 注意力窗口

# ==================== 训练参数 ====================
kt_epochs: 100          # 最大训练轮数
learning_rate: 0.001    # 标准学习率
patience: 10            # 早停耐心值
use_cl: true            # 大数据集可以使用对比学习

# ==================== 实验记录 ====================
experiment_name: "assist17_baseline_v1"
description: "ASSIST17数据集基线实验，保守参数设置"
tags: ["baseline", "assist17", "conservative"]
```

### 自定义配置文件

#### 创建新配置文件

```yaml
# my_experiment.yaml
# 自定义实验配置

# 继承基础配置
base_config: assist17_baseline.yaml

# 覆盖特定参数
d_model: 256
n_heads: 16
kt_epochs: 200
learning_rate: 0.0005

# 实验标识
experiment_name: "high_capacity_experiment"
description: "高容量模型实验"
tags: ["high_capacity", "assist17", "experimental"]
```

## 🔧 配置使用方法

### 1. 自动配置（推荐）

系统根据数据集自动选择最佳配置：

```bash
python scripts/run_stage1_baseline.py \
    --dataset assist17 \
    --auto_config \
    --device cuda
```

**优点**:
- 无需手动指定配置文件
- 使用经过验证的最佳参数
- 适合快速开始实验

### 2. 指定配置文件

使用特定的配置文件：

```bash
python scripts/run_stage1_baseline.py \
    --dataset assist17 \
    --config configs/assist17_baseline.yaml \
    --device cuda
```

**优点**:
- 完全控制所有参数
- 便于实验重现
- 支持复杂的参数组合

### 3. 混合配置

配置文件 + 命令行参数覆盖：

```bash
python scripts/run_stage1_baseline.py \
    --dataset assist17 \
    --config configs/assist17_baseline.yaml \
    --d_model 256 \
    --learning_rate 0.0005 \
    --device cuda
```

**优点**:
- 基于稳定配置进行微调
- 命令行参数优先级更高
- 便于参数扫描

### 4. 纯命令行配置

完全使用命令行参数：

```bash
python scripts/run_stage1_baseline.py \
    --dataset assist17 \
    --d_model 128 \
    --n_heads 8 \
    --n_layers 3 \
    --kt_epochs 100 \
    --learning_rate 0.001 \
    --device cuda
```

**优点**:
- 最大灵活性
- 适合脚本化和自动化
- 便于参数扫描

## 📊 参数类别详解

### 基本参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dataset` | str | 必需 | 数据集名称 |
| `data_dir` | str | "data" | 数据目录路径 |
| `output_dir` | str | None | 输出目录（None时自动生成） |
| `device` | str | "cuda" | 训练设备 |
| `with_pid` | bool | False | 是否使用问题ID |

### 数据参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `batch_size` | int | 32 | 训练批次大小 |
| `test_batch_size` | int | 64 | 测试批次大小 |

### 模型参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `d_model` | int | 128 | 模型隐藏维度 |
| `n_heads` | int | 8 | 注意力头数 |
| `n_know` | int | 16 | 知识概念数 |
| `n_layers` | int | 3 | Transformer层数 |
| `dropout` | float | 0.2 | Dropout率 |
| `lambda_cl` | float | 0.1 | 对比学习权重 |
| `proj` | bool | False | 使用投影层 |
| `hard_neg` | bool | False | 使用困难负样本 |
| `window` | int | 1 | 注意力窗口大小 |

### 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `kt_epochs` | int | 100 | 最大训练轮数 |
| `learning_rate` | float | 0.001 | 学习率 |
| `patience` | int | 10 | 早停耐心值 |
| `use_cl` | bool | False | 使用对比学习 |

## 🎯 配置最佳实践

### 1. 实验命名规范

```yaml
experiment_name: "{dataset}_{model_type}_{version}"
# 例如: "assist17_baseline_v1"

description: "简洁描述实验目的和特点"
# 例如: "ASSIST17数据集基线实验，保守参数设置"

tags: ["category1", "category2", "feature"]
# 例如: ["baseline", "assist17", "conservative"]
```

### 2. 参数组织结构

```yaml
# ==================== 基本参数 ====================
dataset: assist17
device: cuda

# ==================== 数据参数 ====================
batch_size: 16
test_batch_size: 32

# ==================== 模型参数 ====================
d_model: 128
n_heads: 8
# ... 其他模型参数

# ==================== 训练参数 ====================
kt_epochs: 100
learning_rate: 0.001
# ... 其他训练参数
```

### 3. 注释规范

```yaml
# 参数说明和选择理由
d_model: 128            # 中等维度，平衡性能和效率
n_heads: 8              # d_model的因子，确保整除
dropout: 0.2            # 适中的dropout，防止过拟合
```

## 🔍 配置验证

### 自动验证

系统会自动验证配置的有效性：

```python
# 参数一致性检查
assert d_model % n_heads == 0, "d_model必须能被n_heads整除"

# 参数范围检查
assert 0.0 <= dropout <= 1.0, "dropout必须在[0,1]范围内"

# 必需参数检查
required_params = ['dataset', 'device']
for param in required_params:
    assert param in config, f"缺少必需参数: {param}"
```

### 手动验证

```bash
# 验证配置文件语法
python -c "import yaml; yaml.safe_load(open('configs/my_config.yaml'))"

# 验证参数完整性
python scripts/validate_config.py --config configs/my_config.yaml
```

## 📈 配置模板

### 快速开始模板

```yaml
# quick_start.yaml
dataset: assist17
device: cuda
auto_config: true
```

### 调试模板

```yaml
# debug.yaml
dataset: assist17
device: cpu
batch_size: 4
kt_epochs: 5
d_model: 64
n_heads: 4
```

### 高性能模板

```yaml
# high_performance.yaml
dataset: assist17
device: cuda
d_model: 512
n_heads: 16
n_layers: 6
kt_epochs: 200
learning_rate: 0.0005
```

### 消融研究模板

```yaml
# ablation_study.yaml
base_config: assist17_baseline.yaml

# 消融变量
use_cl: false           # 关闭对比学习
proj: false             # 关闭投影层
dropout: 0.0            # 关闭dropout

experiment_name: "ablation_no_cl_no_proj"
```

## 🔄 配置迁移

### 从旧版本迁移

```python
# 配置转换脚本
def migrate_config_v1_to_v2(old_config):
    new_config = {}
    
    # 参数名称映射
    name_mapping = {
        'model_dim': 'd_model',
        'num_heads': 'n_heads',
        'num_layers': 'n_layers'
    }
    
    for old_key, value in old_config.items():
        new_key = name_mapping.get(old_key, old_key)
        new_config[new_key] = value
    
    return new_config
```

### 配置版本控制

```yaml
# 配置文件版本信息
config_version: "2.0"
created_date: "2024-01-01"
last_modified: "2024-01-15"
author: "researcher_name"

# 变更日志
changelog:
  - "2024-01-15: 增加对比学习参数"
  - "2024-01-10: 调整默认学习率"
  - "2024-01-01: 初始版本"
```

## 🛠️ 高级配置

### 条件配置

```yaml
# 根据数据集自动调整参数
dataset_specific:
  assist09:
    d_model: 64
    batch_size: 32
    use_cl: false
  assist17:
    d_model: 256
    batch_size: 16
    use_cl: true
```

### 环境变量支持

```yaml
# 支持环境变量
device: ${CUDA_DEVICE:-cuda}
data_dir: ${DATA_DIR:-data}
output_dir: ${OUTPUT_DIR:-output}
```

### 配置继承

```yaml
# child_config.yaml
extends: "configs/base_config.yaml"

# 只覆盖需要修改的参数
d_model: 256
learning_rate: 0.0005
```

## 📝 配置文档化

### 自动生成文档

```bash
# 生成配置文档
python scripts/generate_config_docs.py \
    --config configs/assist17_baseline.yaml \
    --output docs/config_assist17_baseline.md
```

### 配置对比

```bash
# 对比两个配置文件
python scripts/compare_configs.py \
    --config1 configs/baseline.yaml \
    --config2 configs/experimental.yaml
```

这个配置系统为实验提供了最大的灵活性和可重现性，支持从简单的快速开始到复杂的研究实验的各种需求。
