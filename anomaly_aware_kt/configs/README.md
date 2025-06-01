# 配置文件使用指南

## 概述

本目录包含了不同数据集的基线训练配置文件，支持通过YAML格式的配置文件来管理训练参数。

## 配置文件列表

- `assist17_baseline.yaml` - ASSIST17数据集基线配置（大规模数据集）
- `assist09_baseline.yaml` - ASSIST09数据集基线配置（中等规模数据集）
- `statics_baseline.yaml` - STATICS数据集基线配置（小规模数据集）
- `algebra05_baseline.yaml` - ALGEBRA05数据集基线配置（小规模数据集）

## 使用方法

### 1. 自动配置（推荐）

根据数据集自动选择对应的配置文件：

```bash
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config
```

### 2. 手动指定配置文件

```bash
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --config anomaly_aware_kt/configs/assist17_baseline.yaml
```

### 3. 配置文件 + 命令行参数覆盖

命令行参数优先级更高，可以覆盖配置文件中的设置：

```bash
# 使用配置文件，但覆盖模型维度和学习率
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config --d_model 256 --learning_rate 0.0005
```

## 参数优先级

1. **命令行参数** (最高优先级)
2. **配置文件参数**
3. **脚本默认值** (最低优先级)

## 配置文件格式

```yaml
# 基本参数
dataset: assist17
data_dir: data
device: cuda
with_pid: true

# 数据加载参数
batch_size: 16
test_batch_size: 32

# 模型架构参数
d_model: 128
n_heads: 8
n_know: 16
n_layers: 3
dropout: 0.2
lambda_cl: 0.1
proj: true
hard_neg: false
window: 1

# 训练参数
kt_epochs: 100
learning_rate: 0.001
patience: 10
use_cl: true

# 实验记录
experiment_name: "assist17_baseline_v1"
description: "实验描述"
tags: ["baseline", "assist17"]
```

## 各数据集配置特点

### ASSIST17 (大规模数据集)
- 模型维度: 128
- 批次大小: 16
- 启用对比学习
- 适中的dropout (0.2)

### ASSIST09 (中等规模数据集)
- 模型维度: 128
- 批次大小: 16
- 启用对比学习
- 适中的dropout (0.2)

### STATICS (小规模数据集)
- 模型维度: 64 (较小)
- 批次大小: 8 (较小)
- 禁用对比学习
- 高dropout (0.3) 防止过拟合
- 无问题ID (with_pid: false)

### ALGEBRA05 (小规模数据集)
- 模型维度: 64 (较小)
- 批次大小: 8 (较小)
- 禁用对比学习
- 高dropout (0.3) 防止过拟合
- 有问题ID (with_pid: true)

## 创建自定义配置

1. 复制现有配置文件
2. 修改参数值
3. 保存为新的YAML文件
4. 使用 `--config` 参数指定

```bash
cp anomaly_aware_kt/configs/assist17_baseline.yaml my_custom_config.yaml
# 编辑 my_custom_config.yaml
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --config my_custom_config.yaml
```

## 常用参数调优

### 增加模型容量
```yaml
d_model: 256      # 从128增加到256
n_heads: 16       # 对应增加头数
n_layers: 4       # 增加层数
```

### 防止过拟合
```yaml
dropout: 0.3      # 增加dropout
batch_size: 8     # 减小批次大小
use_cl: false     # 禁用对比学习
```

### 加速训练
```yaml
batch_size: 32    # 增加批次大小
learning_rate: 0.002  # 增加学习率
kt_epochs: 50     # 减少训练轮数
```

## 实验管理

配置文件中的实验记录字段帮助管理实验：

```yaml
experiment_name: "assist17_large_model_v1"
description: "测试更大模型在ASSIST17上的效果"
tags: ["large_model", "assist17", "experiment"]
```

这些信息会保存在输出目录的配置文件中，便于后续分析和复现。
