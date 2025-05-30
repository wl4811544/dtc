# 第一阶段参数说明

本文档详细说明第一阶段（基线模型训练）需要的所有参数。

## 参数分类

### 1. 基本参数

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `device` | str | 是 | 'cuda'/'cpu' | 训练设备 |
| `output_dir` | str | 是 | None | 输出目录路径 |
| `with_pid` | bool | 否 | False | 是否使用问题ID |

### 2. 模型架构参数

这些参数用于创建 DTransformer 模型：

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `d_model` | int | 是 | 128 | 模型维度 |
| `n_heads` | int | 是 | 8 | 注意力头数 |
| `n_know` | int | 是 | 16 | 知识概念数 |
| `n_layers` | int | 是 | 3 | Transformer层数 |
| `dropout` | float | 是 | 0.2 | Dropout率 |
| `lambda_cl` | float | 是 | 0.1 | 对比学习权重 |
| `proj` | bool | 否 | False | 是否使用投影层 |
| `hard_neg` | bool | 否 | False | 是否使用困难负样本 |
| `window` | int | 是 | 1 | 窗口大小 |

### 3. 训练参数

这些参数用于配置训练过程：

| 参数名 | 类型 | 必需 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `kt_epochs` | int | 是 | 100 | 训练轮数 |
| `learning_rate` | float | 是 | 1e-3 | 学习率 |
| `patience` | int | 是 | 10 | 早停耐心值 |
| `use_cl` | bool | 否 | False | 是否使用对比学习 |

### 4. 数据集配置参数

这些参数来自 `dataset_config`：

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| `n_questions` | int | 是 | 数据集中的问题总数 |
| `n_pid` | int | 是 | 数据集中的问题ID总数 |

## 参数使用示例

### 在代码中的使用

```python
from anomaly_kt.stages.stage1_baseline import train_baseline_model

# 模拟参数对象
class Args:
    def __init__(self):
        # 基本参数
        self.device = 'cuda'
        self.output_dir = 'output/baseline_test'
        self.with_pid = True
        
        # 模型参数
        self.d_model = 128
        self.n_heads = 8
        self.n_know = 16
        self.n_layers = 3
        self.dropout = 0.2
        self.lambda_cl = 0.1
        self.proj = False
        self.hard_neg = False
        self.window = 1
        
        # 训练参数
        self.kt_epochs = 50
        self.learning_rate = 1e-3
        self.patience = 10
        self.use_cl = True

# 数据集配置
dataset_config = {
    'n_questions': 100,
    'n_pid': 50
}

# 调用训练函数
args = Args()
model_path = train_baseline_model(args, dataset_config, train_data, val_data)
```

### 命令行参数示例

如果使用 `run_stage1_only.py` 脚本：

```bash
python run_stage1_only.py \
    --dataset assist09 \
    --device cuda \
    --output_dir output/my_baseline \
    --with_pid \
    --d_model 256 \
    --n_heads 16 \
    --n_know 32 \
    --n_layers 4 \
    --dropout 0.1 \
    --lambda_cl 0.2 \
    --proj \
    --hard_neg \
    --window 2 \
    --kt_epochs 200 \
    --learning_rate 5e-4 \
    --patience 15 \
    --use_cl
```

## 参数详细说明

### 模型架构参数详解

1. **`d_model`**: 模型的隐藏维度，影响模型容量和计算复杂度
2. **`n_heads`**: 多头注意力的头数，通常是 d_model 的因子
3. **`n_know`**: 知识概念的数量，用于知识追踪
4. **`n_layers`**: Transformer编码器的层数
5. **`dropout`**: 防止过拟合的dropout率
6. **`lambda_cl`**: 对比学习损失的权重
7. **`proj`**: 是否在注意力层后添加投影层
8. **`hard_neg`**: 是否使用困难负样本策略
9. **`window`**: 注意力窗口大小

### 训练参数详解

1. **`kt_epochs`**: 最大训练轮数
2. **`learning_rate`**: 优化器学习率
3. **`patience`**: 早停机制的耐心值，验证集性能不提升的最大轮数
4. **`use_cl`**: 是否启用对比学习训练策略

## 参数调优建议

### 小数据集 (< 10K 样本)
```python
d_model = 64
n_heads = 4
n_layers = 2
dropout = 0.3
learning_rate = 1e-3
```

### 中等数据集 (10K - 100K 样本)
```python
d_model = 128
n_heads = 8
n_layers = 3
dropout = 0.2
learning_rate = 1e-3
```

### 大数据集 (> 100K 样本)
```python
d_model = 256
n_heads = 16
n_layers = 4
dropout = 0.1
learning_rate = 5e-4
```

## 常见问题

### Q: 如何确定合适的模型大小？
A: 从较小的模型开始（d_model=64），逐步增加直到验证集性能不再提升。

### Q: 对比学习什么时候使用？
A: 当数据集较大且希望提升模型表示能力时，设置 `use_cl=True` 和适当的 `lambda_cl`。

### Q: 如何处理过拟合？
A: 增加 `dropout` 值，减少 `n_layers`，或使用更大的 `patience` 值。

### Q: 训练太慢怎么办？
A: 减少 `d_model`、`n_heads` 或 `n_layers`，或使用更强的GPU。
