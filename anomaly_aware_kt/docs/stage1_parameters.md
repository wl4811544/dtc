# 第一阶段参数说明与实验指南

本文档详细说明第一阶段（基线模型训练）的参数配置、配置文件使用和实验流程。

## 🚀 快速开始

### 立即开始训练（推荐）
```bash
# ASSIST17 基线训练 - 使用自动配置
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config
```

### 配置文件系统
现在支持通过YAML配置文件管理参数，**命令行参数优先级更高**：

1. **自动配置**：`--auto_config` 根据数据集自动选择配置文件
2. **手动配置**：`--config path/to/config.yaml` 指定配置文件
3. **参数覆盖**：命令行参数会覆盖配置文件中的对应参数

### 可用配置文件
- `assist17_baseline.yaml` - 大规模数据集配置
- `assist09_baseline.yaml` - 中等规模数据集配置
- `statics_baseline.yaml` - 小规模数据集配置
- `algebra05_baseline.yaml` - 小规模数据集配置

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

## 🎯 推荐实验流程

### 第1步：基线验证（必须先做）
```bash
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config
```
**目的**：
- 验证代码和数据没问题
- 获得基线性能指标（预期验证集AUC: 0.70-0.75）
- 确认训练稳定性
- 训练时间：约60-90分钟

**如果这一步失败**：需要先解决代码/数据问题，不要继续后续实验

### 第2步：快速验证（可选，节省时间）
```bash
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config --kt_epochs 50
```
**目的**：
- 如果第1步训练时间太长，用这个快速验证
- 确认训练趋势正常
- 训练时间：约20-30分钟

### 第3步：模型容量扩展（基线成功后）
```bash
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config --d_model 256 --n_heads 16
```
**目的**：
- 测试更大模型是否能提升性能
- 预期验证集AUC提升2-5%
- 训练时间：约60-90分钟

### 第4步：学习率调优（如果需要）
```bash
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config --learning_rate 0.0005
```
**目的**：
- 如果前面步骤训练不够稳定，尝试更小学习率
- 或者如果想要更精细的训练

### 决策流程
- ✅ **基线成功** → 进行第3步（扩大模型）
- ❌ **训练不稳定** → 尝试第4步（调整学习率）
- ❌ **性能太差** → 检查数据和代码

## 📊 各数据集配置特点

| 数据集 | 模型大小 | 批次大小 | 对比学习 | 特点 |
|--------|----------|----------|----------|------|
| **assist17** | d_model=128 | batch=16 | ✅ | 大规模，保守起点 |
| **assist09** | d_model=128 | batch=16 | ✅ | 中等规模 |
| **statics** | d_model=64 | batch=8 | ❌ | 小规模，防过拟合 |
| **algebra05** | d_model=64 | batch=8 | ❌ | 小规模，防过拟合 |

## 参数使用示例

### 配置文件使用示例

#### 1. 自动配置（推荐）
```bash
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config
```

#### 2. 手动指定配置文件
```bash
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --config anomaly_aware_kt/configs/assist17_baseline.yaml
```

#### 3. 配置文件 + 参数覆盖
```bash
# 使用配置文件，但覆盖模型大小
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config --d_model 256 --n_heads 16

# 多个参数覆盖
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config --d_model 256 --learning_rate 0.0005 --kt_epochs 150
```

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

#### 基础命令（不使用配置文件）
```bash
python anomaly_aware_kt/scripts/run_stage1_only.py \
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

#### 推荐命令（使用配置文件）
```bash
# 基线训练
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config

# 参数调优
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config --d_model 256 --learning_rate 0.0005
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

## 🔧 参数调优建议

### 基于数据集规模的配置

#### 小数据集 (< 5K 样本) - statics, algebra05
```bash
# 使用预设配置（推荐）
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset statics --auto_config

# 手动配置
--d_model 64 --n_heads 4 --n_layers 2 --dropout 0.3 --batch_size 8 --use_cl false
```

#### 中等数据集 (5K - 15K 样本) - assist09
```bash
# 使用预设配置（推荐）
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist09 --auto_config

# 手动配置
--d_model 128 --n_heads 8 --n_layers 3 --dropout 0.2 --batch_size 16 --use_cl true
```

#### 大数据集 (> 15K 样本) - assist17
```bash
# 基线配置（推荐起点）
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config

# 大模型配置
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config --d_model 256 --n_heads 16 --n_layers 4
```

### 渐进式调优策略

#### 阶段1：验证基线
```bash
# 使用默认配置，确保代码正常运行
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config --kt_epochs 50
```

#### 阶段2：扩展模型容量

##### 方式1：增加模型维度（推荐优先尝试）
```bash
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config --d_model 256 --n_heads 16
```
**作用**：模型隐藏维度 128→256，注意力头数 8→16
**效果**：✅ 表达能力最强，通常效果最好
**代价**：❌ 计算开销最大，过拟合风险高
**适用**：数据量充足，希望获得最佳性能

##### 方式2：增加层数（第二选择）
```bash
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config --n_layers 4
```
**作用**：Transformer层数 3→4
**效果**：✅ 增强序列建模，捕获长期依赖
**代价**：❌ 训练难度增加，收敛变慢
**适用**：序列较长，需要建模长期依赖关系

##### 方式3：增加知识概念数（细节优化）
```bash
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config --n_know 32
```
**作用**：知识概念向量数量 16→32
**效果**：✅ 知识表示更细粒度，训练稳定
**代价**：❌ 提升有限，对整体性能影响较小
**适用**：知识点复杂多样，希望更精细的知识建模

##### 📊 三种扩展方式对比

| 扩展方式 | 参数增加量 | 训练时间增加 | 内存占用增加 | 性能提升潜力 | 过拟合风险 | 推荐优先级 |
|----------|------------|--------------|--------------|--------------|------------|------------|
| **增加维度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 🥇 第1优先 |
| **增加层数** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 🥈 第2优先 |
| **增加知识概念** | ⭐ | ⭐ | ⭐ | ⭐⭐ | ⭐ | 🥉 第3优先 |

##### 🔄 组合使用建议

**保守组合（推荐起点）**：
```bash
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config --d_model 256 --n_heads 16
```

**积极组合（如果保守组合成功）**：
```bash
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config --d_model 256 --n_heads 16 --n_layers 4
```

**完整组合（最大模型，谨慎使用）**：
```bash
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config --d_model 256 --n_heads 16 --n_layers 4 --n_know 32
```

**⚠️ 重要提醒**：
- 建议**逐步扩展**，一次只增加一种维度
- 每次扩展后观察验证集性能和过拟合情况
- 如果出现过拟合，停止扩展并增加正则化

#### 阶段3：优化训练策略
```bash
# 调整学习率
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config --learning_rate 0.0005

# 启用高级特性
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config --hard_neg --lambda_cl 0.2

# 调整批次大小
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config --batch_size 32
```

## ❓ 常见问题与解决方案

### Q: 如何确定合适的模型大小？
A: **使用渐进式策略**：
1. 先运行基线配置：`--auto_config`
2. 如果性能良好且无过拟合，按优先级扩展：
   - 第1步：`--d_model 256 --n_heads 16` (增加维度)
   - 第2步：`--n_layers 4` (增加层数)
   - 第3步：`--n_know 32` (增加知识概念)
3. 每步都要检查验证集性能，出现过拟合就停止

### Q: 三种模型扩展方式应该选哪个？
A: **按优先级选择**：
- **首选**：增加模型维度 (`--d_model 256 --n_heads 16`) - 效果最好
- **次选**：增加层数 (`--n_layers 4`) - 增强序列建模
- **最后**：增加知识概念 (`--n_know 32`) - 细节优化
- **组合**：可以逐步组合，但要注意过拟合

### Q: 如何判断模型扩展是否成功？
A: **观察这些指标**：
- ✅ 验证集AUC提升 > 2%
- ✅ 训练集和验证集AUC差距 < 5%
- ✅ 训练过程稳定，损失平稳下降
- ❌ 如果验证集AUC下降或差距过大，说明过拟合了

### Q: 配置文件和命令行参数冲突怎么办？
A: **命令行参数优先级更高**，会自动覆盖配置文件中的对应参数。例如：
```bash
# 配置文件中 d_model=128，但命令行指定 d_model=256，最终使用256
python run_stage1_only.py --dataset assist17 --auto_config --d_model 256
```

### Q: 对比学习什么时候使用？
A: **根据数据集规模**：
- 大数据集（assist17, assist09）：默认启用 `use_cl=true`
- 小数据集（statics, algebra05）：默认禁用 `use_cl=false`
- 手动控制：`--use_cl` 或不加此参数

### Q: 如何处理过拟合？
A: **按优先级尝试**：
1. 使用小数据集配置：`--dataset statics --auto_config`
2. 增加dropout：`--dropout 0.4`
3. 减少模型容量：`--d_model 64 --n_layers 2`
4. 禁用对比学习：不加 `--use_cl` 参数

### Q: 训练太慢怎么办？
A: **优化策略**：
1. 减少训练轮数：`--kt_epochs 50`
2. 增加批次大小：`--batch_size 32`
3. 使用小模型：`--d_model 64 --n_layers 2`
4. 检查GPU利用率：`nvidia-smi`

### Q: 如何快速验证代码是否正常？
A: **快速测试**：
```bash
# 5-10分钟快速验证
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config --kt_epochs 5
```

### Q: 实验结果如何记录和比较？
A: **自动记录**：
- 每次训练会自动生成带时间戳的输出目录
- 配置参数保存在 `output/*/config.yaml`
- 训练日志保存在输出目录中
- 建议手动记录关键指标：验证集AUC、训练时间

## 🎯 最佳实践总结

### 1. 新手入门
```bash
# 第一次运行，验证环境
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config --kt_epochs 10

# 确认无误后，完整基线训练
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config
```

### 2. 参数调优（推荐流程）
```bash
# 第1步：基线验证
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config

# 第2步：扩展模型维度（优先级最高）
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config --d_model 256 --n_heads 16

# 第3步：如果第2步成功，继续增加层数
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config --d_model 256 --n_heads 16 --n_layers 4

# 第4步：学习率微调
python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config --d_model 256 --n_heads 16 --learning_rate 0.0005
```

**⚠️ 重要**：每一步都要检查验证集AUC，如果下降就停止扩展

### 3. 实验管理
- 一次只改变一个参数
- 记录每次实验的验证集AUC
- 保存最佳模型的配置
- 使用有意义的输出目录名称

### 4. 故障排除
- 训练失败：检查数据路径和GPU内存
- 性能差：确认数据预处理正确
- 过拟合：使用小数据集配置或增加正则化
- 欠拟合：增加模型容量或训练轮数


