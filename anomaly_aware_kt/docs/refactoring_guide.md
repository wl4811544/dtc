# 第一阶段代码提取指南

本文档说明了从 `full_pipeline.py` 中提取第一阶段代码的过程和使用方法。

## 提取概述

### 提取目标

1. **代码分离**: 将第一阶段（基线模型训练）从完整流程中提取出来
2. **独立运行**: 第一阶段可以单独运行和测试
3. **保持兼容**: 与原始 `full_pipeline.py` 完全兼容
4. **简化结构**: 清晰的代码组织

### 新的目录结构

```
anomaly_aware_kt/
├── anomaly_kt/
│   ├── stages/                    # 新增：阶段模块
│   │   ├── __init__.py           # 模块导入
│   │   └── stage1_baseline.py    # 第一阶段：基线模型训练
│   ├── detector.py               # 原有模块
│   ├── model.py
│   └── ...
├── scripts/
│   ├── full_pipeline.py          # 原始完整流程（保持兼容）
│   ├── run_stage1_only.py        # 新增：单独运行第一阶段
│   └── ...
└── docs/
    ├── refactoring_guide.md       # 本文档
    └── ...
```

## 第一阶段重构详情

### 核心组件

#### 1. `common.py` - 通用工具模块

**主要功能:**
- 数据准备 (`prepare_data`)
- 配置管理 (`load_config`, `save_config`)
- 输出目录设置 (`setup_output_directory`)
- 基类定义 (`BaseStage`, `StageConfig`)

**关键类:**
```python
class StageConfig:
    """阶段配置基类"""

class BaseStage:
    """阶段基类，所有阶段都继承此类"""
```

#### 2. `stage1_baseline.py` - 第一阶段实现

**主要类:**

```python
class BaselineConfig(StageConfig):
    """基线模型配置，管理所有相关参数"""

class BaselineTrainer(BaseStage):
    """基线模型训练器，实现具体的训练逻辑"""
```

**主要方法:**
- `create_model()`: 创建DTransformer模型
- `create_trainer()`: 创建知识追踪训练器
- `run()`: 执行训练流程

### 使用方法

#### 方法1: 使用新的模块化接口

```python
from anomaly_kt.stages.common import prepare_data, setup_output_directory
from anomaly_kt.stages.stage1_baseline import BaselineConfig, BaselineTrainer

# 准备数据
train_data, val_data, test_data, dataset_config = prepare_data(...)

# 创建配置
config = BaselineConfig(args, dataset_config)

# 创建训练器并运行
trainer = BaselineTrainer(config)
model_path = trainer.run(train_data, val_data)
```

#### 方法2: 使用便捷函数（向后兼容）

```python
from anomaly_kt.stages.stage1_baseline import train_baseline_model

# 直接调用，与原始接口相同
model_path = train_baseline_model(args, dataset_config, train_data, val_data)
```

#### 方法3: 使用独立脚本

```bash
# 单独运行第一阶段
python scripts/run_stage1_only.py --dataset assist09 --d_model 128 --n_heads 8
```

### 配置管理

#### BaselineConfig 参数

**模型参数:**
- `d_model`: 模型维度
- `n_heads`: 注意力头数
- `n_know`: 知识概念数
- `n_layers`: 层数
- `dropout`: Dropout率
- `lambda_cl`: 对比学习权重
- `proj`: 是否使用投影
- `hard_neg`: 是否使用困难负样本
- `window`: 窗口大小

**训练参数:**
- `epochs`: 训练轮数
- `learning_rate`: 学习率
- `patience`: 早停耐心
- `use_cl`: 是否使用对比学习

### 优势

#### 1. 模块化设计
- 每个阶段独立，便于测试和调试
- 清晰的接口和职责分离
- 易于扩展和修改

#### 2. 配置管理
- 集中的参数管理
- 类型安全的配置访问
- 便于参数验证和默认值设置

#### 3. 代码复用
- 通用工具函数避免重复代码
- 基类提供统一的接口
- 便于在不同阶段间共享逻辑

#### 4. 向后兼容
- 保留原始函数接口
- 现有脚本无需修改
- 渐进式迁移

## 下一步计划

1. **第二阶段重构**: 提取异常检测器训练代码
2. **第三阶段重构**: 提取异常感知模型训练代码
3. **第四阶段重构**: 提取模型评估代码
4. **完整集成**: 更新 `full_pipeline.py` 使用新模块
5. **测试和文档**: 完善测试用例和使用文档

## 迁移指南

### 对于现有用户

1. **无需立即迁移**: 原始 `full_pipeline.py` 继续工作
2. **渐进式采用**: 可以逐步使用新的模块化接口
3. **测试新功能**: 使用 `run_stage1_only.py` 测试单阶段运行

### 对于开发者

1. **新功能开发**: 优先使用新的模块化结构
2. **Bug修复**: 在新模块中修复，保持向后兼容
3. **代码审查**: 确保新代码遵循模块化设计原则

## 示例用法

### 完整示例：单独运行第一阶段

```python
#!/usr/bin/env python
import argparse
from anomaly_kt.stages.common import prepare_data, setup_output_directory
from anomaly_kt.stages.stage1_baseline import BaselineConfig, BaselineTrainer

def main():
    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    # ... 其他参数
    args = parser.parse_args()

    # 设置环境
    args.output_dir = setup_output_directory(args.output_dir, args.dataset)

    # 准备数据
    train_data, val_data, test_data, dataset_config = prepare_data(
        args.dataset, args.data_dir, args.batch_size, args.test_batch_size
    )

    # 训练模型
    config = BaselineConfig(args, dataset_config)
    trainer = BaselineTrainer(config)
    model_path = trainer.run(train_data, val_data)

    print(f"模型保存到: {model_path}")

if __name__ == '__main__':
    main()
```

这个重构为项目提供了更好的结构和可维护性，同时保持了与现有代码的兼容性。
