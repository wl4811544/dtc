# 第三阶段详细指南：异常感知知识追踪

## 📋 概述

第三阶段是整个系统的核心目标：将第一阶段的基线知识追踪模型与第二阶段的异常检测器融合，实现异常感知的知识追踪，**目标提升AUC 0.05-0.1**。

## 🎯 核心目标

### 📊 **性能提升目标**
- **基线AUC**: ~0.74-0.76 (第一阶段)
- **目标AUC**: ~0.79-0.86 (+0.05-0.1)
- **最小提升**: 0.05 AUC (显著改进)
- **理想提升**: 0.1 AUC (重大突破)

### 🔬 **技术创新**
1. **异常权重调整**: 降低异常数据对知识状态更新的负面影响
2. **上下文增强**: 利用异常检测信息增强序列表示
3. **渐进式训练**: 稳定的三阶段训练策略
4. **多任务学习**: 保持异常检测一致性

## 🏗️ **技术架构**

### 核心组件

#### 1. **异常感知融合层**
```python
class AnomalyAwareFusion:
    # 异常权重调整器
    weight_adjuster: AnomalyWeightAdjuster
    
    # 融合策略 (attention/gating/weighted)
    fusion_strategy: FusionStrategy
    
    # 上下文增强器
    context_enhancer: ContextEnhancer
```

#### 2. **渐进式训练策略**
```
阶段1 (10轮): 只训练融合层，冻结预训练模型
阶段2 (20轮): 解冻异常检测器，联合训练
阶段3 (10轮): 端到端微调，解冻所有模型
```

#### 3. **多任务学习**
```python
total_loss = kt_loss + λ * anomaly_consistency_loss
```

### 🔧 **融合策略详解**

#### 策略1: 注意力融合 (推荐)
```python
# 使用异常信息作为注意力权重
attention_weight = f(anomaly_prob)
fused_state = attention_weight * enhanced_state + (1 - attention_weight) * baseline_state
```

#### 策略2: 门控融合
```python
# 学习自适应门控
gate = learned_gate(baseline_state, anomaly_prob)
fused_state = gate * baseline_state + (1 - gate) * anomaly_adjustment
```

#### 策略3: 加权融合
```python
# 简单线性组合
weight = weight_net(anomaly_prob)
fused_state = weight * baseline_state + (1 - weight) * adjusted_state
```

## 📊 **预期性能贡献**

| 组件 | 预期AUC提升 | 技术原理 | 优先级 |
|------|-------------|----------|--------|
| **异常权重调整** | +0.02-0.04 | 降低异常数据影响 | 🔥 高 |
| **上下文增强** | +0.03-0.05 | 利用异常信息增强表示 | 🔥 高 |
| **多任务学习** | +0.01-0.02 | 保持检测一致性 | 中 |
| **渐进式训练** | +0.01-0.02 | 稳定训练过程 | 中 |
| **总计** | **+0.07-0.13** | - | - |

## 🚀 **使用指南**

### 基础使用

#### 1. **基础模型异常感知知识追踪**

```bash
python scripts/run_stage3_anomaly_aware_kt.py \
    --dataset assist17 \
    --model_type basic \
    --baseline_model_path output/stage1_basic_assist17_*/baseline/best_model.pt \
    --anomaly_detector_path output/stage2_basic_assist17_*/anomaly_classifier/best_anomaly_detector.pt \
    --auto_config \
    --device cuda
```

#### 2. **扩展模型异常感知知识追踪**

```bash
python scripts/run_stage3_anomaly_aware_kt.py \
    --dataset assist17 \
    --model_type extended \
    --baseline_model_path output/stage1_extended_assist17_*/baseline/best_model.pt \
    --anomaly_detector_path output/stage2_extended_assist17_*/anomaly_classifier/best_anomaly_detector.pt \
    --auto_config \
    --device cuda
```

### 高级配置

#### 自定义融合策略

```bash
python scripts/run_stage3_anomaly_aware_kt.py \
    --dataset assist17 \
    --model_type basic \
    --baseline_model_path "..." \
    --anomaly_detector_path "..." \
    --fusion_type gating \
    --enable_context_enhancement \
    --lambda_anomaly 0.2 \
    --device cuda
```

#### 调整训练策略

```bash
python scripts/run_stage3_anomaly_aware_kt.py \
    --dataset assist17 \
    --model_type basic \
    --baseline_model_path "..." \
    --anomaly_detector_path "..." \
    --fusion_epochs 15 \
    --joint_epochs 30 \
    --finetune_epochs 15 \
    --learning_rate 0.0005 \
    --device cuda
```

## ⚙️ **配置参数详解**

### 融合参数

```yaml
# 融合策略
fusion_type: attention              # attention, gating, weighted
enable_context_enhancement: true    # 启用上下文增强
lambda_anomaly: 0.1                # 异常一致性损失权重

# 异常权重调整
anomaly_discount_factor: 0.7       # 异常折扣因子
min_weight: 0.1                    # 最小权重
adjustment_type: linear             # linear, exponential, sigmoid
```

### 渐进式训练参数

```yaml
# 三阶段训练轮数
fusion_epochs: 10      # 阶段1：融合层训练
joint_epochs: 20       # 阶段2：联合训练
finetune_epochs: 10    # 阶段3：端到端微调

# 学习率策略
learning_rate: 0.001   # 初始学习率
joint_lr_factor: 0.5   # 阶段2学习率衰减
finetune_lr_factor: 0.1 # 阶段3学习率衰减
```

## 📈 **训练过程监控**

### 关键指标

1. **知识追踪AUC**: 主要目标指标
2. **异常一致性损失**: 多任务学习指标
3. **融合权重分布**: 融合效果监控
4. **梯度流**: 训练稳定性监控

### 预期训练曲线

```
阶段1 (融合层训练):
  Epoch 1-5:   AUC 0.74 → 0.76 (快速提升)
  Epoch 6-10:  AUC 0.76 → 0.78 (稳定提升)

阶段2 (联合训练):
  Epoch 11-20: AUC 0.78 → 0.81 (显著提升)
  Epoch 21-30: AUC 0.81 → 0.83 (持续优化)

阶段3 (端到端微调):
  Epoch 31-35: AUC 0.83 → 0.85 (精细调优)
  Epoch 36-40: AUC 0.85 → 0.86 (收敛)
```

## 📁 **输出结构**

```
output/stage3_basic_assist17_20240101_120000/
├── config.yaml                        # 训练配置
├── anomaly_aware_kt/                   # 异常感知模型目录
│   ├── best_anomaly_aware_kt.pt       # 最佳模型
│   ├── training_log.txt               # 训练日志
│   └── performance_analysis.txt       # 性能分析
├── evaluation/                        # 评估结果
│   ├── auc_comparison.png             # AUC对比图
│   ├── ablation_study.txt             # 消融研究
│   └── component_analysis.txt         # 组件贡献分析
└── plots/                             # 可视化图表
    ├── training_curves.png            # 训练曲线
    ├── fusion_weights.png             # 融合权重分布
    └── anomaly_impact.png             # 异常影响分析
```

## 🔧 **性能调优指南**

### 如果AUC提升不足0.05

#### 1. **增强融合策略**
```bash
--fusion_type gating \
--lambda_anomaly 0.2 \
--enable_context_enhancement
```

#### 2. **延长训练时间**
```bash
--fusion_epochs 15 \
--joint_epochs 30 \
--finetune_epochs 15
```

#### 3. **调整学习率**
```bash
--learning_rate 0.0005
```

#### 4. **尝试不同权重调整**
```yaml
adjustment_type: exponential
anomaly_discount_factor: 0.8
```

### 如果训练不稳定

#### 1. **降低学习率**
```bash
--learning_rate 0.0005
```

#### 2. **增加预热轮数**
```bash
--fusion_epochs 15
```

#### 3. **调整损失权重**
```bash
--lambda_anomaly 0.05
```

## 🔬 **实验建议**

### 消融研究

1. **无异常权重调整**
   ```bash
   # 修改配置禁用权重调整
   anomaly_discount_factor: 0.0
   ```

2. **无上下文增强**
   ```bash
   --enable_context_enhancement false
   ```

3. **不同融合策略对比**
   ```bash
   # 分别测试 attention, gating, weighted
   --fusion_type attention
   --fusion_type gating
   --fusion_type weighted
   ```

### 性能分析

1. **组件贡献分析**: 评估各组件对AUC提升的贡献
2. **异常类型分析**: 分析不同异常类型的处理效果
3. **序列长度分析**: 评估在不同序列长度上的性能
4. **数据集泛化**: 在其他数据集上验证效果

## 📊 **成功标准**

### 🎯 **最小成功标准**
- AUC提升 ≥ 0.05
- 训练过程稳定
- 相对于基线有统计显著性提升

### 🚀 **理想成功标准**
- AUC提升 ≥ 0.1
- 在多个数据集上都有提升
- 消融研究证明各组件有效

### 📈 **评估指标**
- **主要指标**: AUC提升
- **辅助指标**: Accuracy, Precision, Recall, F1
- **稳定性**: 多次运行的方差
- **效率**: 训练时间和推理速度

## 🔄 **与其他阶段的衔接**

### 输入依赖
- **第一阶段**: 基线知识追踪模型
- **第二阶段**: 异常检测器

### 输出产物
- **异常感知知识追踪模型**: 最终的高性能模型
- **性能分析报告**: 详细的提升分析
- **消融研究结果**: 各组件贡献验证

这个第三阶段是整个项目的核心，成功实现AUC 0.05-0.1的提升将证明异常感知知识追踪方法的有效性，为教育AI领域做出重要贡献。
