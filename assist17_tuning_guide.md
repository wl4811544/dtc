# ASSIST17 参数调优指南

## 基线配置说明

### 当前起点参数
```python
d_model = 128        # 模型维度
n_heads = 8          # 注意力头数  
n_know = 16          # 知识概念数
n_layers = 3         # Transformer层数
dropout = 0.2        # Dropout率
batch_size = 16      # 批次大小
learning_rate = 1e-3 # 学习率
```

### 为什么选择这些参数？

1. **d_model=128**: 
   - 对于13K+样本的数据集，128维是安全的起点
   - 既不会过拟合，也有足够的表达能力
   - 训练速度适中

2. **n_heads=8**: 
   - d_model的因子，确保每个头的维度是16
   - 8个头能捕获不同类型的注意力模式

3. **n_know=16**: 
   - 保守设置，避免知识概念过多导致过拟合
   - 可以根据验证集性能逐步调整

4. **n_layers=3**: 
   - 3层足以建模序列依赖关系
   - 避免过深网络的训练困难

5. **dropout=0.2**: 
   - 适中的正则化强度
   - 平衡过拟合和欠拟合

## 训练监控指标

### 关键指标
- **训练集AUC**: 应该稳定上升
- **验证集AUC**: 最重要的指标
- **训练集ACC**: 辅助参考
- **验证集ACC**: 辅助参考
- **损失值**: 应该稳定下降

### 健康的训练曲线
```
Epoch 1: Train AUC=0.65, Val AUC=0.63, Train Loss=0.68
Epoch 5: Train AUC=0.72, Val AUC=0.70, Train Loss=0.61
Epoch 10: Train AUC=0.78, Val AUC=0.75, Train Loss=0.55
...
```

### 问题信号
- **过拟合**: 训练集AUC >> 验证集AUC (差距>0.05)
- **欠拟合**: 两者都很低且不再提升
- **训练不稳定**: 损失震荡或发散

## 参数调优策略

### 阶段1: 验证基线 (当前配置)
**目标**: 确认代码正常运行，获得基线性能

**预期结果**:
- 验证集AUC: 0.70-0.75
- 训练稳定，无明显过拟合
- 训练时间: ~30-60分钟

### 阶段2: 模型容量调优
**如果基线表现良好**，可以尝试：

```python
# 选项A: 增加模型维度
d_model = 256
n_heads = 16
# 其他参数保持不变

# 选项B: 增加层数
n_layers = 4
# 其他参数保持不变

# 选项C: 增加知识概念数
n_know = 32
# 其他参数保持不变
```

### 阶段3: 训练策略优化
**如果模型容量足够**，可以尝试：

```python
# 选项A: 调整学习率
learning_rate = 5e-4  # 更小的学习率

# 选项B: 启用困难负样本
hard_neg = True

# 选项C: 调整对比学习权重
lambda_cl = 0.2

# 选项D: 增加批次大小
batch_size = 32
```

### 阶段4: 高级优化
**如果基础配置都调优完毕**：

```python
# 更大的模型 (仅在前面阶段都成功时)
d_model = 512
n_heads = 16
n_layers = 6
n_know = 64
dropout = 0.1
```

## 实验记录模板

### 实验日志
```
实验ID: assist17_exp_001
配置: d_model=128, n_heads=8, n_layers=3, n_know=16
结果: Val AUC=0.73, Train AUC=0.75, 训练时间=45min
备注: 基线实验，训练稳定

实验ID: assist17_exp_002  
配置: d_model=256, n_heads=16, n_layers=3, n_know=16
结果: Val AUC=0.76, Train AUC=0.78, 训练时间=65min
备注: 增加模型维度，性能提升

...
```

## 快速启动命令

### 1. 检查配置 (推荐先运行)
```bash
python run_assist17_baseline.py --dry_run
```

### 2. 开始基线训练
```bash
python run_assist17_baseline.py
```

### 3. 监控训练进度
```bash
# 查看日志
tail -f output/assist17_baseline/train.log

# 查看GPU使用情况
nvidia-smi
```

## 预期时间线

- **基线训练**: 30-60分钟
- **参数调优**: 每个实验30-90分钟
- **完整调优**: 1-2天

## 故障排除

### 常见问题
1. **CUDA内存不足**: 减小batch_size或d_model
2. **训练过慢**: 检查GPU利用率，考虑增加batch_size
3. **性能不佳**: 检查数据预处理，确认标签正确性
4. **过拟合**: 增加dropout，减小模型容量

### 调试技巧
1. 先用小数据集验证代码
2. 监控梯度范数，避免梯度爆炸
3. 保存检查点，方便恢复训练
4. 记录详细的实验日志
