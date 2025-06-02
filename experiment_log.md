# ASSIST17 实验记录

## 基准测试结果

### 实验1: 基线配置 (assist17_baseline.yaml)
**日期**: 2024年12月  
**配置**: 使用自动配置 `--dataset assist17 --auto_config`

**模型参数**:
- d_model: 128
- n_heads: 8  
- n_layers: 3
- dropout: 0.2
- batch_size: 16
- learning_rate: 0.001

**训练结果**:
- **最佳AUC**: 0.7407 (Epoch 49)
- **最终ACC**: 0.704
- **最终MAE**: 0.381
- **训练轮数**: 59 (早停)
- **最终损失**: 0.5641

**详细训练过程**:
```
Epoch 30: Train Loss: 0.5773, Val AUC: 0.736, ACC: 0.703, MAE: 0.388
Epoch 31: Train Loss: 0.5772, Val AUC: 0.738, ACC: 0.704, MAE: 0.390
Epoch 32: Train Loss: 0.5764, Val AUC: 0.736, ACC: 0.703, MAE: 0.382
Epoch 33: Train Loss: 0.5750, Val AUC: 0.737, ACC: 0.703, MAE: 0.387
Epoch 34: Train Loss: 0.5759, Val AUC: 0.733, ACC: 0.703, MAE: 0.389
Epoch 35: Train Loss: 0.5740, Val AUC: 0.737, ACC: 0.703, MAE: 0.384
Epoch 36: Train Loss: 0.5734, Val AUC: 0.737, ACC: 0.704, MAE: 0.383
Epoch 37: Train Loss: 0.5737, Val AUC: 0.735, ACC: 0.703, MAE: 0.384
Epoch 38: Train Loss: 0.5729, Val AUC: 0.739, ACC: 0.703, MAE: 0.386 ✓ New best AUC
Epoch 39: Train Loss: 0.5720, Val AUC: 0.737, ACC: 0.703, MAE: 0.381
Epoch 40: Train Loss: 0.5714, Val AUC: 0.737, ACC: 0.703, MAE: 0.384
Epoch 41: Train Loss: 0.5716, Val AUC: 0.737, ACC: 0.703, MAE: 0.385
Epoch 42: Train Loss: 0.5717, Val AUC: 0.735, ACC: 0.702, MAE: 0.383
Epoch 43: Train Loss: 0.5704, Val AUC: 0.740, ACC: 0.704, MAE: 0.388 ✓ New best AUC
Epoch 44: Train Loss: 0.5702, Val AUC: 0.738, ACC: 0.705, MAE: 0.379
Epoch 45: Train Loss: 0.5698, Val AUC: 0.741, ACC: 0.705, MAE: 0.385 ✓ New best AUC
Epoch 46: Train Loss: 0.5691, Val AUC: 0.740, ACC: 0.705, MAE: 0.385
Epoch 47: Train Loss: 0.5685, Val AUC: 0.739, ACC: 0.704, MAE: 0.385
Epoch 48: Train Loss: 0.5675, Val AUC: 0.737, ACC: 0.703, MAE: 0.381
Epoch 49: Train Loss: 0.5685, Val AUC: 0.741, ACC: 0.704, MAE: 0.384 ✓ New best AUC: 0.7407
Epoch 50: Train Loss: 0.5673, Val AUC: 0.736, ACC: 0.701, MAE: 0.384
Epoch 51: Train Loss: 0.5671, Val AUC: 0.737, ACC: 0.703, MAE: 0.381
Epoch 52: Train Loss: 0.5680, Val AUC: 0.740, ACC: 0.704, MAE: 0.381
Epoch 53: Train Loss: 0.5663, Val AUC: 0.741, ACC: 0.705, MAE: 0.379
Epoch 54: Train Loss: 0.5657, Val AUC: 0.739, ACC: 0.703, MAE: 0.383
Epoch 55: Train Loss: 0.5662, Val AUC: 0.738, ACC: 0.704, MAE: 0.381
Epoch 56: Train Loss: 0.5654, Val AUC: 0.739, ACC: 0.704, MAE: 0.384
Epoch 57: Train Loss: 0.5651, Val AUC: 0.740, ACC: 0.704, MAE: 0.379
Epoch 58: Train Loss: 0.5651, Val AUC: 0.738, ACC: 0.702, MAE: 0.385
Epoch 59: Train Loss: 0.5641, Val AUC: 0.740, ACC: 0.704, MAE: 0.381
Early stopping after 10 epochs
```

**关键观察**:
- **损失下降趋势**: 从0.5773稳定下降到0.5641 (下降2.3%)
- **AUC提升过程**: 4次刷新最佳记录 (Epoch 38→43→45→49)
- **性能稳定性**: 最后10轮AUC在0.737-0.741之间波动，变异系数小
- **早停触发**: 连续10轮无改善后正确触发早停
- **无过拟合**: 训练损失持续下降，验证性能保持稳定

**结论**: 
✅ 基线配置工作正常，为后续实验提供了可靠的对比基准

---

---

## 实验计划与执行

### 实验2: 模型容量扩展 (进行中)
**目标**: 验证更大模型是否能提升性能
**假设**: 增加模型维度能够提升AUC到0.75+
**配置变更**: d_model: 128→256, n_heads: 8→16
**预期结果**:
- AUC > 0.75
- 训练仍然稳定
- 训练时间增加30-50%

**执行命令**:
```bash
nohup python anomaly_aware_kt/scripts/run_stage1_only.py \
    --dataset assist17 \
    --auto_config \
    --d_model 256 \
    --n_heads 16 \
    > nohup_exp2.out 2>&1 &
```

**监控方法**:
```bash
# 查看训练进度
tail -f nohup_exp2.out

# 查看GPU使用情况
nvidia-smi

# 查看进程状态
ps aux | grep python
```

### 实验3: 层数扩展 (计划中)
**目标**: 测试更深网络的效果
**配置**: `--n_layers 4`
**状态**: 等待实验2完成

### 实验4: 学习率调优 (计划中)
**目标**: 测试更小学习率的效果
**配置**: `--learning_rate 0.0005`
**状态**: 等待实验2完成

### 实验5: 课程学习方法 (开发中)
**目标**: 实现并测试课程学习异常检测
**状态**: 需要先实现BaselineAnomalyGenerator

---

## 实验执行时间表

**今天**:
- [x] 完成实验1基线测试
- [ ] 启动实验2 (模型扩展)
- [ ] 开始实现BaselineAnomalyGenerator

**本周**:
- [ ] 完成实验2-4的对比测试
- [ ] 实现课程学习核心模块
- [ ] 开始文献调研

**下周**:
- [ ] 课程学习方法实验
- [ ] 跨数据集验证
- [ ] 撰写初步结果报告
