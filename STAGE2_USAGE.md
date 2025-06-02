# 第二阶段使用指南：课程学习异常检测训练

## 🎯 概述

第二阶段实现了基于课程学习的异常检测训练，这是我们研究设计的核心创新。

### 核心特性
- ✅ **时序因果约束下的课程学习**
- ✅ **认知理论驱动的异常分级**
- ✅ **4级难度体系**
- ✅ **自适应课程调度**
- ✅ **教育异常与基线异常分离**

## 📁 文件结构

```
anomaly_aware_kt/
├── anomaly_kt/
│   ├── curriculum_learning/          # 课程学习核心模块
│   │   ├── __init__.py
│   │   ├── baseline_generator.py     # 基线异常生成器
│   │   ├── curriculum_generator.py   # 课程学习异常生成器
│   │   ├── difficulty_estimator.py   # 难度评估器
│   │   ├── curriculum_scheduler.py   # 课程调度器
│   │   └── curriculum_trainer.py     # 课程学习训练器
│   └── stages/
│       ├── stage2_curriculum_anomaly.py  # 第二阶段核心执行模块
│       └── ...
├── scripts/
│   ├── run_stage2_curriculum.py     # 第二阶段调用脚本
│   └── ...
├── configs/
│   ├── assist17_curriculum.yaml     # ASSIST17课程学习配置
│   └── ...
└── test_stage2_components.py        # 组件测试脚本
```

## 🚀 快速开始

### 1. 测试组件

首先测试所有组件是否正常工作：

```bash
# 测试组件导入和基本功能
python test_stage2_components.py
```

### 2. 准备基线模型

确保您已经完成第一阶段训练：

```bash
# 如果还没有运行第一阶段
python anomaly_aware_kt/scripts/run_stage1_only.py \
  --dataset assist17 \
  --auto_config
```

### 3. 运行第二阶段训练

#### 方式1：使用自动配置（推荐）

```bash
python anomaly_aware_kt/scripts/run_stage2_curriculum.py \
  --dataset assist17 \
  --baseline_model_path output/stage1_assist17_xxx/baseline/best_model.pt \
  --auto_config
```

#### 方式2：使用自定义配置

```bash
python anomaly_aware_kt/scripts/run_stage2_curriculum.py \
  --dataset assist17 \
  --baseline_model_path output/stage1_assist17_xxx/baseline/best_model.pt \
  --config anomaly_aware_kt/configs/assist17_curriculum.yaml
```

#### 方式3：纯命令行参数

```bash
python anomaly_aware_kt/scripts/run_stage2_curriculum.py \
  --dataset assist17 \
  --baseline_model_path output/stage1_assist17_xxx/baseline/best_model.pt \
  --curriculum_strategy hybrid \
  --curriculum_epochs 100 \
  --anomaly_ratio 0.1 \
  --detector_hidden_dim 256 \
  --learning_rate 1e-4
```

### 4. 测试模式（推荐先运行）

```bash
# 只测试组件，不进行实际训练
python anomaly_aware_kt/scripts/run_stage2_curriculum.py \
  --dataset assist17 \
  --baseline_model_path output/stage1_assist17_xxx/baseline/best_model.pt \
  --auto_config \
  --dry_run
```

## 📋 参数说明

### 必需参数

- `--dataset`: 数据集名称 (assist09, assist17, algebra05, statics)
- `--baseline_model_path`: 第一阶段训练的基线模型路径

### 课程学习参数

- `--curriculum_strategy`: 课程调度策略
  - `performance_driven`: 性能驱动
  - `time_driven`: 时间驱动
  - `hybrid`: 混合策略（推荐）
- `--curriculum_epochs`: 课程学习训练轮数 (默认: 100)
- `--anomaly_ratio`: 异常比例 (默认: 0.1)
- `--baseline_ratio`: 基线异常比例 (默认: 0.05)
- `--max_patience`: 课程推进最大耐心值 (默认: 5)

### 异常检测器参数

- `--detector_hidden_dim`: 隐藏层维度 (默认: 256)
- `--detector_num_layers`: 层数 (默认: 3)
- `--detector_dropout`: Dropout率 (默认: 0.3)
- `--detector_window_size`: 因果窗口大小 (默认: 10)

### 训练参数

- `--learning_rate`: 学习率 (默认: 1e-4)
- `--patience`: 早停耐心值 (默认: 10)
- `--batch_size`: 训练批次大小 (默认: 16)

## 📊 输出说明

### 训练过程输出

```
第二阶段：课程学习异常检测训练
============================================================
✓ 基线模型文件存在: output/stage1_xxx/baseline/best_model.pt
📄 已加载配置文件: anomaly_aware_kt/configs/assist17_curriculum.yaml
🔄 合并配置文件和命令行参数...
✅ 参数合并完成，命令行参数优先级更高
📁 输出目录: output/stage2_assist17_20241201_143022

PHASE 2: Curriculum Learning Anomaly Detection Training
============================================================
📥 加载基线模型: output/stage1_xxx/baseline/best_model.pt
✓ 模型权重加载成功 (来自checkpoint)
  基线模型AUC: 0.7407

🔍 创建异常检测器...
✓ 异常检测器创建成功
  输入维度: 128
  隐藏维度: 256
  层数: 3

🎓 初始化课程学习组件...
✓ 课程调度器: hybrid 策略
✓ 异常生成器: assist17 数据集优化
✓ 难度评估器: 启用

开始课程学习训练...
============================================================
Epoch   1 | Phase 1/4 | Train Loss: 0.6234 | Val AUC: 0.7123 | Val F1: 0.6789
Epoch   5 | Phase 1/4 | Train Loss: 0.5891 | Val AUC: 0.7345 | Val F1: 0.7012
  🎓 进入新阶段: Phase 2
Epoch  15 | Phase 2/4 | Train Loss: 0.5567 | Val AUC: 0.7523 | Val F1: 0.7234
...
```

### 输出文件

```
output/stage2_assist17_xxx/
├── config.yaml                      # 保存的配置
├── curriculum_anomaly/               # 异常检测器模型
│   ├── best_model.pt                # 最佳模型
│   ├── training_log.txt             # 训练日志
│   └── checkpoints/                 # 检查点
├── curriculum_training_summary.json # 训练总结
└── curriculum_training.log          # 详细日志
```

## 🎓 课程学习阶段说明

### Phase 1: 建立信心 (Epochs 1-15)
- **难度级别**: Level 1 (简单异常)
- **目标**: 检测明显的连续错误、完全随机答案
- **预期Recall**: >90%
- **重点**: 建立检测器的基础能力

### Phase 2: 渐进复杂 (Epochs 16-35)
- **难度级别**: Level 1 + Level 2 (中等异常)
- **目标**: 引入模式性异常、局部突发异常
- **预期Recall**: >80%
- **重点**: 逐步增加复杂度

### Phase 3: 高级模式 (Epochs 36-60)
- **难度级别**: Level 1 + Level 2 + Level 3 (困难异常)
- **目标**: 微妙的能力不匹配、复杂作弊模式
- **预期Recall**: >70%
- **重点**: 学习复杂异常模式

### Phase 4: 专家级别 (Epochs 61-100)
- **难度级别**: Level 1-4 (所有级别)
- **目标**: 高度伪装的异常、智能作弊行为
- **预期Recall**: >60%
- **重点**: 最高难度异常检测

## 🔧 故障排除

### 常见问题

1. **导入错误**
   ```bash
   # 检查Python路径
   python -c "import sys; print(sys.path)"
   
   # 测试组件导入
   python test_stage2_components.py
   ```

2. **基线模型加载失败**
   ```bash
   # 检查模型文件
   ls -la output/stage1_xxx/baseline/best_model.pt
   
   # 验证模型格式
   python -c "import torch; print(torch.load('path/to/model.pt').keys())"
   ```

3. **内存不足**
   ```bash
   # 减少批次大小
   --batch_size 8 --test_batch_size 16
   
   # 减少检测器维度
   --detector_hidden_dim 128
   ```

4. **训练不稳定**
   ```bash
   # 降低学习率
   --learning_rate 5e-5
   
   # 增加耐心值
   --patience 15 --max_patience 8
   ```

## 📈 预期结果

基于我们的设计，预期看到：

- **训练稳定性提升**: 损失曲线更平滑，减少震荡
- **检测性能改善**: AUC从基线的0.7407提升到0.75+
- **阶段性进步**: 4个阶段的清晰性能提升轨迹
- **异常分级有效性**: 不同难度级别的检测率差异明显

## 🔬 实验分析

### 关键指标监控

1. **课程进度**: 阶段转换的时机和条件
2. **难度分级**: 各级别异常的检测率
3. **训练稳定性**: 损失方差和收敛速度
4. **最终性能**: 与基线方法的对比

### 日志分析

```bash
# 查看课程转换
grep "进入新阶段" output/stage2_xxx/curriculum_training.log

# 分析性能趋势
grep "Val AUC" output/stage2_xxx/curriculum_training.log

# 检查异常生成统计
grep "异常生成" output/stage2_xxx/curriculum_training.log
```

## 💡 下一步

完成第二阶段后，您可以：

1. **分析结果**: 检查课程学习的效果
2. **对比实验**: 与传统训练方法对比
3. **参数调优**: 优化课程配置
4. **扩展实验**: 在其他数据集上验证

---

**注意**: 这是我们研究设计的核心创新实现，包含了时序因果约束、认知理论驱动等关键技术创新。
