# 第4阶段直接评估使用指南

本指南说明如何使用修改后的 `full_pipeline.py` 直接执行第4阶段（模型评估），跳过前面的训练阶段。

## 新增功能

### 新增参数

- `--skip_anomaly_training`: 跳过第3阶段的异常感知模型训练
- `--anomaly_path`: 指定已训练好的异常感知模型路径

### 参数验证

当使用 `--skip_anomaly_training` 时，系统会自动验证：
1. `--anomaly_path` 参数是否已提供
2. 指定的异常感知模型文件是否存在

## 使用方法

### 直接执行第4阶段的完整命令示例

```bash
python full_pipeline.py \
    --dataset assist09 \
    --skip_baseline \
    --skip_detector \
    --skip_anomaly_training \
    --baseline_path "output/assist09_20241201_120000/baseline/best_model.pt" \
    --detector_path "output/assist09_20241201_120000/detector/best_model.pt" \
    --anomaly_path "output/assist09_20241201_120000/anomaly_aware/best_model.pt" \
    --output_dir "output/evaluation_only" \
    --d_model 128 \
    --n_heads 8 \
    --n_know 16 \
    --n_layers 3 \
    --detector_d_model 128 \
    --detector_n_heads 8 \
    --detector_n_layers 2 \
    --window_size 10 \
    --anomaly_weight 0.5
```

### 必需参数说明

执行第4阶段需要以下必需参数：

#### 基本参数
- `--dataset`: 数据集名称 (assist09, assist17, algebra05, statics)
- `--skip_baseline`: 跳过第1阶段
- `--skip_detector`: 跳过第2阶段  
- `--skip_anomaly_training`: 跳过第3阶段

#### 模型路径参数
- `--baseline_path`: 基线DTransformer模型路径
- `--detector_path`: 异常检测器模型路径
- `--anomaly_path`: 异常感知模型路径

#### 模型架构参数
这些参数必须与训练时使用的参数一致：

**基线模型参数:**
- `--d_model`: 模型维度 (默认: 128)
- `--n_heads`: 注意力头数 (默认: 8)
- `--n_know`: 知识概念数 (默认: 16)
- `--n_layers`: 层数 (默认: 3)

**异常检测器参数:**
- `--detector_d_model`: 检测器模型维度 (默认: 128)
- `--detector_n_heads`: 检测器注意力头数 (默认: 8)
- `--detector_n_layers`: 检测器层数 (默认: 2)
- `--window_size`: 窗口大小 (默认: 10)

**异常感知参数:**
- `--anomaly_weight`: 异常权重 (默认: 0.5)

### 可选参数

- `--output_dir`: 输出目录 (默认: output/{dataset}_{timestamp})
- `--device`: 设备 (默认: cuda if available else cpu)
- `--test_batch_size`: 测试批次大小 (默认: 64)
- `--with_pid`: 是否使用问题ID

## 使用示例脚本

我们提供了一个示例脚本 `run_evaluation_only.py`，展示了如何使用这个功能：

```bash
python run_evaluation_only.py
```

**注意**: 运行前请修改脚本中的模型路径！

## 输出结果

第4阶段会生成以下输出：

1. **控制台输出**: 详细的评估结果比较
2. **JSON文件**: `evaluation_results.json` 包含完整的评估指标
3. **配置文件**: `config.yaml` 记录使用的参数

### 评估指标

- AUC (Area Under Curve)
- ACC (Accuracy) 
- Precision
- Recall
- F1 Score
- 性能提升百分比

## 注意事项

1. **参数一致性**: 确保模型架构参数与训练时完全一致
2. **文件路径**: 确保所有模型文件路径正确且文件存在
3. **数据集**: 确保指定的数据集与训练时使用的数据集一致
4. **设备兼容**: 确保加载模型的设备与训练时兼容

## 错误排查

### 常见错误

1. **模型文件不存在**
   ```
   ERROR: Anomaly-aware model file not found: path/to/model.pt
   ```
   解决: 检查文件路径是否正确

2. **参数不匹配**
   ```
   RuntimeError: Error(s) in loading state_dict
   ```
   解决: 确保模型架构参数与训练时一致

3. **设备不匹配**
   ```
   RuntimeError: Expected all tensors to be on the same device
   ```
   解决: 检查 `--device` 参数设置

## 完整的阶段控制

现在 `full_pipeline.py` 支持灵活的阶段控制：

- 执行所有阶段: 不使用任何 skip 参数
- 跳过第1阶段: `--skip_baseline --baseline_path path/to/baseline.pt`
- 跳过第1-2阶段: `--skip_baseline --skip_detector --baseline_path path1 --detector_path path2`
- 跳过第1-3阶段: `--skip_baseline --skip_detector --skip_anomaly_training --baseline_path path1 --detector_path path2 --anomaly_path path3`

这样可以根据需要灵活地从任何阶段开始执行流程。
