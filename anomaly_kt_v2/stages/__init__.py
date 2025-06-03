"""
训练阶段模块

包含四个主要训练阶段：
1. Stage 1: 基线模型训练
2. Stage 2: 异常分类器训练  
3. Stage 3: 异常感知知识追踪训练
4. Stage 4: 性能评估与对比
"""

from .stage1_baseline import train_baseline_model
from .stage2_anomaly_classifier import train_anomaly_classifier
from .stage3_anomaly_aware_kt import train_anomaly_aware_kt

__all__ = [
    'train_baseline_model',
    'train_anomaly_classifier',
    'train_anomaly_aware_kt'
]
