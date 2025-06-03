"""
异常感知知识追踪模块

第三阶段：将基线知识追踪模型与异常检测器融合，
实现异常感知的知识追踪，目标提升AUC 0.05-0.1。
"""

from .fusion import AnomalyAwareFusion, AnomalyWeightAdjuster
from .model import AnomalyAwareKT
from .trainer import AnomalyAwareTrainer

__all__ = [
    'AnomalyAwareFusion',
    'AnomalyWeightAdjuster', 
    'AnomalyAwareKT',
    'AnomalyAwareTrainer'
]
