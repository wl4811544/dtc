"""
Pipeline stages for anomaly-aware knowledge tracing

This module contains the individual stages of the training pipeline:
- Stage 1: Baseline model training
- Stage 2: Anomaly detector training
- Stage 3: Anomaly-aware model training
- Stage 4: Model evaluation
"""

from .stage1_baseline import train_baseline_model
from .stage2_curriculum_anomaly import train_curriculum_anomaly_detector, test_curriculum_components

__all__ = [
    'train_baseline_model',
    'train_curriculum_anomaly_detector',
    'test_curriculum_components'
]
