"""
Pipeline stages for anomaly-aware knowledge tracing

This module contains the individual stages of the training pipeline:
- Stage 1: Baseline model training
- Stage 2: Anomaly detector training
- Stage 3: Anomaly-aware model training
- Stage 4: Model evaluation
"""

from .stage1_baseline import train_baseline_model

__all__ = [
    'train_baseline_model'
]
