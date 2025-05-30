"""
Anomaly-Aware Knowledge Tracing Package

This package implements anomaly detection and integration for knowledge tracing systems.
"""

from .generator import AnomalyGenerator
from .detector import CausalAnomalyDetector
from .model import AnomalyAwareDTransformer
from .trainer import AnomalyDetectorTrainer, KnowledgeTracingTrainer
from .unified_trainer import UnifiedAnomalyTrainer
from .training_strategies import StrategyFactory
from .evaluator import AnomalyEvaluator, KTEvaluator

__version__ = "0.2.0"
__author__ = "Your Name"

__all__ = [
    "AnomalyGenerator",
    "CausalAnomalyDetector",
    "AnomalyAwareDTransformer",
    "AnomalyDetectorTrainer",
    "KnowledgeTracingTrainer",
    "UnifiedAnomalyTrainer",
    "StrategyFactory",
    "AnomalyEvaluator",
    "KTEvaluator",
]