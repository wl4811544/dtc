"""
Anomaly-Aware Knowledge Tracing v2

基于DTransformer的异常感知知识追踪项目重构版本

项目结构：
- Stage 1: 基线模型训练
- Stage 2: 异常分类器训练  
- Stage 3: 异常感知知识追踪训练
- Stage 4: 性能评估与对比

作者: 研究团队
版本: 2.0
"""

__version__ = "2.0.0"
__author__ = "Research Team"

# 项目阶段定义
STAGES = {
    1: "基线模型训练",
    2: "异常分类器训练", 
    3: "异常感知知识追踪训练",
    4: "性能评估与对比"
}

# 支持的数据集
SUPPORTED_DATASETS = [
    "assist09",
    "assist17", 
    "algebra05",
    "bridge06"
]

# 默认配置
DEFAULT_CONFIG = {
    "data_dir": "data",
    "output_dir": "output",
    "device": "cuda",
    "batch_size": 16,
    "test_batch_size": 256
}
