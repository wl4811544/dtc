"""
核心模块

包含通用工具函数和基类
"""

from .common import (
    prepare_data,
    setup_output_directory, 
    save_config,
    print_stage_header,
    validate_model_path,
    load_model_with_compatibility,
    StageConfig,
    BaseStage,
    merge_config_with_args,
    print_training_summary
)

__all__ = [
    'prepare_data',
    'setup_output_directory',
    'save_config', 
    'print_stage_header',
    'validate_model_path',
    'load_model_with_compatibility',
    'StageConfig',
    'BaseStage',
    'merge_config_with_args',
    'print_training_summary'
]
