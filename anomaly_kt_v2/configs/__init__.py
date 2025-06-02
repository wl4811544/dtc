"""
配置管理模块

提供自动配置加载和参数管理功能
"""

import os
import yaml
import argparse
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> argparse.Namespace:
    """将配置文件参数与命令行参数合并
    
    Args:
        config: 配置文件字典
        args: 命令行参数
        
    Returns:
        合并后的参数对象
    """
    # 命令行参数优先级更高
    for key, value in config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)
    
    return args


def get_auto_config_path(dataset: str, stage: str) -> str:
    """获取自动配置文件路径
    
    Args:
        dataset: 数据集名称
        stage: 训练阶段
        
    Returns:
        配置文件路径
    """
    config_dir = os.path.dirname(__file__)
    config_file = f"{dataset}_{stage}.yaml"
    return os.path.join(config_dir, config_file)


def load_auto_config(dataset: str, stage: str) -> Optional[Dict[str, Any]]:
    """自动加载配置文件
    
    Args:
        dataset: 数据集名称
        stage: 训练阶段
        
    Returns:
        配置字典，如果文件不存在则返回None
    """
    config_path = get_auto_config_path(dataset, stage)
    
    if os.path.exists(config_path):
        return load_config(config_path)
    else:
        print(f"⚠️ 自动配置文件不存在: {config_path}")
        return None
