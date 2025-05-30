"""
Common utilities and configurations for pipeline stages
"""

import os
import sys
import torch
import tomlkit
import yaml
from datetime import datetime
from typing import Dict, Any, Tuple

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from DTransformer.data import KTData


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        else:
            raise ValueError("Only YAML config files are supported")
    return config


def prepare_data(dataset_name: str, data_dir: str, batch_size: int, test_batch_size: int) -> Tuple:
    """准备数据集
    
    Args:
        dataset_name: 数据集名称
        data_dir: 数据目录
        batch_size: 训练批次大小
        test_batch_size: 测试批次大小
        
    Returns:
        Tuple of (train_data, val_data, test_data, dataset_config)
    """
    # 加载数据集配置
    datasets = tomlkit.load(open(os.path.join(data_dir, 'datasets.toml')))
    dataset_config = datasets[dataset_name]

    # 创建数据加载器
    train_data = KTData(
        os.path.join(data_dir, dataset_config['train']),
        dataset_config['inputs'],
        batch_size=batch_size,
        shuffle=True
    )

    val_data = KTData(
        os.path.join(data_dir, dataset_config.get('valid', dataset_config['test'])),
        dataset_config['inputs'],
        batch_size=test_batch_size
    )

    test_data = KTData(
        os.path.join(data_dir, dataset_config['test']),
        dataset_config['inputs'],
        batch_size=test_batch_size
    )

    return train_data, val_data, test_data, dataset_config


def setup_output_directory(output_dir: str = None, dataset_name: str = None) -> str:
    """设置输出目录
    
    Args:
        output_dir: 指定的输出目录，如果为None则自动生成
        dataset_name: 数据集名称，用于自动生成目录名
        
    Returns:
        输出目录路径
    """
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"output/{dataset_name}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_config(config: Dict[str, Any], output_dir: str) -> str:
    """保存配置到文件
    
    Args:
        config: 配置字典
        output_dir: 输出目录
        
    Returns:
        配置文件路径
    """
    config_save_path = os.path.join(output_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    return config_save_path


def print_stage_header(stage_name: str, stage_number: int = None):
    """打印阶段标题
    
    Args:
        stage_name: 阶段名称
        stage_number: 阶段编号
    """
    header = f"PHASE {stage_number}: {stage_name}" if stage_number else stage_name
    print("\n" + "="*60)
    print(header)
    print("="*60)


def validate_model_path(path: str, model_type: str) -> bool:
    """验证模型文件路径
    
    Args:
        path: 模型文件路径
        model_type: 模型类型（用于错误信息）
        
    Returns:
        是否验证通过
    """
    if not path:
        print(f"ERROR: {model_type} model path is required")
        return False
        
    if not os.path.exists(path):
        print(f"ERROR: {model_type} model file not found: {path}")
        return False
        
    print(f"✓ {model_type} model found: {path}")
    return True


class StageConfig:
    """阶段配置基类"""
    
    def __init__(self, args, dataset_config):
        self.args = args
        self.dataset_config = dataset_config
        self.device = args.device
        self.output_dir = args.output_dir
        
    def get_model_save_path(self, stage_name: str) -> str:
        """获取模型保存路径"""
        return os.path.join(self.output_dir, stage_name, 'best_model.pt')
        
    def print_config(self):
        """打印配置信息"""
        print("Configuration:")
        for key, value in vars(self.args).items():
            print(f"  {key}: {value}")


class BaseStage:
    """阶段基类"""
    
    def __init__(self, config: StageConfig):
        self.config = config
        self.args = config.args
        self.dataset_config = config.dataset_config
        self.device = config.device
        self.output_dir = config.output_dir
        
    def run(self, *args, **kwargs):
        """运行阶段，子类需要实现"""
        raise NotImplementedError("Subclasses must implement run method")
        
    def print_header(self, stage_name: str, stage_number: int = None):
        """打印阶段标题"""
        print_stage_header(stage_name, stage_number)
        
    def print_results(self, metrics: Dict[str, float], metric_name: str = "AUC"):
        """打印结果"""
        print(f"\nTraining completed!")
        if metric_name.lower() in metrics:
            print(f"Best {metric_name}: {metrics[metric_name.lower()]:.4f}")
