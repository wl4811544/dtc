#!/usr/bin/env python3
"""
完美研究计划的配置文件生成器

为所有实验配置生成对应的配置文件
"""

import os
import yaml
from pathlib import Path


def create_base_config():
    """创建基础配置模板"""
    return {
        'dataset': {
            'name': 'assist17',
            'features': ['q', 's', 'pid', 'it', 'at'],
            'has_temporal': True,
            'has_pid': True,
            'complexity_factor': 1.0,
            'max_sequence_length': 200
        },
        'model': {
            'd_model': 128,
            'n_heads': 8,
            'n_layers': 3,
            'dropout': 0.2,
            'lambda_cl': 0.1,
            'n_know': 16
        },
        'training': {
            'batch_size': 16,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'patience': 10
        },
        'curriculum': {
            'strategy': 'hybrid',
            'curriculum_epochs': 100,
            'anomaly_ratio': 0.1,
            'baseline_ratio': 0.05,
            'max_patience': 5
        },
        'detector': {
            'hidden_dim': 256,
            'num_layers': 3,
            'dropout': 0.3,
            'window_size': 10,
            'learning_rate': 1e-4
        }
    }


def create_expanded_config():
    """创建扩展配置"""
    config = create_base_config()
    config['model'].update({
        'd_model': 256,
        'n_heads': 16,
        'n_layers': 3
    })
    config['detector'].update({
        'hidden_dim': 512,
        'num_layers': 4,
        'learning_rate': 5e-5
    })
    config['training'].update({
        'batch_size': 12,  # 减少批次大小适应更大模型
        'patience': 15
    })
    config['curriculum'].update({
        'curriculum_epochs': 120,
        'max_patience': 8
    })
    return config


def create_deep_config():
    """创建深度配置"""
    config = create_base_config()
    config['model'].update({
        'd_model': 128,
        'n_heads': 8,
        'n_layers': 4  # 增加层数
    })
    config['detector'].update({
        'hidden_dim': 256,
        'num_layers': 4,
        'learning_rate': 8e-5
    })
    config['training'].update({
        'patience': 12
    })
    config['curriculum'].update({
        'curriculum_epochs': 110,
        'max_patience': 6
    })
    return config


def create_assist09_config():
    """创建ASSIST09配置"""
    config = create_base_config()
    config['dataset'].update({
        'name': 'assist09',
        'complexity_factor': 0.9,
        'max_sequence_length': 150
    })
    config['curriculum'].update({
        'curriculum_epochs': 80,  # 较小数据集用较少轮数
        'strategy': 'hybrid'
    })
    return config


def create_algebra05_config():
    """创建Algebra05配置"""
    config = create_base_config()
    config['dataset'].update({
        'name': 'algebra05',
        'has_temporal': False,
        'complexity_factor': 0.7,
        'max_sequence_length': 80
    })
    config['curriculum'].update({
        'curriculum_epochs': 60,  # 小数据集用更少轮数
        'strategy': 'time_driven'  # 小数据集用时间驱动
    })
    config['training'].update({
        'batch_size': 8,  # 小数据集用小批次
        'patience': 8
    })
    return config


def flatten_config_for_stage2(config):
    """将配置扁平化，便于第二阶段使用"""
    flat_config = {}
    
    # 基本参数
    flat_config.update({
        'curriculum_strategy': config['curriculum']['strategy'],
        'curriculum_epochs': config['curriculum']['curriculum_epochs'],
        'anomaly_ratio': config['curriculum']['anomaly_ratio'],
        'baseline_ratio': config['curriculum']['baseline_ratio'],
        'max_patience': config['curriculum']['max_patience'],
        
        'detector_hidden_dim': config['detector']['hidden_dim'],
        'detector_num_layers': config['detector']['num_layers'],
        'detector_dropout': config['detector']['dropout'],
        'detector_window_size': config['detector']['window_size'],
        
        'learning_rate': config['detector']['learning_rate'],
        'patience': config['training']['patience'],
        'batch_size': config['training']['batch_size'],
        'difficulty_estimation': True
    })
    
    return flat_config


def save_config(config, filename):
    """保存配置文件"""
    config_dir = Path('anomaly_aware_kt/configs')
    config_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = config_dir / filename
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✅ 已创建配置文件: {filepath}")


def main():
    """生成所有配置文件"""
    print("🔧 生成完美研究计划的配置文件...")
    
    configs = {
        # ASSIST17配置
        'assist17_base.yaml': create_base_config(),
        'assist17_expanded.yaml': create_expanded_config(),
        'assist17_deep.yaml': create_deep_config(),
        
        # 其他数据集配置
        'assist09_base.yaml': create_assist09_config(),
        'algebra05_base.yaml': create_algebra05_config(),
    }
    
    # 生成第一阶段配置文件
    for filename, config in configs.items():
        save_config(config, filename)
    
    # 生成第二阶段配置文件
    stage2_configs = {}
    for filename, config in configs.items():
        stage2_filename = filename.replace('.yaml', '_curriculum.yaml')
        stage2_config = config.copy()
        stage2_config.update(flatten_config_for_stage2(config))
        stage2_configs[stage2_filename] = stage2_config
    
    for filename, config in stage2_configs.items():
        save_config(config, filename)
    
    print(f"\n📊 总计生成 {len(configs) + len(stage2_configs)} 个配置文件")
    print("\n🎯 配置文件用途:")
    print("  第一阶段: assist17_base.yaml, assist17_expanded.yaml, assist17_deep.yaml")
    print("  第二阶段: assist17_base_curriculum.yaml, assist17_expanded_curriculum.yaml, assist17_deep_curriculum.yaml")
    print("  跨数据集: assist09_base.yaml, algebra05_base.yaml")


if __name__ == "__main__":
    main()
