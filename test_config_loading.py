#!/usr/bin/env python3
"""
测试配置文件加载功能
"""

import sys
import os
import yaml

def load_config(config_path: str):
    """加载YAML配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print(f"📄 已加载配置文件: {config_path}")
    return config

def auto_detect_config(dataset: str) -> str:
    """根据数据集自动检测配置文件"""
    config_dir = 'anomaly_aware_kt/configs'
    config_file = f"{dataset}_baseline.yaml"
    config_path = os.path.join(config_dir, config_file)

    if os.path.exists(config_path):
        return config_path
    else:
        print(f"⚠️  未找到数据集 {dataset} 的默认配置文件: {config_path}")
        return None

def test_config_loading():
    """测试配置文件加载"""
    print("🧪 测试配置文件加载功能")
    print("=" * 50)
    
    # 测试1: 自动检测配置文件
    print("测试1: 自动检测配置文件")
    config_path = auto_detect_config('assist17')
    print(f"  检测到的配置文件: {config_path}")
    
    if config_path and os.path.exists(config_path):
        print("  ✅ 配置文件存在")
        
        # 测试2: 加载配置文件
        print("\n测试2: 加载配置文件")
        try:
            config = load_config(config_path)
            print(f"  ✅ 配置加载成功，包含 {len(config)} 个参数")
            print("  主要参数:")
            for key in ['dataset', 'd_model', 'n_heads', 'batch_size', 'learning_rate']:
                if key in config:
                    print(f"    {key}: {config[key]}")
        except Exception as e:
            print(f"  ❌ 配置加载失败: {e}")
            return
        
        # 测试3: 参数合并
        print("\n测试3: 参数合并")
        try:
            parser = create_parser()
            # 模拟命令行参数
            test_args = parser.parse_args(['--dataset', 'assist17', '--auto_config'])
            print(f"  原始命令行参数 d_model: {test_args.d_model}")
            
            # 合并配置
            merged_args = merge_config_with_args(config, test_args)
            print(f"  合并后 d_model: {merged_args.d_model}")
            print(f"  合并后 dataset: {merged_args.dataset}")
            print(f"  合并后 batch_size: {merged_args.batch_size}")
            print("  ✅ 参数合并成功")
            
        except Exception as e:
            print(f"  ❌ 参数合并失败: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # 测试4: 参数覆盖
        print("\n测试4: 参数覆盖测试")
        try:
            # 模拟用户提供了d_model参数
            test_args_override = parser.parse_args(['--dataset', 'assist17', '--auto_config', '--d_model', '256'])
            merged_args_override = merge_config_with_args(config, test_args_override)
            
            print(f"  配置文件 d_model: {config.get('d_model', 'N/A')}")
            print(f"  命令行 d_model: {test_args_override.d_model}")
            print(f"  最终 d_model: {merged_args_override.d_model}")
            
            if merged_args_override.d_model == 256:
                print("  ✅ 命令行参数正确覆盖了配置文件")
            else:
                print("  ❌ 参数覆盖失败")
                
        except Exception as e:
            print(f"  ❌ 参数覆盖测试失败: {e}")
            
    else:
        print("  ❌ 配置文件不存在")

def test_all_datasets():
    """测试所有数据集的配置文件"""
    print("\n🧪 测试所有数据集配置文件")
    print("=" * 50)
    
    datasets = ['assist17', 'assist09', 'statics', 'algebra05']
    
    for dataset in datasets:
        print(f"\n测试 {dataset}:")
        config_path = auto_detect_config(dataset)
        if config_path and os.path.exists(config_path):
            try:
                config = load_config(config_path)
                print(f"  ✅ {dataset} 配置加载成功")
                print(f"    d_model: {config.get('d_model', 'N/A')}")
                print(f"    batch_size: {config.get('batch_size', 'N/A')}")
                print(f"    use_cl: {config.get('use_cl', 'N/A')}")
            except Exception as e:
                print(f"  ❌ {dataset} 配置加载失败: {e}")
        else:
            print(f"  ❌ {dataset} 配置文件不存在")

if __name__ == "__main__":
    test_config_loading()
    test_all_datasets()
    
    print("\n🎯 如果所有测试都通过，您可以运行:")
    print("python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config")
