#!/usr/bin/env python3
"""
简化的配置文件测试，不依赖torch等库
"""

import os
import yaml

def test_config_files():
    """测试配置文件是否存在和格式正确"""
    print("🧪 测试配置文件")
    print("=" * 50)
    
    config_dir = "anomaly_aware_kt/configs"
    datasets = ['assist17', 'assist09', 'statics', 'algebra05']
    
    for dataset in datasets:
        config_file = f"{dataset}_baseline.yaml"
        config_path = os.path.join(config_dir, config_file)
        
        print(f"\n测试 {dataset}:")
        print(f"  配置文件: {config_path}")
        
        if os.path.exists(config_path):
            print("  ✅ 文件存在")
            
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                print(f"  ✅ YAML格式正确，包含 {len(config)} 个参数")
                
                # 检查关键参数
                key_params = ['dataset', 'd_model', 'n_heads', 'batch_size', 'learning_rate']
                print("  关键参数:")
                for param in key_params:
                    if param in config:
                        print(f"    {param}: {config[param]}")
                    else:
                        print(f"    {param}: ❌ 缺失")
                
                # 检查数据集匹配
                if config.get('dataset') == dataset:
                    print("  ✅ 数据集名称匹配")
                else:
                    print(f"  ❌ 数据集名称不匹配: 期望 {dataset}, 实际 {config.get('dataset')}")
                    
            except yaml.YAMLError as e:
                print(f"  ❌ YAML格式错误: {e}")
            except Exception as e:
                print(f"  ❌ 读取失败: {e}")
        else:
            print("  ❌ 文件不存在")

def test_parameter_priority():
    """测试参数优先级逻辑"""
    print("\n🧪 测试参数优先级逻辑")
    print("=" * 50)
    
    # 模拟配置文件参数
    config = {
        'dataset': 'assist17',
        'd_model': 128,
        'batch_size': 16,
        'learning_rate': 0.001
    }
    
    # 模拟命令行参数（默认值）
    cmd_defaults = {
        'd_model': 128,  # 默认值
        'batch_size': 32,  # 默认值
        'learning_rate': 0.001,  # 默认值
        'n_heads': 8  # 默认值
    }
    
    # 模拟用户提供的命令行参数
    user_args = {
        'd_model': 256,  # 用户覆盖
        'batch_size': 16,  # 等于配置文件值
        'learning_rate': 0.001,  # 等于默认值
        'n_heads': 8  # 等于默认值
    }
    
    print("配置文件参数:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    print("\n命令行默认值:")
    for k, v in cmd_defaults.items():
        print(f"  {k}: {v}")
    
    print("\n用户提供的参数:")
    for k, v in user_args.items():
        print(f"  {k}: {v}")
    
    # 合并逻辑
    final_params = {}
    
    # 1. 先设置配置文件的值
    for k, v in config.items():
        final_params[k] = v
    
    # 2. 用命令行参数覆盖（如果用户显式提供了）
    for k, v in user_args.items():
        default_val = cmd_defaults.get(k)
        if v != default_val:  # 用户显式提供了不同于默认值的参数
            final_params[k] = v
        elif k not in final_params:  # 配置文件中没有这个参数
            final_params[k] = v
    
    print("\n最终参数:")
    for k, v in final_params.items():
        print(f"  {k}: {v}")
    
    # 验证优先级
    print("\n优先级验证:")
    if final_params['d_model'] == 256:
        print("  ✅ d_model: 命令行参数正确覆盖配置文件")
    else:
        print("  ❌ d_model: 优先级错误")
    
    if final_params['batch_size'] == 16:
        print("  ✅ batch_size: 配置文件值正确保留")
    else:
        print("  ❌ batch_size: 应该使用配置文件值")

def show_usage_examples():
    """显示使用示例"""
    print("\n📋 使用示例")
    print("=" * 50)
    
    examples = [
        "# 1. 使用自动配置",
        "python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config",
        "",
        "# 2. 手动指定配置文件", 
        "python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --config anomaly_aware_kt/configs/assist17_baseline.yaml",
        "",
        "# 3. 配置文件 + 参数覆盖",
        "python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config --d_model 256",
        "",
        "# 4. 多个参数覆盖",
        "python anomaly_aware_kt/scripts/run_stage1_only.py --dataset assist17 --auto_config --d_model 256 --learning_rate 0.0005",
        "",
        "# 5. 小数据集",
        "python anomaly_aware_kt/scripts/run_stage1_only.py --dataset statics --auto_config",
    ]
    
    for example in examples:
        print(example)

if __name__ == "__main__":
    test_config_files()
    test_parameter_priority()
    show_usage_examples()
    
    print("\n🎯 如果配置文件测试通过，您可以直接运行训练命令！")
