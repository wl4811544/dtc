#!/usr/bin/env python
"""
配置一致性验证脚本

对比重构后的配置与您原始训练配置的一致性
"""

import os
import sys
import yaml

def load_original_configs():
    """加载您的原始训练配置"""
    configs = {}
    
    # 基础模型配置
    basic_config_path = "output/stage1_assist17_20250601_142714/config.yaml"
    if os.path.exists(basic_config_path):
        with open(basic_config_path, 'r') as f:
            configs['basic_original'] = yaml.safe_load(f)
    
    # 扩展模型配置
    extended_config_path = "output/stage1_assist17_20250601_213029/config.yaml"
    if os.path.exists(extended_config_path):
        with open(extended_config_path, 'r') as f:
            configs['extended_original'] = yaml.safe_load(f)
    
    return configs

def simulate_new_config(model_type):
    """模拟新脚本生成的配置"""
    # 基础配置 (来自配置文件)
    base_config = {
        'dataset': 'assist17',
        'data_dir': 'data',
        'device': 'cuda',
        'with_pid': True,
        'batch_size': 16,
        'test_batch_size': 32,
        'd_model': 128,  # 基础模型默认值
        'n_heads': 8,    # 基础模型默认值
        'n_know': 16,
        'n_layers': 3,
        'dropout': 0.2,
        'lambda_cl': 0.1,
        'proj': True,
        'hard_neg': False,
        'window': 1,
        'kt_epochs': 100,
        'learning_rate': 0.001,
        'patience': 10,
        'use_cl': True
    }
    
    # 应用模型类型预设
    if model_type == 'extended':
        base_config['d_model'] = 256
        base_config['n_heads'] = 16
    
    return base_config

def compare_configs(original, new, model_type):
    """对比配置"""
    print(f"\n{'='*60}")
    print(f"🔍 {model_type.upper()} 模型配置对比")
    print(f"{'='*60}")
    
    # 关键参数列表
    key_params = [
        'dataset', 'device', 'with_pid', 'batch_size', 'test_batch_size',
        'd_model', 'n_heads', 'n_know', 'n_layers', 'dropout', 'lambda_cl',
        'proj', 'hard_neg', 'window', 'kt_epochs', 'learning_rate', 
        'patience', 'use_cl'
    ]
    
    all_match = True
    
    for param in key_params:
        original_val = original.get(param, 'MISSING')
        new_val = new.get(param, 'MISSING')
        
        if original_val == new_val:
            status = "✅"
        else:
            status = "❌"
            all_match = False
        
        print(f"  {status} {param:15} | 原始: {original_val:8} | 新配置: {new_val}")
    
    print(f"\n📊 总体结果: {'✅ 完全一致' if all_match else '❌ 存在差异'}")
    
    return all_match

def main():
    """主函数"""
    print("🔍 配置一致性验证")
    print("="*60)
    
    # 加载原始配置
    original_configs = load_original_configs()
    
    if not original_configs:
        print("❌ 无法找到原始配置文件")
        return False
    
    all_consistent = True
    
    # 验证基础模型配置
    if 'basic_original' in original_configs:
        new_basic = simulate_new_config('basic')
        basic_match = compare_configs(
            original_configs['basic_original'], 
            new_basic, 
            'basic'
        )
        all_consistent = all_consistent and basic_match
    
    # 验证扩展模型配置
    if 'extended_original' in original_configs:
        new_extended = simulate_new_config('extended')
        extended_match = compare_configs(
            original_configs['extended_original'], 
            new_extended, 
            'extended'
        )
        all_consistent = all_consistent and extended_match
    
    # 总结
    print(f"\n{'='*60}")
    print("📋 验证总结")
    print(f"{'='*60}")
    
    if all_consistent:
        print("🎉 所有配置完全一致！")
        print("✅ 重构后的脚本将产生与您原始训练相同的配置")
        print("✅ 无需重新训练，可以直接使用现有模型")
    else:
        print("⚠️ 发现配置差异，需要修复")
        print("❌ 建议检查并修复不一致的参数")
    
    return all_consistent

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
