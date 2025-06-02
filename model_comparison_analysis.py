#!/usr/bin/env python3
"""
基础模型 vs 扩展模型训练对比分析

分析两个模型的训练过程和性能差异
"""

import re
import numpy as np
from pathlib import Path


def extract_training_metrics(log_file):
    """从日志文件中提取训练指标"""
    metrics = {
        'epochs': [],
        'train_loss': [],
        'val_acc': [],
        'val_auc': [],
        'val_mae': []
    }
    
    if not Path(log_file).exists():
        print(f"❌ 文件不存在: {log_file}")
        return metrics
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # 提取训练指标
    pattern = r'Train Loss: ([\d.]+)\nVal - ACC: ([\d.]+), AUC: ([\d.]+), MAE: ([\d.]+)'
    matches = re.findall(pattern, content)
    
    for i, (train_loss, val_acc, val_auc, val_mae) in enumerate(matches):
        metrics['epochs'].append(i + 1)
        metrics['train_loss'].append(float(train_loss))
        metrics['val_acc'].append(float(val_acc))
        metrics['val_auc'].append(float(val_auc))
        metrics['val_mae'].append(float(val_mae))
    
    return metrics


def analyze_training_speed(log_file):
    """分析训练速度"""
    if not Path(log_file).exists():
        return None
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # 提取训练速度信息
    speed_pattern = r'Training:.*?(\d+\.\d+)it/s'
    speeds = re.findall(speed_pattern, content)
    
    if speeds:
        speeds = [float(s) for s in speeds]
        return {
            'avg_speed': np.mean(speeds),
            'min_speed': np.min(speeds),
            'max_speed': np.max(speeds),
            'std_speed': np.std(speeds)
        }
    return None


def print_comparison_analysis():
    """打印详细的对比分析"""
    print("🔍 基础模型 vs 扩展模型训练对比分析")
    print("=" * 60)
    
    # 提取两个模型的指标
    base_metrics = extract_training_metrics('nohup.out')
    exp_metrics = extract_training_metrics('nohup_exp2.out')
    
    # 分析训练速度
    base_speed = analyze_training_speed('nohup.out')
    exp_speed = analyze_training_speed('nohup_exp2.out')
    
    print(f"\n📊 模型配置对比:")
    print(f"{'配置':<15} {'基础模型':<15} {'扩展模型':<15}")
    print("-" * 45)
    print(f"{'d_model':<15} {'128':<15} {'256':<15}")
    print(f"{'n_heads':<15} {'8':<15} {'16':<15}")
    print(f"{'n_layers':<15} {'3':<15} {'3':<15}")
    print(f"{'参数量估计':<15} {'~1.2M':<15} {'~4.8M':<15}")
    
    if base_metrics['epochs'] and exp_metrics['epochs']:
        print(f"\n📈 训练进度对比:")
        print(f"{'指标':<15} {'基础模型':<15} {'扩展模型':<15}")
        print("-" * 45)
        print(f"{'完成轮数':<15} {len(base_metrics['epochs']):<15} {len(exp_metrics['epochs']):<15}")
        
        # 最新性能对比
        if base_metrics['val_auc'] and exp_metrics['val_auc']:
            base_latest_auc = base_metrics['val_auc'][-1]
            exp_latest_auc = exp_metrics['val_auc'][-1]
            base_best_auc = max(base_metrics['val_auc'])
            exp_best_auc = max(exp_metrics['val_auc'])
            
            print(f"{'最新AUC':<15} {base_latest_auc:<15.4f} {exp_latest_auc:<15.4f}")
            print(f"{'最佳AUC':<15} {base_best_auc:<15.4f} {exp_best_auc:<15.4f}")
            print(f"{'AUC差异':<15} {'':<15} {exp_best_auc - base_best_auc:+.4f}")
            
            # 性能分析
            print(f"\n🎯 性能分析:")
            if exp_best_auc > base_best_auc:
                improvement = ((exp_best_auc - base_best_auc) / base_best_auc) * 100
                print(f"✅ 扩展模型性能更好，提升 {improvement:.2f}%")
            elif exp_best_auc < base_best_auc:
                decline = ((base_best_auc - exp_best_auc) / base_best_auc) * 100
                print(f"❌ 扩展模型性能更差，下降 {decline:.2f}%")
            else:
                print(f"⚖️  两个模型性能相当")
    
    # 训练速度对比
    if base_speed and exp_speed:
        print(f"\n⚡ 训练速度对比:")
        print(f"{'指标':<15} {'基础模型':<15} {'扩展模型':<15}")
        print("-" * 45)
        print(f"{'平均速度':<15} {base_speed['avg_speed']:<15.2f} {exp_speed['avg_speed']:<15.2f}")
        print(f"{'速度比':<15} {'1.00x':<15} {exp_speed['avg_speed']/base_speed['avg_speed']:<15.2f}")
        
        speed_ratio = base_speed['avg_speed'] / exp_speed['avg_speed']
        print(f"\n🚀 速度分析:")
        print(f"扩展模型训练速度是基础模型的 {1/speed_ratio:.2f}x")
        print(f"扩展模型每轮训练时间约为基础模型的 {speed_ratio:.2f}x")
    
    # 收敛性分析
    if base_metrics['val_auc'] and exp_metrics['val_auc']:
        print(f"\n📉 收敛性分析:")
        
        # 计算前10轮的AUC提升
        base_early_improvement = 0
        exp_early_improvement = 0
        
        if len(base_metrics['val_auc']) >= 10:
            base_early_improvement = base_metrics['val_auc'][9] - base_metrics['val_auc'][0]
        
        if len(exp_metrics['val_auc']) >= 10:
            exp_early_improvement = exp_metrics['val_auc'][9] - exp_metrics['val_auc'][0]
        
        print(f"前10轮AUC提升:")
        print(f"  基础模型: {base_early_improvement:.4f}")
        print(f"  扩展模型: {exp_early_improvement:.4f}")
        
        # 分析收敛稳定性
        if len(base_metrics['val_auc']) >= 20:
            base_late_std = np.std(base_metrics['val_auc'][-10:])
            print(f"后期稳定性 (最后10轮AUC标准差):")
            print(f"  基础模型: {base_late_std:.4f}")
        
        if len(exp_metrics['val_auc']) >= 20:
            exp_late_std = np.std(exp_metrics['val_auc'][-10:])
            print(f"  扩展模型: {exp_late_std:.4f}")
    
    # 问题诊断
    print(f"\n🔧 问题诊断:")
    
    if exp_metrics['val_auc'] and base_metrics['val_auc']:
        exp_best = max(exp_metrics['val_auc'])
        base_best = max(base_metrics['val_auc'])
        
        if exp_best < base_best:
            print("❌ 扩展模型性能不如基础模型，可能原因:")
            print("   1. 过拟合 - 模型容量过大，数据量不足")
            print("   2. 学习率不匹配 - 可能需要更小的学习率")
            print("   3. 训练不充分 - 需要更多训练轮数")
            print("   4. 正则化不足 - dropout可能需要增加")
            
            print(f"\n💡 建议的改进措施:")
            print("   1. 降低学习率至 0.0005 或 0.0001")
            print("   2. 增加dropout至 0.3-0.4")
            print("   3. 增加训练轮数至 150-200")
            print("   4. 考虑使用学习率调度器")
            print("   5. 增加权重衰减 (weight_decay)")
        
        elif exp_best > base_best:
            print("✅ 扩展模型性能更好，说明:")
            print("   1. 模型容量增加带来了性能提升")
            print("   2. 当前超参数设置合理")
            print("   3. 数据量足以支撑更大的模型")
        
        else:
            print("⚖️  两个模型性能相当，说明:")
            print("   1. 当前任务可能不需要更大的模型容量")
            print("   2. 数据质量比模型大小更重要")
            print("   3. 可以考虑其他改进方向")
    
    # 资源效率分析
    print(f"\n💰 资源效率分析:")
    if base_speed and exp_speed and base_metrics['val_auc'] and exp_metrics['val_auc']:
        base_best = max(base_metrics['val_auc'])
        exp_best = max(exp_metrics['val_auc'])
        
        # 计算性能/速度比
        base_efficiency = base_best * base_speed['avg_speed']
        exp_efficiency = exp_best * exp_speed['avg_speed']
        
        print(f"性能效率 (AUC × 训练速度):")
        print(f"  基础模型: {base_efficiency:.4f}")
        print(f"  扩展模型: {exp_efficiency:.4f}")
        
        if base_efficiency > exp_efficiency:
            print("📊 基础模型在性能效率上更优")
        else:
            print("📊 扩展模型在性能效率上更优")
    
    # 建议的下一步行动
    print(f"\n🎯 建议的下一步行动:")
    
    if not exp_metrics['epochs']:
        print("⏳ 等待扩展模型训练完成")
    elif len(exp_metrics['epochs']) < 30:
        print("⏳ 继续观察扩展模型训练进展")
    else:
        if exp_metrics['val_auc'] and base_metrics['val_auc']:
            exp_best = max(exp_metrics['val_auc'])
            base_best = max(base_metrics['val_auc'])
            
            if exp_best < base_best - 0.005:  # 显著更差
                print("🔧 建议调优扩展模型超参数")
                print("📊 开始基础配置的第二阶段训练")
            elif exp_best > base_best + 0.005:  # 显著更好
                print("✅ 继续扩展模型训练")
                print("📊 准备两个配置的第二阶段训练")
            else:  # 相当
                print("📊 开始两个配置的第二阶段训练")
                print("🔍 在第二阶段中进一步对比")


def main():
    """主函数"""
    print_comparison_analysis()


if __name__ == "__main__":
    main()
