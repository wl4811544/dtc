#!/usr/bin/env python3
"""
模型验证脚本 - 检查所有可用的模型
"""

import os
import torch
import glob
from datetime import datetime

def verify_all_models():
    """验证所有模型文件"""
    
    print("🔍 开始验证模型文件...")
    print("=" * 60)
    
    # 搜索所有可能的输出目录
    search_patterns = [
        "output/**/*.pt",
        "../output/**/*.pt",
        "../../output/**/*.pt"
    ]
    
    all_models = []
    
    for pattern in search_patterns:
        models = glob.glob(pattern, recursive=True)
        all_models.extend(models)
    
    # 去重
    all_models = list(set(all_models))
    
    if not all_models:
        print("❌ 没有找到任何模型文件")
        print("\n🔍 请检查以下位置:")
        print("  - output/")
        print("  - ../output/")
        print("  - ../../output/")
        return []
    
    print(f"✅ 找到 {len(all_models)} 个模型文件")
    print()
    
    valid_models = []
    
    for i, model_path in enumerate(sorted(all_models), 1):
        print(f"📁 模型 {i}: {os.path.basename(model_path)}")
        print(f"   路径: {model_path}")
        
        try:
            # 检查文件大小
            size_mb = os.path.getsize(model_path) / 1024 / 1024
            mtime = datetime.fromtimestamp(os.path.getmtime(model_path))
            
            print(f"   📊 大小: {size_mb:.1f} MB")
            print(f"   🕒 时间: {mtime}")
            
            # 尝试加载模型
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # 检查内容
            epoch = checkpoint.get('epoch', 'Unknown')
            print(f"   🔢 轮次: {epoch}")
            
            if 'metrics' in checkpoint:
                metrics = checkpoint['metrics']
                print(f"   📈 性能:")
                
                for key in ['recall', 'precision', 'f1_score', 'auc_roc']:
                    if key in metrics:
                        value = metrics[key]
                        if isinstance(value, (int, float)):
                            print(f"     {key}: {value:.4f}")
                
                valid_models.append({
                    'path': model_path,
                    'epoch': epoch,
                    'metrics': metrics,
                    'size_mb': size_mb,
                    'mtime': mtime
                })
            else:
                print("   ⚠️  没有保存的指标")
            
            print(f"   ✅ 模型有效")
            
        except Exception as e:
            print(f"   ❌ 加载失败: {e}")
        
        print()
    
    # 总结最佳模型
    if valid_models:
        print("=" * 60)
        print("🏆 最佳模型总结:")
        
        # 找到各项最佳
        best_recall = max(valid_models, key=lambda x: x['metrics'].get('recall', 0))
        best_auc = max(valid_models, key=lambda x: x['metrics'].get('auc_roc', 0))
        best_f1 = max(valid_models, key=lambda x: x['metrics'].get('f1_score', 0))
        
        print(f"\n📈 最佳召回率: {best_recall['metrics'].get('recall', 0):.4f}")
        print(f"   模型: {os.path.basename(best_recall['path'])}")
        print(f"   轮次: {best_recall['epoch']}")
        
        print(f"\n🎯 最佳AUC: {best_auc['metrics'].get('auc_roc', 0):.4f}")
        print(f"   模型: {os.path.basename(best_auc['path'])}")
        print(f"   轮次: {best_auc['epoch']}")
        
        print(f"\n⚖️  最佳F1: {best_f1['metrics'].get('f1_score', 0):.4f}")
        print(f"   模型: {os.path.basename(best_f1['path'])}")
        print(f"   轮次: {best_f1['epoch']}")
        
        # 推荐最佳模型
        print("\n" + "=" * 60)
        print("💡 推荐使用的模型:")
        
        # 综合评分
        scored_models = []
        for model in valid_models:
            metrics = model['metrics']
            recall = metrics.get('recall', 0)
            auc = metrics.get('auc_roc', 0)
            f1 = metrics.get('f1_score', 0)
            
            # 综合评分 (可以调整权重)
            score = recall * 0.4 + auc * 0.4 + f1 * 0.2
            scored_models.append((score, model))
        
        scored_models.sort(reverse=True)
        best_overall = scored_models[0][1]
        
        print(f"\n🏆 综合最佳模型:")
        print(f"   文件: {best_overall['path']}")
        print(f"   轮次: {best_overall['epoch']}")
        print(f"   召回率: {best_overall['metrics'].get('recall', 0):.4f}")
        print(f"   AUC: {best_overall['metrics'].get('auc_roc', 0):.4f}")
        print(f"   F1: {best_overall['metrics'].get('f1_score', 0):.4f}")
        
        print(f"\n🚀 继续训练命令:")
        print(f"python scripts/continue_training.py \\")
        print(f"    --checkpoint \"{best_overall['path']}\" \\")
        print(f"    --dataset assist17 \\")
        print(f"    --device cuda \\")
        print(f"    --with_pid --use_cl --proj \\")
        print(f"    --n_know 32 \\")
        print(f"    --batch_size 16 \\")
        print(f"    --test_batch_size 32 \\")
        print(f"    --epochs 50")
    
    return valid_models

if __name__ == "__main__":
    models = verify_all_models()
    
    if models:
        print(f"\n✅ 验证完成! 找到 {len(models)} 个有效模型")
    else:
        print("\n❌ 没有找到有效的模型文件")
        print("💡 可能需要重新训练")
