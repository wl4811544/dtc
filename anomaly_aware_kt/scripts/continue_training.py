#!/usr/bin/env python3
"""
继续训练脚本 - 从最后保存的模型继续训练
"""

import os
import sys
import torch
import argparse
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly_kt.unified_trainer import UnifiedTrainer
from anomaly_kt.training_strategies import EnhancedStrategy, AggressiveStrategy, BasicStrategy
from anomaly_kt.detector import CausalAnomalyDetector
from anomaly_kt.evaluator import AnomalyEvaluator
from data_loader import get_data_loader

def find_latest_checkpoint(base_dir="output/"):
    """找到最新的检查点"""
    
    all_checkpoints = []
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.pt'):
                full_path = os.path.join(root, file)
                try:
                    checkpoint = torch.load(full_path, map_location='cpu')
                    epoch = checkpoint.get('epoch', 0)
                    metrics = checkpoint.get('metrics', {})
                    
                    all_checkpoints.append({
                        'path': full_path,
                        'epoch': epoch,
                        'metrics': metrics,
                        'mtime': os.path.getmtime(full_path)
                    })
                except:
                    continue
    
    if not all_checkpoints:
        return None
    
    # 按轮次排序
    all_checkpoints.sort(key=lambda x: x['epoch'], reverse=True)
    return all_checkpoints[0]

def continue_training():
    """继续训练主函数"""
    
    parser = argparse.ArgumentParser(description='继续训练异常检测器')
    parser.add_argument('--checkpoint', type=str, help='指定检查点路径')
    parser.add_argument('--epochs', type=int, default=50, help='总训练轮数')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--dataset', type=str, default='assist17', help='数据集')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--test_batch_size', type=int, default=32, help='测试批次大小')
    parser.add_argument('--with_pid', action='store_true', help='使用问题ID')
    parser.add_argument('--use_cl', action='store_true', help='使用对比学习')
    parser.add_argument('--proj', action='store_true', help='使用投影')
    parser.add_argument('--n_know', type=int, default=32, help='知识点数量')
    
    args = parser.parse_args()
    
    # 1. 找到最新检查点
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        print("🔍 查找最新检查点...")
        latest = find_latest_checkpoint()
        if not latest:
            print("❌ 没有找到任何检查点文件")
            return
        checkpoint_path = latest['path']
    
    print(f"📁 使用检查点: {checkpoint_path}")
    
    # 2. 加载检查点
    print("📥 加载检查点...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    start_epoch = checkpoint.get('epoch', 0)
    saved_metrics = checkpoint.get('metrics', {})
    
    print(f"🔢 从第 {start_epoch} 轮继续")
    print(f"📊 上次性能:")
    for key, value in saved_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
    
    # 3. 准备数据
    print("📚 准备数据...")
    dataset_config = {
        'dataset_name': args.dataset,
        'batch_size': args.batch_size,
        'test_batch_size': args.test_batch_size,
        'with_pid': args.with_pid,
        'use_cl': args.use_cl,
        'proj': args.proj,
        'n_know': args.n_know
    }
    
    train_data, val_data, test_data = get_data_loader(dataset_config)
    
    # 4. 创建模型
    print("🤖 创建模型...")
    model = CausalAnomalyDetector(
        n_know=args.n_know,
        d_model=256,
        n_heads=8,
        n_layers=4,
        dropout=0.1,
        with_pid=args.with_pid,
        use_cl=args.use_cl,
        proj=args.proj
    )
    
    # 5. 加载模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    
    # 6. 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 7. 创建训练策略
    print("⚙️  创建训练策略...")
    
    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"output/continue_{args.dataset}_{timestamp}/detector"
    os.makedirs(save_dir, exist_ok=True)
    
    # 使用Enhanced策略（根据之前的成功经验）
    strategy = EnhancedStrategy(model, args.device, save_dir, patience=15)
    
    # 8. 创建训练器
    evaluator = AnomalyEvaluator()
    trainer = UnifiedTrainer(strategy, evaluator, save_dir)
    
    # 9. 设置训练配置
    train_config = {
        'epochs': args.epochs,
        'learning_rate': 0.001,
        'optimize_for': 'recall',  # 继续优化召回率
        'anomaly_ratio': 0.3,
        'patience': 15,
        'start_epoch': start_epoch  # 从指定轮次开始
    }
    
    print(f"🚀 继续训练...")
    print(f"📊 从第 {start_epoch} 轮开始，目标 {args.epochs} 轮")
    print(f"🎯 优化目标: {train_config['optimize_for']}")
    
    # 10. 开始训练
    try:
        final_metrics = trainer.train(train_data, val_data, train_config)
        
        print("\n🎉 训练完成!")
        print("📊 最终性能:")
        for key, value in final_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
                
    except KeyboardInterrupt:
        print("\n⏹️  训练被用户中断")
        print("💾 所有模型已保存，可以随时继续")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    continue_training()
