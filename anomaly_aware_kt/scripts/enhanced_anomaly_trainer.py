#!/usr/bin/env python
"""
优化的异常检测器训练脚本
针对Recall和检测性能进行专门优化
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import tomlkit
import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DTransformer.data import KTData
from anomaly_kt.generator import AnomalyGenerator
from anomaly_kt.detector import CausalAnomalyDetector
from anomaly_kt.trainer import BaseTrainer
from anomaly_kt.evaluator import AnomalyEvaluator


class EnhancedAnomalyTrainer(BaseTrainer):
    """增强的异常检测训练器 - 专门优化检测性能"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu', 
                 save_dir: str = 'output/enhanced_detector', patience: int = 15):
        super().__init__(model, device, save_dir, patience)
        self.generator = AnomalyGenerator()
        self.evaluator = AnomalyEvaluator()
        
        # 增强配置
        self.class_weights = None
        self.best_threshold = 0.5
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
        
    def train(self, 
              train_loader, 
              val_loader,
              epochs: int = 50,
              learning_rate: float = 5e-4,
              anomaly_ratio: float = 0.3,
              optimize_for: str = 'recall',
              use_focal_loss: bool = True,
              use_class_weights: bool = True,
              use_progressive_training: bool = True,
              gradient_accumulation_steps: int = 2,
              warmup_epochs: int = 5) -> Dict:
        """
        增强的训练方法
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            learning_rate: 学习率
            anomaly_ratio: 异常生成比例
            optimize_for: 优化目标 ('recall', 'f1', 'precision', 'auc')
            use_focal_loss: 使用Focal Loss
            use_class_weights: 使用类别权重
            use_progressive_training: 使用渐进式训练
            gradient_accumulation_steps: 梯度累积步数
            warmup_epochs: 预热轮数
        """
        
        print(f"🚀 开始增强异常检测器训练")
        print(f"  📊 优化目标: {optimize_for}")
        print(f"  🎯 异常比例: {anomaly_ratio}")
        print(f"  🔥 Focal Loss: {use_focal_loss}")
        print(f"  ⚖️  类别权重: {use_class_weights}")
        print(f"  📈 渐进训练: {use_progressive_training}")
        
        # 1. 设置优化器和调度器
        optimizer = self._setup_optimizer(learning_rate, warmup_epochs, epochs)
        scheduler = self._setup_scheduler(optimizer, epochs)
        
        # 2. 保存配置
        self._save_config({
            'epochs': epochs,
            'learning_rate': learning_rate,
            'anomaly_ratio': anomaly_ratio,
            'optimize_for': optimize_for,
            'use_focal_loss': use_focal_loss,
            'use_class_weights': use_class_weights,
            'use_progressive_training': use_progressive_training
        })
        
        # 3. 训练循环
        best_score = 0
        no_improve = 0
        
        for epoch in range(epochs):
            print(f"\n{'='*20} Epoch {epoch+1}/{epochs} {'='*20}")
            
            # 动态调整训练参数
            current_anomaly_ratio = self._get_dynamic_anomaly_ratio(
                epoch, epochs, anomaly_ratio, use_progressive_training
            )
            
            # 训练一个epoch
            train_metrics = self._train_epoch_enhanced(
                train_loader, optimizer, current_anomaly_ratio,
                use_focal_loss, use_class_weights, gradient_accumulation_steps
            )
            
            # 验证
            val_metrics = self._validate_epoch_enhanced(
                val_loader, current_anomaly_ratio, optimize_for
            )
            
            # 更新学习率
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics[optimize_for])
            else:
                scheduler.step()
            
            # 记录指标
            self._record_metrics(epoch, train_metrics, val_metrics, optimizer)
            
            # 打印结果
            self._print_epoch_results(epoch, train_metrics, val_metrics, current_anomaly_ratio)
            
            # 检查最佳模型
            current_score = val_metrics[optimize_for]
            if current_score > best_score:
                best_score = current_score
                self.best_metrics = val_metrics.copy()
                self.best_epoch = epoch + 1
                self.best_threshold = val_metrics.get('best_threshold', 0.5)
                self.save_checkpoint(epoch, optimizer, val_metrics, is_best=True)
                no_improve = 0
                print(f"  ✅ 新的最佳{optimize_for}: {current_score:.4f}")
            else:
                no_improve += 1
                print(f"  📊 当前{optimize_for}: {current_score:.4f} (最佳: {best_score:.4f})")
            
            # 早停检查
            if no_improve >= self.patience:
                print(f"\n⏹️  早停触发 - {self.patience}轮无改善")
                break
        
        # 训练完成
        print(f"\n🎉 训练完成!")
        print(f"  🏆 最佳轮次: {self.best_epoch}")
        print(f"  📈 最佳{optimize_for}: {best_score:.4f}")
        print(f"  🎯 最佳阈值: {self.best_threshold:.3f}")
        
        # 绘制训练曲线
        self._plot_training_curves()
        
        return self.best_metrics
    
    def _setup_optimizer(self, learning_rate: float, warmup_epochs: int, total_epochs: int):
        """设置优化器"""
        # 使用AdamW优化器，更好的权重衰减
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        return optimizer
    
    def _setup_scheduler(self, optimizer, total_epochs: int):
        """设置学习率调度器"""
        # 使用余弦退火调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=total_epochs, 
            eta_min=1e-6
        )
        return scheduler
    
    def _get_dynamic_anomaly_ratio(self, epoch: int, total_epochs: int, 
                                  base_ratio: float, use_progressive: bool) -> float:
        """动态调整异常生成比例"""
        if not use_progressive:
            return base_ratio
        
        # 前期生成更多异常，后期逐渐减少到正常比例
        progress = epoch / total_epochs
        if progress < 0.3:  # 前30%轮次
            return min(base_ratio * 2.0, 0.5)  # 最多50%
        elif progress < 0.7:  # 中间40%轮次
            return base_ratio * 1.5
        else:  # 后30%轮次
            return base_ratio
    
    def _train_epoch_enhanced(self, train_loader, optimizer, anomaly_ratio: float,
                             use_focal_loss: bool, use_class_weights: bool,
                             gradient_accumulation_steps: int) -> Dict:
        """增强的训练epoch"""
        self.model.train()
        
        total_loss = 0
        total_samples = 0
        step_count = 0
        
        # 统计信息
        pos_samples = 0
        neg_samples = 0
        
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc="训练中", leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            q, s, pid = self._get_batch_data(batch)
            
            # 生成平衡的异常数据
            s_anomaly, labels = self._generate_balanced_anomalies(q, s, anomaly_ratio)
            labels = labels.to(self.device)
            
            # 统计样本
            pos_samples += (labels == 1).sum().item()
            neg_samples += (labels == 0).sum().item()
            
            # 前向传播
            if use_focal_loss:
                loss = self._compute_focal_loss(q, s_anomaly, labels, pid)
            else:
                loss = self._compute_weighted_loss(q, s_anomaly, labels, pid, use_class_weights)
            
            # 梯度累积
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 优化器步骤
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                step_count += 1
            
            total_loss += loss.item() * gradient_accumulation_steps
            total_samples += 1
            
            # 更新进度条
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Pos%': f'{pos_samples/(pos_samples+neg_samples)*100:.1f}%'
                })
        
        # 处理剩余的梯度
        if total_samples % gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = total_loss / total_samples
        pos_ratio = pos_samples / (pos_samples + neg_samples) if (pos_samples + neg_samples) > 0 else 0
        
        return {
            'loss': avg_loss,
            'positive_ratio': pos_ratio,
            'total_samples': pos_samples + neg_samples
        }
    
    def _validate_epoch_enhanced(self, val_loader, anomaly_ratio: float, 
                                optimize_for: str) -> Dict:
        """增强的验证epoch"""
        self.model.eval()
        self.evaluator.reset()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="验证中", leave=False):
                q, s, pid = self._get_batch_data(batch)
                
                # 生成异常
                s_anomaly, labels = self._generate_balanced_anomalies(q, s, anomaly_ratio)
                
                # 预测
                predictions = self.model(q, s_anomaly, pid)
                
                # 收集预测结果
                mask = (s >= 0)
                all_predictions.extend(predictions[mask].cpu().numpy())
                all_labels.extend(labels[mask].cpu().numpy())
                
                # 更新评估器
                self.evaluator.update(predictions, labels, q, s)
        
        # 计算指标
        metrics = self.evaluator.compute_metrics()
        
        # 寻找最佳阈值
        best_threshold = self._find_optimal_threshold(
            np.array(all_predictions), np.array(all_labels), optimize_for
        )
        metrics['best_threshold'] = best_threshold
        
        return metrics
    
    def _generate_balanced_anomalies(self, q: torch.Tensor, s: torch.Tensor, 
                                   anomaly_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成更平衡的异常数据"""
        # 使用改进的策略权重
        strategy_weights = {
            'consecutive': 0.25,    # 连续异常
            'pattern': 0.25,        # 模式异常  
            'random_burst': 0.25,   # 随机突发
            'difficulty_based': 0.25 # 基于难度
        }
        
        return self.generator.generate_anomalies(
            q, s, anomaly_ratio=anomaly_ratio, strategy_weights=strategy_weights
        )
    
    def _compute_focal_loss(self, q: torch.Tensor, s: torch.Tensor, 
                           labels: torch.Tensor, pid: Optional[torch.Tensor]) -> torch.Tensor:
        """计算Focal Loss"""
        # 获取原始logits
        predictions = self.model(q, s, pid)
        
        # 创建有效样本掩码
        mask = (s >= 0)
        valid_preds = predictions[mask]
        valid_labels = labels[mask].float()
        
        if len(valid_preds) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 计算BCE损失（使用logits版本以提高数值稳定性）
        # 先转换sigmoid输出回logits
        epsilon = 1e-7
        valid_preds_clipped = torch.clamp(valid_preds, epsilon, 1 - epsilon)
        logits = torch.log(valid_preds_clipped / (1 - valid_preds_clipped))
        
        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, valid_labels, reduction='none'
        )
        
        # 计算pt
        pt = torch.exp(-bce_loss)
        
        # Alpha权重
        alpha_weight = torch.where(valid_labels == 1, self.focal_alpha, 1 - self.focal_alpha)
        
        # Focal权重
        focal_weight = alpha_weight * (1 - pt) ** self.focal_gamma
        
        # 最终损失
        focal_loss = focal_weight * bce_loss
        
        return focal_loss.mean()
    
    def _compute_weighted_loss(self, q: torch.Tensor, s: torch.Tensor,
                              labels: torch.Tensor, pid: Optional[torch.Tensor],
                              use_class_weights: bool) -> torch.Tensor:
        """计算加权损失"""
        predictions = self.model(q, s, pid)
        
        mask = (s >= 0)
        valid_preds = predictions[mask]
        valid_labels = labels[mask].float()
        
        if len(valid_preds) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 计算类别权重
        if use_class_weights:
            pos_count = valid_labels.sum()
            neg_count = len(valid_labels) - pos_count
            
            if pos_count > 0:
                pos_weight = neg_count / pos_count
                pos_weight = torch.clamp(pos_weight, min=1.0, max=10.0)
            else:
                pos_weight = torch.tensor(5.0, device=self.device)
        else:
            pos_weight = torch.tensor(1.0, device=self.device)
        
        # 使用BCEWithLogitsLoss
        epsilon = 1e-7
        valid_preds_clipped = torch.clamp(valid_preds, epsilon, 1 - epsilon)
        logits = torch.log(valid_preds_clipped / (1 - valid_preds_clipped))
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = criterion(logits, valid_labels)
        
        return loss
    
    def _find_optimal_threshold(self, predictions: np.ndarray, labels: np.ndarray, 
                               metric: str = 'f1') -> float:
        """寻找最优阈值"""
        best_score = 0
        best_threshold = 0.5
        
        # 尝试更多阈值
        thresholds = np.linspace(0.1, 0.9, 33)  # 33个阈值点
        
        for threshold in thresholds:
            pred_binary = (predictions >= threshold).astype(int)
            
            # 避免除零错误
            tp = ((pred_binary == 1) & (labels == 1)).sum()
            fp = ((pred_binary == 1) & (labels == 0)).sum()
            fn = ((pred_binary == 0) & (labels == 1)).sum()
            
            if metric == 'recall':
                score = tp / (tp + fn) if (tp + fn) > 0 else 0
            elif metric == 'precision':
                score = tp / (tp + fp) if (tp + fp) > 0 else 0
            elif metric == 'f1':
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            else:
                score = 0
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold
    
    def _record_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict, optimizer):
        """记录训练指标"""
        # 记录训练指标
        for key, value in train_metrics.items():
            self.history[f'train_{key}'].append(value)
        
        # 记录验证指标  
        for key, value in val_metrics.items():
            self.history[f'val_{key}'].append(value)
        
        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        self.history['learning_rate'].append(current_lr)
    
    def _print_epoch_results(self, epoch: int, train_metrics: Dict, val_metrics: Dict, anomaly_ratio: float):
        """打印epoch结果"""
        print(f"  📊 训练 - Loss: {train_metrics['loss']:.4f}, "
              f"异常比例: {train_metrics['positive_ratio']:.1%}")
        print(f"  📈 验证 - Recall: {val_metrics['recall']:.3f}, "
              f"Precision: {val_metrics['precision']:.3f}, "
              f"F1: {val_metrics['f1_score']:.3f}, "
              f"AUC: {val_metrics['auc_roc']:.3f}")
        print(f"  🎯 最佳阈值: {val_metrics['best_threshold']:.3f}, "
              f"当前异常比例: {anomaly_ratio:.1%}")
    
    def _plot_training_curves(self):
        """绘制训练曲线"""
        if not self.history:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss曲线
        if 'train_loss' in self.history:
            axes[0, 0].plot(self.history['train_loss'], label='Train Loss', alpha=0.8)
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 学习率曲线
        if 'learning_rate' in self.history:
            axes[0, 1].plot(self.history['learning_rate'], color='orange', alpha=0.8)
            axes[0, 1].set_title('Learning Rate')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 验证指标
        if 'val_recall' in self.history:
            axes[1, 0].plot(self.history['val_recall'], label='Recall', marker='o', alpha=0.8)
            axes[1, 0].plot(self.history['val_precision'], label='Precision', marker='s', alpha=0.8)
            axes[1, 0].plot(self.history['val_f1_score'], label='F1 Score', marker='^', alpha=0.8)
            axes[1, 0].set_title('Validation Metrics')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # AUC曲线
        if 'val_auc_roc' in self.history:
            axes[1, 1].plot(self.history['val_auc_roc'], color='green', marker='o', alpha=0.8)
            axes[1, 1].set_title('AUC-ROC Score')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('AUC')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'enhanced_training_curves.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  📊 训练曲线已保存到: {self.save_dir}/enhanced_training_curves.png")
    
    def _get_batch_data(self, batch):
        """获取批次数据"""
        if len(batch.data) == 2:
            q, s = batch.get("q", "s")
            pid = None
        else:
            q, s, pid = batch.get("q", "s", "pid")
        
        q = q.to(self.device)
        s = s.to(self.device)
        if pid is not None:
            pid = pid.to(self.device)
        
        return q, s, pid


def train_enhanced_detector(args):
    """训练增强的异常检测器"""
    print("🚀 启动增强异常检测器训练")
    
    # 加载数据配置
    datasets = tomlkit.load(open(os.path.join(args.data_dir, 'datasets.toml')))
    dataset_config = datasets[args.dataset]
    
    print(f"📊 数据集: {args.dataset}")
    print(f"  问题数: {dataset_config['n_questions']}")
    print(f"  知识点数: {dataset_config['n_pid']}")
    
    # 准备数据
    train_data = KTData(
        os.path.join(args.data_dir, dataset_config['train']),
        dataset_config['inputs'],
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_data = KTData(
        os.path.join(args.data_dir, dataset_config.get('valid', dataset_config['test'])),
        dataset_config['inputs'],
        batch_size=args.batch_size
    )
    
    print(f"📁 训练批次: {len(train_data)}, 验证批次: {len(val_data)}")
    
    # 创建增强的检测器
    detector = CausalAnomalyDetector(
        n_questions=dataset_config['n_questions'],
        n_pid=dataset_config['n_pid'] if args.with_pid else 0,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        window_size=args.window_size
    )
    
    print(f"🧠 模型参数: {sum(p.numel() for p in detector.parameters()):,}")
    
    # 使用增强的训练器
    trainer = EnhancedAnomalyTrainer(
        model=detector,
        device=args.device,
        save_dir=args.save_dir,
        patience=args.patience
    )
    
    # 训练
    best_metrics = trainer.train(
        train_loader=train_data,
        val_loader=val_data,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        anomaly_ratio=args.anomaly_ratio,
        optimize_for=args.optimize_for,
        use_focal_loss=args.use_focal_loss,
        use_class_weights=args.use_class_weights,
        use_progressive_training=args.use_progressive_training,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_epochs=args.warmup_epochs
    )
    
    print("\n🎉 训练完成!")
    print("📊 最终结果:")
    for metric, value in best_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return best_metrics


def main():
    parser = argparse.ArgumentParser(description='Enhanced Anomaly Detector Training')
    
    # 数据参数
    parser.add_argument('-d', '--dataset', required=True, 
                        choices=['assist09', 'assist17', 'algebra05', 'statics'])
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('-p', '--with_pid', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--window_size', type=int, default=15)
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--anomaly_ratio', type=float, default=0.3)
    parser.add_argument('--optimize_for', default='recall', 
                        choices=['recall', 'precision', 'f1', 'auc_roc'])
    parser.add_argument('--patience', type=int, default=15)
    
    # 增强策略
    parser.add_argument('--use_focal_loss', action='store_true', 
                        help='使用Focal Loss处理类别不平衡')
    parser.add_argument('--use_class_weights', action='store_true',
                        help='使用类别权重')
    parser.add_argument('--use_progressive_training', action='store_true',
                        help='使用渐进式训练策略')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help='梯度累积步数')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='学习率预热轮数')
    
    # 其他参数
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', default='output/enhanced_detector')
    
    args = parser.parse_args()
    
    print("⚙️  配置参数:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    # 创建输出目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 训练
    try:
        metrics = train_enhanced_detector(args)
        print(f"\n✅ 训练成功完成!")
        
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())