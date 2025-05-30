#!/usr/bin/env python
"""
完整的异常感知知识追踪训练流程 - 增强版

包括：
1. 训练基线DTransformer模型（可选）
2. 使用增强方法训练异常检测器
3. 训练异常感知的知识追踪模型
4. 评估性能提升
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import tomlkit
import yaml
import json
import numpy as np
import random
from datetime import datetime
from tqdm import tqdm
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DTransformer.data import KTData
from DTransformer.model import DTransformer

from anomaly_kt.generator import AnomalyGenerator
from anomaly_kt.detector import CausalAnomalyDetector
from anomaly_kt.model import AnomalyAwareDTransformer
from anomaly_kt.trainer import AnomalyDetectorTrainer, KnowledgeTracingTrainer
from anomaly_kt.evaluator import ComparisonEvaluator, plot_training_curves

# 导入增强训练器
from anomaly_kt.trainer import AnomalyDetectorTrainer as ImprovedAnomalyTrainer


# 激进的异常训练器类
class AggressiveAnomalyTrainer(ImprovedAnomalyTrainer):
    """激进的异常检测训练器 - 处理严重不平衡"""

    def train_aggressive(self, train_loader, val_loader, epochs=30,
                        learning_rate=1e-3, anomaly_ratio=0.3,
                        min_anomaly_ratio=0.2, max_anomaly_ratio=0.5,
                        force_balance=True, extreme_weights=True):
        """激进的训练策略"""

        print("\n🔥 激进训练模式启动!")
        print(f"  - 动态异常比例: {max_anomaly_ratio:.0%} → {min_anomaly_ratio:.0%}")
        print(f"  - 强制批次平衡: {force_balance}")
        print(f"  - 极端类别权重: {extreme_weights}")

        # 使用Adam优化器（对学习率更敏感）
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-5
        )

        # 激进的学习率调度
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2,
            threshold=0.01
        )

        best_recall = 0
        no_improve = 0

        for epoch in range(epochs):
            print(f"\n{'='*20} Epoch {epoch+1}/{epochs} {'='*20}")

            # 激进的动态异常比例
            if epoch < 5:
                current_ratio = max_anomaly_ratio  # 50%
            elif epoch < 10:
                current_ratio = 0.4
            elif epoch < 15:
                current_ratio = 0.3
            else:
                progress = (epoch - 15) / (epochs - 15)
                current_ratio = max(min_anomaly_ratio,
                                  0.3 * (1 - progress) + min_anomaly_ratio * progress)

            # 训练
            train_metrics = self._train_epoch_aggressive(
                train_loader, optimizer, current_ratio,
                force_balance, extreme_weights
            )

            # 验证
            val_metrics = self._validate_epoch_improved(
                val_loader, current_ratio, 'recall', use_ema=False
            )

            # 打印结果
            print(f"  📊 异常比例: {current_ratio:.1%}")
            print(f"  📈 训练 - Loss: {train_metrics['loss']:.4f}, "
                  f"正样本比例: {train_metrics['positive_ratio']:.1%}")
            print(f"  📊 验证 - Recall: {val_metrics['recall']:.3f}, "
                  f"Precision: {val_metrics['precision']:.3f}, "
                  f"F1: {val_metrics['f1_score']:.3f}, "
                  f"AUC: {val_metrics['auc_roc']:.3f}")

            # 学习率调整
            scheduler.step(val_metrics['recall'])
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  📉 学习率: {current_lr:.2e}")

            # 检查是否需要采取紧急措施
            if epoch > 5 and val_metrics['recall'] < 0.3:
                print("\n⚠️  Recall过低，采取紧急措施!")
                self._emergency_measures(optimizer, learning_rate)

            # 保存最佳模型
            if val_metrics['recall'] > best_recall:
                best_recall = val_metrics['recall']
                self.best_metrics = val_metrics
                self.best_epoch = epoch + 1
                self.save_checkpoint(epoch, optimizer, val_metrics, is_best=True)
                no_improve = 0
                print(f"  ✅ 新的最佳Recall: {best_recall:.4f}")
            else:
                no_improve += 1

            # 早停
            if no_improve >= 8:
                print("\n⏹️  早停触发")
                break

        return self.best_metrics

    def _train_epoch_aggressive(self, train_loader, optimizer, anomaly_ratio,
                               force_balance, extreme_weights):
        """激进的训练epoch"""
        self.model.train()
        total_loss = 0
        n_batches = 0

        # 统计
        total_pos = 0
        total_neg = 0

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(train_loader, desc="激进训练")):
            q, s, pid = self._get_batch_data(batch)

            # 生成异常
            if force_balance:
                s_anomaly, labels = self._force_balanced_generation(q, s, anomaly_ratio)
            else:
                s_anomaly, labels = self._generate_smart_anomalies(q, s, anomaly_ratio, 0)

            labels = labels.to(self.device)

            # 统计
            mask = (s >= 0)
            pos_count = (labels[mask] == 1).sum().item()
            neg_count = (labels[mask] == 0).sum().item()
            total_pos += pos_count
            total_neg += neg_count

            # 前向传播
            predictions = self.model(q, s_anomaly, pid)

            # 计算损失
            if extreme_weights:
                loss = self._compute_extreme_weighted_loss(predictions, labels, s)
            else:
                loss = self._compute_focal_loss(predictions, labels, s, gamma=3.0, alpha=0.3)

            # 反向传播
            loss.backward()

            # 每4步更新一次（梯度累积）
            if (batch_idx + 1) % 4 == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            n_batches += 1

            # 定期打印预测分布
            if batch_idx % 100 == 0:
                with torch.no_grad():
                    pred_mean = predictions[mask].mean().item()
                    pred_std = predictions[mask].std().item()
                    tqdm.write(f"Batch {batch_idx} - Pred mean: {pred_mean:.3f}, "
                             f"std: {pred_std:.3f}, pos/neg: {pos_count}/{neg_count}")

        # 最后的梯度更新
        if n_batches % 4 != 0:
            optimizer.step()
            optimizer.zero_grad()

        print(f"  📊 批次统计 - 正样本: {total_pos}, 负样本: {total_neg}, "
              f"正样本比例: {total_pos/(total_pos+total_neg)*100:.1f}%")

        return {
            'loss': total_loss / n_batches,
            'positive_ratio': total_pos / (total_pos + total_neg)
        }

    def _force_balanced_generation(self, q, s, target_ratio):
        """强制生成平衡的批次"""
        batch_size, seq_len = s.shape
        s_anomaly = s.clone()
        labels = torch.zeros_like(s)

        # 确保每个序列都有异常
        for i in range(batch_size):
            valid_mask = (s[i] >= 0)
            valid_indices = torch.where(valid_mask)[0]

            if len(valid_indices) < 5:
                continue

            # 每个序列至少20%的位置是异常
            n_anomalies = max(1, int(len(valid_indices) * max(0.2, target_ratio)))
            anomaly_positions = torch.randperm(len(valid_indices))[:n_anomalies]

            for pos in anomaly_positions:
                idx = valid_indices[pos]
                # 随机选择异常类型
                if np.random.random() < 0.5:
                    # 翻转答案
                    s_anomaly[i, idx] = 1 - s_anomaly[i, idx]
                else:
                    # 随机答案
                    s_anomaly[i, idx] = np.random.randint(0, 2)
                labels[i, idx] = 1

        return s_anomaly, labels

    def _compute_extreme_weighted_loss(self, predictions, labels, s):
        """极端加权的损失函数"""
        mask = (s >= 0)
        valid_preds = predictions[mask]
        valid_labels = labels[mask].float()

        if len(valid_preds) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # 计算极端的类别权重
        pos_count = valid_labels.sum()
        neg_count = len(valid_labels) - pos_count

        if pos_count > 0:
            # 使用更极端的权重
            pos_weight = (neg_count / pos_count) * 2
            pos_weight = torch.clamp(pos_weight, min=5.0, max=50.0)
        else:
            pos_weight = torch.tensor(20.0, device=self.device)

        # 使用加权BCE
        loss = F.binary_cross_entropy(valid_preds, valid_labels, reduction='none')
        weights = torch.where(valid_labels == 1, pos_weight, 1.0)

        # 额外惩罚假阴性（漏检）
        fn_penalty = torch.where(
            (valid_labels == 1) & (valid_preds < 0.5),
            torch.tensor(2.0, device=self.device),
            torch.tensor(1.0, device=self.device)
        )

        weighted_loss = (loss * weights * fn_penalty).mean()

        return weighted_loss

    def _emergency_measures(self, optimizer, base_lr):
        """紧急措施：当性能太差时"""
        print("  🚨 执行紧急措施:")

        # 1. 重新初始化分类头
        if hasattr(self.model, 'classifier'):
            for layer in self.model.classifier:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.1)  # 轻微正偏置
            print("    - 重新初始化分类头")

        # 2. 增加学习率
        new_lr = base_lr * 3
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        print(f"    - 学习率提升到: {new_lr:.2e}")

        # 3. 减少正则化
        for param_group in optimizer.param_groups:
            param_group['weight_decay'] = 1e-6
        print("    - 减少权重衰减")

    def _validate_epoch_improved(self, val_loader, anomaly_ratio, optimize_for='recall', use_ema=False):
        """改进的验证epoch"""
        self.model.eval()
        self.evaluator.reset()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="验证中", leave=False):
                q, s, pid = self._get_batch_data(batch)

                # 生成异常
                s_anomaly, labels = self.generator.generate_anomalies(
                    q, s, anomaly_ratio=anomaly_ratio
                )

                # 预测
                predictions = self.model(q, s_anomaly, pid)

                # 更新评估器
                self.evaluator.update(predictions, labels, q, s)

        return self.evaluator.compute_metrics()

    def _generate_smart_anomalies(self, q, s, anomaly_ratio, strategy_id):
        """生成智能异常"""
        return self.generator.generate_anomalies(q, s, anomaly_ratio=anomaly_ratio)

    def _compute_focal_loss(self, predictions, labels, s, gamma=2.0, alpha=0.25):
        """计算Focal Loss"""
        mask = (s >= 0)
        valid_preds = predictions[mask]
        valid_labels = labels[mask].float()

        if len(valid_preds) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # 计算BCE损失
        epsilon = 1e-7
        valid_preds_clipped = torch.clamp(valid_preds, epsilon, 1 - epsilon)
        bce_loss = -(valid_labels * torch.log(valid_preds_clipped) +
                    (1 - valid_labels) * torch.log(1 - valid_preds_clipped))

        # 计算pt
        pt = torch.where(valid_labels == 1, valid_preds_clipped, 1 - valid_preds_clipped)

        # Alpha权重
        alpha_weight = torch.where(valid_labels == 1, alpha, 1 - alpha)

        # Focal权重
        focal_weight = alpha_weight * (1 - pt) ** gamma

        # 最终损失
        focal_loss = focal_weight * bce_loss

        return focal_loss.mean()


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        else:
            raise ValueError("Only YAML config files are supported")
    return config


def prepare_data(dataset_name: str, data_dir: str, batch_size: int, test_batch_size: int):
    """准备数据集"""
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


def train_baseline_model(args, dataset_config, train_data, val_data):
    """训练基线DTransformer模型"""
    print("\n" + "="*60)
    print("PHASE 1: Training Baseline DTransformer")
    print("="*60)

    # 创建模型
    model = DTransformer(
        n_questions=dataset_config['n_questions'],
        n_pid=dataset_config['n_pid'] if args.with_pid else 0,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_know=args.n_know,
        n_layers=args.n_layers,
        dropout=args.dropout,
        lambda_cl=args.lambda_cl,
        proj=args.proj,
        hard_neg=args.hard_neg,
        window=args.window
    )

    # 训练器
    trainer = KnowledgeTracingTrainer(
        model=model,
        device=args.device,
        save_dir=os.path.join(args.output_dir, 'baseline'),
        patience=args.patience
    )

    # 训练
    baseline_metrics = trainer.train(
        train_loader=train_data,
        val_loader=val_data,
        epochs=args.kt_epochs,
        learning_rate=args.learning_rate,
        use_cl=args.use_cl
    )

    print(f"\nBaseline training completed!")
    print(f"Best AUC: {baseline_metrics['auc']:.4f}")

    return os.path.join(args.output_dir, 'baseline', 'best_model.pt')


def train_anomaly_detector_enhanced(args, dataset_config, train_data, val_data):
    """使用增强方法训练异常检测器"""
    print("\n" + "="*60)
    print("PHASE 2: Training Anomaly Detector (Enhanced Version)")
    print("="*60)

    # 创建检测器
    detector = CausalAnomalyDetector(
        n_questions=dataset_config['n_questions'],
        n_pid=dataset_config['n_pid'] if args.with_pid else 0,
        d_model=args.detector_d_model,
        n_heads=args.detector_n_heads,
        n_layers=args.detector_n_layers,
        dropout=args.detector_dropout,
        window_size=args.window_size
    )

    print(f"🧠 模型参数: {sum(p.numel() for p in detector.parameters()):,}")

    # 检查是否使用激进策略
    if args.use_aggressive_strategy:
        print("\n⚠️  使用激进训练策略！")
        print("  - 动态异常比例: 50% → 20%")
        print("  - 强制批次平衡")
        print("  - 极端类别权重")

        trainer = AggressiveAnomalyTrainer(
            model=detector,
            device=args.device,
            save_dir=os.path.join(args.output_dir, 'detector'),
            patience=args.detector_patience
        )

        # 使用激进的训练参数
        detector_metrics = trainer.train_aggressive(
            train_loader=train_data,
            val_loader=val_data,
            epochs=args.detector_epochs,
            learning_rate=args.detector_lr * 2,  # 更高的学习率
            anomaly_ratio=args.anomaly_ratio,
            min_anomaly_ratio=0.2,  # 最小20%异常
            max_anomaly_ratio=0.5,  # 最大50%异常
            force_balance=True,
            extreme_weights=True
        )
    else:
        # 使用标准增强训练器
        trainer = ImprovedAnomalyTrainer(
            model=detector,
            device=args.device,
            save_dir=os.path.join(args.output_dir, 'detector'),
            patience=args.detector_patience
        )

        print("\n📋 训练配置:")
        print(f"  - 优化目标: {args.optimize_for}")
        print(f"  - 异常比例: {args.anomaly_ratio}")
        print(f"  - Focal Loss: {args.use_focal_loss}")
        print(f"  - Mixup: {args.use_mixup}")
        print(f"  - Label Smoothing: {args.use_label_smoothing}")
        print(f"  - 梯度累积: {args.gradient_accumulation_steps}步")

        # 训练
        detector_metrics = trainer.train(
            train_loader=train_data,
            val_loader=val_data,
            epochs=args.detector_epochs,
            learning_rate=args.detector_lr,
            anomaly_ratio=args.anomaly_ratio,
            optimize_for=args.optimize_for
        )

    print(f"\n✅ Detector training completed!")
    print(f"Best {args.optimize_for}: {detector_metrics[args.optimize_for]:.4f}")
    print(f"F1 Score: {detector_metrics['f1_score']:.4f}")
    print(f"AUC-ROC: {detector_metrics['auc_roc']:.4f}")
    print(f"Best Threshold: {detector_metrics.get('best_threshold', 0.5):.3f}")

    # 保存训练结果摘要
    summary = {
        'metrics': detector_metrics,
        'config': {
            'anomaly_ratio': args.anomaly_ratio,
            'optimize_for': args.optimize_for,
            'use_focal_loss': args.use_focal_loss,
            'use_mixup': args.use_mixup,
            'use_label_smoothing': args.use_label_smoothing
        }
    }

    with open(os.path.join(args.output_dir, 'detector', 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # 使用EMA模型作为最终模型（如果存在）
    ema_model_path = os.path.join(args.output_dir, 'detector', f'ema_model_epoch_{trainer.best_epoch}.pt')
    if os.path.exists(ema_model_path):
        print(f"📌 使用EMA模型作为最终检测器")
        return ema_model_path
    else:
        return os.path.join(args.output_dir, 'detector', 'best_model.pt')


def train_anomaly_aware_model(args, dataset_config, train_data, val_data, detector_path):
    """训练异常感知的知识追踪模型"""
    print("\n" + "="*60)
    print("PHASE 3: Training Anomaly-Aware DTransformer")
    print("="*60)

    # 加载异常检测器
    detector = CausalAnomalyDetector(
        n_questions=dataset_config['n_questions'],
        n_pid=dataset_config['n_pid'] if args.with_pid else 0,
        d_model=args.detector_d_model,
        n_heads=args.detector_n_heads,
        n_layers=args.detector_n_layers,
        dropout=args.detector_dropout,
        window_size=args.window_size
    )

    # 加载检测器权重
    if 'ema_model' in detector_path:
        print("📌 加载EMA检测器模型")
        detector.load_state_dict(torch.load(detector_path, map_location=args.device))
    else:
        checkpoint = torch.load(detector_path, map_location=args.device)
        if 'model_state_dict' in checkpoint:
            detector.load_state_dict(checkpoint['model_state_dict'])
        else:
            detector.load_state_dict(checkpoint)

    detector.to(args.device)
    detector.eval()

    # 创建异常感知模型
    model = AnomalyAwareDTransformer(
        n_questions=dataset_config['n_questions'],
        n_pid=dataset_config['n_pid'] if args.with_pid else 0,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_know=args.n_know,
        n_layers=args.n_layers,
        dropout=args.dropout,
        lambda_cl=args.lambda_cl,
        proj=args.proj,
        hard_neg=args.hard_neg,
        window=args.window,
        anomaly_detector=detector,
        anomaly_weight=args.anomaly_weight
    )

    print(f"🧠 模型参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"⚖️  异常权重: {args.anomaly_weight}")

    # 训练器
    trainer = KnowledgeTracingTrainer(
        model=model,
        device=args.device,
        save_dir=os.path.join(args.output_dir, 'anomaly_aware'),
        patience=args.patience
    )

    # 训练
    anomaly_metrics = trainer.train(
        train_loader=train_data,
        val_loader=val_data,
        epochs=args.kt_epochs,
        learning_rate=args.learning_rate,
        use_cl=args.use_cl
    )

    print(f"\n✅ Anomaly-aware training completed!")
    print(f"Best AUC: {anomaly_metrics['auc']:.4f}")

    return os.path.join(args.output_dir, 'anomaly_aware', 'best_model.pt')


def evaluate_models(args, dataset_config, test_data, baseline_path, anomaly_path, detector_path):
    """评估模型性能"""
    print("\n" + "="*60)
    print("PHASE 4: Model Evaluation")
    print("="*60)

    # 加载基线模型
    baseline_model = DTransformer(
        n_questions=dataset_config['n_questions'],
        n_pid=dataset_config['n_pid'] if args.with_pid else 0,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_know=args.n_know,
        n_layers=args.n_layers
    )

    # 处理基线模型的checkpoint格式
    baseline_checkpoint = torch.load(baseline_path, map_location=args.device)
    if isinstance(baseline_checkpoint, dict) and 'model_state_dict' in baseline_checkpoint:
        baseline_model.load_state_dict(baseline_checkpoint['model_state_dict'])
    else:
        baseline_model.load_state_dict(baseline_checkpoint)
    baseline_model.to(args.device)

    # 加载异常检测器
    detector = CausalAnomalyDetector(
        n_questions=dataset_config['n_questions'],
        n_pid=dataset_config['n_pid'] if args.with_pid else 0,
        d_model=args.detector_d_model,
        n_heads=args.detector_n_heads,
        n_layers=args.detector_n_layers,
        dropout=args.detector_dropout,
        window_size=args.window_size
    )

    if 'ema_model' in detector_path:
        detector.load_state_dict(torch.load(detector_path, map_location=args.device))
    else:
        checkpoint = torch.load(detector_path, map_location=args.device)
        if 'model_state_dict' in checkpoint:
            detector.load_state_dict(checkpoint['model_state_dict'])
        else:
            detector.load_state_dict(checkpoint)
    detector.to(args.device)

    # 加载异常感知模型
    anomaly_model = AnomalyAwareDTransformer(
        n_questions=dataset_config['n_questions'],
        n_pid=dataset_config['n_pid'] if args.with_pid else 0,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_know=args.n_know,
        n_layers=args.n_layers,
        anomaly_detector=detector,
        anomaly_weight=args.anomaly_weight
    )

    # 处理异常感知模型的checkpoint格式
    anomaly_checkpoint = torch.load(anomaly_path, map_location=args.device)
    if isinstance(anomaly_checkpoint, dict) and 'model_state_dict' in anomaly_checkpoint:
        anomaly_model.load_state_dict(anomaly_checkpoint['model_state_dict'])
    else:
        anomaly_model.load_state_dict(anomaly_checkpoint)
    anomaly_model.to(args.device)

    # 评估
    evaluator = ComparisonEvaluator()
    results = evaluator.evaluate_models(test_data, baseline_model, anomaly_model, args.device)

    # 打印结果
    evaluator.print_comparison(results)

    # 保存详细结果
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    results['config'] = {
        'anomaly_weight': args.anomaly_weight,
        'detector_epochs': args.detector_epochs,
        'anomaly_ratio': args.anomaly_ratio,
        'optimize_for': args.optimize_for
    }

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📊 Results saved to: {results_path}")

    # 生成评估报告
    generate_evaluation_report(args.output_dir, results)

    return results


def generate_evaluation_report(output_dir: str, results: Dict):
    """生成评估报告"""
    report_path = os.path.join(output_dir, 'evaluation_report.md')

    with open(report_path, 'w') as f:
        f.write("# 异常感知知识追踪评估报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 模型性能对比\n\n")
        f.write("### 基线模型\n")
        for metric, value in results['baseline'].items():
            f.write(f"- {metric.upper()}: {value:.4f}\n")

        f.write("\n### 异常感知模型\n")
        for metric, value in results['anomaly_aware'].items():
            f.write(f"- {metric.upper()}: {value:.4f}\n")

        f.write("\n## 性能提升\n")
        for metric, improvement in results['improvements'].items():
            symbol = "↑" if improvement > 0 else "↓"
            f.write(f"- {metric.upper()}: {improvement:+.2f}% {symbol}\n")

        f.write("\n## 配置信息\n")
        if 'config' in results:
            for key, value in results['config'].items():
                f.write(f"- {key}: {value}\n")

    print(f"📄 评估报告已生成: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Full Anomaly-Aware KT Pipeline (Enhanced)')

    # 基本参数
    parser.add_argument('--dataset', required=True, choices=['assist09', 'assist17', 'algebra05', 'statics'])
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--config', help='Config file path')
    parser.add_argument('-p', '--with_pid', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    # 如果提供了配置文件，从配置文件加载参数
    args, _ = parser.parse_known_args()

    if args.config:
        config = load_config(args.config)
        parser.set_defaults(**config)

    # 数据参数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=64)

    # 基线模型参数
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_know', type=int, default=16)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lambda_cl', type=float, default=0.1)
    parser.add_argument('--proj', action='store_true')
    parser.add_argument('--hard_neg', action='store_true')
    parser.add_argument('--window', type=int, default=1)

    # 异常检测器参数
    parser.add_argument('--detector_d_model', type=int, default=128)
    parser.add_argument('--detector_n_heads', type=int, default=8)
    parser.add_argument('--detector_n_layers', type=int, default=2)
    parser.add_argument('--detector_dropout', type=float, default=0.1)
    parser.add_argument('--window_size', type=int, default=15)
    parser.add_argument('--anomaly_ratio', type=float, default=0.25)
    parser.add_argument('--optimize_for', default='recall', choices=['f1_score', 'auc_roc', 'recall', 'precision'])

    # 增强训练参数
    parser.add_argument('--use_focal_loss', action='store_true', default=True)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--use_mixup', action='store_true', default=True)
    parser.add_argument('--mixup_alpha', type=float, default=0.2)
    parser.add_argument('--use_label_smoothing', action='store_true', default=True)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--use_gradient_accumulation', action='store_true', default=True)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--use_warmup', action='store_true', default=True)
    parser.add_argument('--warmup_epochs', type=int, default=5)

    # 训练参数
    parser.add_argument('--kt_epochs', type=int, default=100)
    parser.add_argument('--detector_epochs', type=int, default=40)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--detector_lr', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--detector_patience', type=int, default=15)
    parser.add_argument('--use_cl', action='store_true')

    # 异常感知参数
    parser.add_argument('--anomaly_weight', type=float, default=0.5)

    # 控制参数
    parser.add_argument('--skip_baseline', action='store_true', help='Skip baseline training')
    parser.add_argument('--skip_detector', action='store_true', help='Skip detector training')
    parser.add_argument('--baseline_path', help='Path to existing baseline model')
    parser.add_argument('--detector_path', help='Path to existing detector model')
    parser.add_argument('--use_aggressive_strategy', action='store_true', default=False,
                        help='使用激进策略处理严重不平衡')

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 检查文件存在性
    if args.skip_baseline:
        if not args.baseline_path:
            print("ERROR: --skip_baseline requires --baseline_path to be specified")
            sys.exit(1)
        if not os.path.exists(args.baseline_path):
            print(f"ERROR: Baseline model file not found: {args.baseline_path}")
            sys.exit(1)
        print(f"✓ Baseline model found: {args.baseline_path}")

    if args.skip_detector:
        if not args.detector_path:
            print("ERROR: --skip_detector requires --detector_path to be specified")
            sys.exit(1)
        if not os.path.exists(args.detector_path):
            print(f"ERROR: Detector model file not found: {args.detector_path}")
            sys.exit(1)
        print(f"✓ Detector model found: {args.detector_path}")

    # 设置输出目录
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"output/{args.dataset}_{timestamp}_enhanced"

    os.makedirs(args.output_dir, exist_ok=True)

    # 保存配置
    config_save_path = os.path.join(args.output_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)

    print("\n🚀 异常感知知识追踪训练流程（增强版）")
    print("="*60)
    print("配置信息:")
    print(f"  数据集: {args.dataset}")
    print(f"  设备: {args.device}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  优化目标: {args.optimize_for}")
    print(f"  异常比例: {args.anomaly_ratio}")
    print(f"  异常权重: {args.anomaly_weight}")

    # 准备数据
    print("\n📊 准备数据...")
    train_data, val_data, test_data, dataset_config = prepare_data(
        args.dataset, args.data_dir, args.batch_size, args.test_batch_size
    )
    print(f"  训练批次: {len(train_data)}")
    print(f"  验证批次: {len(val_data)}")
    print(f"  测试批次: {len(test_data)}")

    # 1. 训练基线模型
    if not args.skip_baseline:
        baseline_path = train_baseline_model(args, dataset_config, train_data, val_data)
    else:
        baseline_path = args.baseline_path
        print(f"\n📌 使用已有基线模型: {baseline_path}")

    # 2. 训练异常检测器（增强版）
    if not args.skip_detector:
        detector_path = train_anomaly_detector_enhanced(args, dataset_config, train_data, val_data)
    else:
        detector_path = args.detector_path
        print(f"\n📌 使用已有检测器: {detector_path}")

    # 3. 训练异常感知模型
    anomaly_path = train_anomaly_aware_model(args, dataset_config, train_data, val_data, detector_path)

    # 4. 评估结果
    results = evaluate_models(args, dataset_config, test_data, baseline_path, anomaly_path, detector_path)

    print("\n" + "="*60)
    print("🎉 训练流程完成!")
    print("="*60)

    # 检查是否达到目标
    improvement = results['improvements']['auc']
    if improvement >= 1.0:
        print(f"✅ 成功: 目标达成! AUC提升 {improvement:.2f}%")
    else:
        print(f"⚠️  未达标: AUC提升 {improvement:.2f}% (目标: ≥1%)")
        print("\n💡 改进建议:")
        print("  1. 调整异常权重 (--anomaly_weight)，当前值: {}".format(args.anomaly_weight))
        print("  2. 增加检测器训练轮数 (--detector_epochs)")
        print("  3. 尝试不同的异常比例 (--anomaly_ratio)")
        print("  4. 调整优化目标 (--optimize_for)")
        print("  5. 增加训练数据的批次大小")

    print(f"\n📁 所有结果已保存至: {args.output_dir}")


if __name__ == '__main__':
    main()