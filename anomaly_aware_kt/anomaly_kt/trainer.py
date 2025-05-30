"""
训练器模块

提供异常检测器和知识追踪模型的训练功能。
"""

import os
import json
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, Optional, Tuple

from .generator import AnomalyGenerator
from .evaluator import AnomalyEvaluator


class BaseTrainer:
    """基础训练器"""

    def __init__(self,
                 model: nn.Module,
                 device: str = 'cpu',
                 save_dir: str = 'output',
                 patience: int = 10):
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        self.patience = patience

        os.makedirs(save_dir, exist_ok=True)

        self.history = defaultdict(list)
        self.best_metrics = {}
        self.best_epoch = 0

    def save_checkpoint(self, epoch: int, optimizer, metrics: Dict, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch + 1,  # 保存实际轮次
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'history': dict(self.history)
        }

        # 总是保存当前轮次的模型
        epoch_path = os.path.join(self.save_dir, f'model_epoch_{epoch+1}.pt')
        try:
            torch.save(checkpoint, epoch_path)
            print(f"  💾 保存模型: epoch_{epoch+1}.pt")
        except Exception as e:
            print(f"  ❌ 保存失败 epoch_{epoch+1}: {e}")

        # 如果是最佳模型，额外保存一份
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            try:
                torch.save(checkpoint, best_path)
                print(f"  🏆 保存最佳模型: best_model.pt")
            except Exception as e:
                print(f"  ❌ 保存最佳模型失败: {e}")

    def load_checkpoint(self, path: str, optimizer: Optional = None):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})


class AnomalyDetectorTrainer(BaseTrainer):
    """异常检测器训练器"""

    def __init__(self, model: nn.Module, device: str = 'cpu',
                 save_dir: str = 'output/detector', patience: int = 10):
        super().__init__(model, device, save_dir, patience)
        self.generator = AnomalyGenerator()
        self.evaluator = AnomalyEvaluator()

    def train(self,
              train_loader,
              val_loader,
              epochs: int = 30,
              learning_rate: float = 1e-3,
              anomaly_ratio: float = 0.1,
              optimize_for: str = 'f1_score') -> Dict:
        """训练异常检测器"""

        # 优化器和调度器
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )

        # 保存配置
        config = {
            'epochs': epochs,
            'learning_rate': learning_rate,
            'anomaly_ratio': anomaly_ratio,
            'optimize_for': optimize_for
        }
        with open(os.path.join(self.save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        # 训练循环
        no_improve = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            # 训练
            train_loss = self._train_epoch(
                train_loader, optimizer, anomaly_ratio
            )

            # 验证
            val_metrics = self._validate_epoch(
                val_loader, anomaly_ratio
            )

            # 记录
            self.history['train_loss'].append(train_loss)
            for k, v in val_metrics.items():
                self.history[f'val_{k}'].append(v)

            # 调整学习率
            scheduler.step(val_metrics[optimize_for])

            # 打印指标
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val - Recall: {val_metrics['recall']:.3f}, "
                  f"Precision: {val_metrics['precision']:.3f}, "
                  f"F1: {val_metrics['f1_score']:.3f}, "
                  f"AUC: {val_metrics['auc_roc']:.3f}")

            # 检查最佳
            current_score = val_metrics[optimize_for]
            if not self.best_metrics or current_score > self.best_metrics.get(optimize_for, 0):
                self.best_metrics = val_metrics.copy()
                self.best_epoch = epoch + 1
                self.save_checkpoint(epoch, optimizer, val_metrics, is_best=True)
                no_improve = 0
                print(f"✓ New best {optimize_for}: {current_score:.4f}")
            else:
                no_improve += 1

            # 早停
            if no_improve >= self.patience:
                print(f"Early stopping after {self.patience} epochs")
                break

        print(f"\nBest epoch: {self.best_epoch}")
        print(f"Best metrics: {self.best_metrics}")

        return self.best_metrics

    def _train_epoch(self, train_loader, optimizer, anomaly_ratio):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        n_batches = 0

        for batch in tqdm(train_loader, desc="Training"):
            q, s, pid = self._get_batch_data(batch)

            # 生成异常
            s_anomaly, labels = self.generator.generate_anomalies(
                q, s, anomaly_ratio=anomaly_ratio
            )
            labels = labels.to(self.device)

            # 前向传播
            loss = self.model.get_loss(q, s_anomaly, labels, pid)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def _validate_epoch(self, val_loader, anomaly_ratio):
        """验证一个epoch"""
        self.model.eval()
        self.evaluator.reset()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
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


class KnowledgeTracingTrainer(BaseTrainer):
    """知识追踪模型训练器"""

    def __init__(self, model: nn.Module, device: str = 'cpu',
                 save_dir: str = 'output/kt_model', patience: int = 10):
        super().__init__(model, device, save_dir, patience)

    def train(self,
              train_loader,
              val_loader,
              epochs: int = 100,
              learning_rate: float = 1e-3,
              use_cl: bool = True) -> Dict:
        """训练知识追踪模型"""

        # 优化器
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )

        # 保存配置
        config = {
            'epochs': epochs,
            'learning_rate': learning_rate,
            'use_cl': use_cl,
            'anomaly_weight': getattr(self.model, 'anomaly_weight', 0)
        }
        with open(os.path.join(self.save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        # 训练循环
        no_improve = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            # 训练
            train_metrics = self._train_epoch(
                train_loader, optimizer, use_cl
            )

            # 验证
            val_metrics = self._validate_epoch(val_loader)

            # 记录
            for k, v in train_metrics.items():
                self.history[f'train_{k}'].append(v)
            for k, v in val_metrics.items():
                self.history[f'val_{k}'].append(v)

            # 打印
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val - ACC: {val_metrics['acc']:.3f}, "
                  f"AUC: {val_metrics['auc']:.3f}, "
                  f"MAE: {val_metrics['mae']:.3f}")

            # 检查最佳
            current_auc = val_metrics['auc']
            if not self.best_metrics or current_auc > self.best_metrics.get('auc', 0):
                self.best_metrics = val_metrics.copy()
                self.best_epoch = epoch + 1
                self.save_checkpoint(epoch, optimizer, val_metrics, is_best=True)
                no_improve = 0
                print(f"✓ New best AUC: {current_auc:.4f}")
            else:
                no_improve += 1

            # 早停
            if no_improve >= self.patience:
                print(f"Early stopping after {self.patience} epochs")
                break

        return self.best_metrics

    def _train_epoch(self, train_loader, optimizer, use_cl):
        """训练一个epoch"""
        self.model.train()

        total_loss = 0
        total_pred_loss = 0
        total_cl_loss = 0
        n_batches = 0

        for batch in tqdm(train_loader, desc="Training"):
            q, s, pid = self._get_batch_data(batch)

            # 计算损失
            if use_cl and hasattr(self.model, 'get_cl_loss'):
                loss, pred_loss, cl_loss = self.model.get_cl_loss(q, s, pid)
                total_pred_loss += pred_loss.item() if pred_loss is not None else 0
                total_cl_loss += cl_loss.item() if cl_loss is not None else 0
            else:
                loss = self.model.get_loss(q, s, pid)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        metrics = {'loss': total_loss / n_batches}
        if use_cl:
            metrics['pred_loss'] = total_pred_loss / n_batches
            metrics['cl_loss'] = total_cl_loss / n_batches

        return metrics

    def _validate_epoch(self, val_loader):
        """验证一个epoch"""
        self.model.eval()

        from DTransformer.eval import Evaluator
        evaluator = Evaluator()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                q, s, pid = self._get_batch_data(batch)

                # 预测
                if hasattr(self.model, 'predict_with_anomaly'):
                    y, *_ = self.model.predict_with_anomaly(q, s, pid)
                else:
                    y, *_ = self.model.predict(q, s, pid)

                # 评估
                evaluator.evaluate(s, torch.sigmoid(y))

        return evaluator.report()

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