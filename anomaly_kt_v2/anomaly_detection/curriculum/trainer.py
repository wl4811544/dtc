"""
课程学习训练器

实现基于课程学习的异常检测器训练。
完全基于DTransformer原始代码，不依赖任何anomaly_kt模块。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import os
from tqdm import tqdm

from .scheduler import CurriculumScheduler
from .difficulty_estimator import DifficultyEstimator
from ..generators.curriculum_generator import CurriculumAnomalyGenerator
from ..generators.baseline_generator import BaselineAnomalyGenerator


class CurriculumTrainer:
    """课程学习训练器"""
    
    def __init__(self,
                 model: nn.Module,
                 device: str = 'cuda',
                 learning_rate: float = 0.001,
                 save_dir: str = 'output/stage2',
                 patience: int = 10,
                 with_pid: bool = True):
        """
        初始化课程学习训练器

        Args:
            model: 异常检测模型
            device: 训练设备
            learning_rate: 学习率
            save_dir: 保存目录
            patience: 早停耐心值
            with_pid: 是否使用问题ID
        """
        self.model = model
        self.device = device
        self.save_dir = save_dir
        self.patience = patience
        self.with_pid = with_pid
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 优化器
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 组件
        self.curriculum_generator = CurriculumAnomalyGenerator()
        self.baseline_generator = BaselineAnomalyGenerator()
        self.difficulty_estimator = DifficultyEstimator()
        
        # 训练状态
        self.best_auc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        
    def train(self,
              train_loader,
              val_loader,
              epochs: int = 50,
              curriculum_config: Optional[Dict] = None) -> Dict:
        """
        执行课程学习训练
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            curriculum_config: 课程学习配置
            
        Returns:
            训练结果
        """
        # 默认课程配置
        if curriculum_config is None:
            curriculum_config = {
                'initial_difficulty': 0.1,
                'final_difficulty': 0.8,
                'schedule_type': 'linear',
                'warmup_epochs': 5
            }
        
        # 创建课程调度器
        scheduler = CurriculumScheduler(
            total_epochs=epochs,
            **curriculum_config
        )
        
        print(f"开始课程学习训练，共{epochs}轮")
        print(f"课程配置: {curriculum_config}")
        
        training_history = []
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            
            # 更新课程状态
            curriculum_state = scheduler.step(epoch)
            print(f"课程状态: 难度={curriculum_state['difficulty']:.3f}, "
                  f"阶段={curriculum_state['phase']}")
            
            # 训练阶段
            train_metrics = self._train_epoch(train_loader, curriculum_state)
            
            # 验证阶段
            val_metrics = self._validate_epoch(val_loader)
            
            # 记录训练历史
            epoch_result = {
                'epoch': epoch,
                'curriculum_state': curriculum_state,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }
            training_history.append(epoch_result)
            
            # 打印详细结果
            print(f"训练 - Loss: {train_metrics['loss']:.4f}, "
                  f"AUC: {train_metrics['auc']:.4f}, "
                  f"P: {train_metrics.get('precision', 0):.3f}, "
                  f"R: {train_metrics.get('recall', 0):.3f}, "
                  f"F1: {train_metrics.get('f1', 0):.3f}")
            print(f"验证 - Loss: {val_metrics['loss']:.4f}, "
                  f"AUC: {val_metrics['auc']:.4f}, "
                  f"P: {val_metrics.get('precision', 0):.3f}, "
                  f"R: {val_metrics.get('recall', 0):.3f}, "
                  f"F1: {val_metrics.get('f1', 0):.3f}, "
                  f"样本: {val_metrics['sample_count']}")
            
            # 早停检查
            if val_metrics['auc'] > self.best_auc:
                self.best_auc = val_metrics['auc']
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # 保存最佳模型
                self._save_model(epoch, val_metrics)
                print(f"✅ 新的最佳模型 (AUC: {self.best_auc:.4f})")
            else:
                self.patience_counter += 1
                print(f"⏳ 无改进 ({self.patience_counter}/{self.patience})")
            
            if self.patience_counter >= self.patience:
                print(f"🛑 早停触发！最佳AUC: {self.best_auc:.4f} (Epoch {self.best_epoch})")
                break
        
        # 训练完成
        final_result = {
            'best_auc': self.best_auc,
            'best_epoch': self.best_epoch,
            'total_epochs': epoch,
            'training_history': training_history
        }
        
        return final_result
    
    def _train_epoch(self, train_loader, curriculum_state: Dict) -> Dict:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_difficulties = []
        
        for batch in tqdm(train_loader, desc="Training"):
            # 获取批次数据
            if self.with_pid:
                q, s, pid = batch.get("q", "s", "pid")
            else:
                q, s = batch.get("q", "s")
                pid = None
            
            q = q.to(self.device)
            s = s.to(self.device)
            if pid is not None:
                pid = pid.to(self.device)
            
            # 生成异常数据
            s_anomaly, anomaly_labels, difficulty_scores = self._generate_curriculum_data(
                q, s, curriculum_state
            )
            
            # 前向传播
            logits = self.model(q, s_anomaly, pid)
            loss = self.model.get_loss(q, s_anomaly, anomaly_labels, pid)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 记录结果
            total_loss += loss.item()
            
            # 收集预测结果用于评估
            with torch.no_grad():
                pred_probs = torch.sigmoid(logits)
                mask = (s_anomaly >= 0)
                
                all_predictions.extend(pred_probs[mask].cpu().numpy())
                all_labels.extend(anomaly_labels[mask].cpu().numpy())
                all_difficulties.extend(difficulty_scores[mask].cpu().numpy())
        
        # 计算训练指标
        avg_loss = total_loss / len(train_loader)

        # 评估检测性能
        if all_predictions:
            predictions_tensor = torch.tensor(all_predictions)
            labels_tensor = torch.tensor(all_labels)
            difficulties_tensor = torch.tensor(all_difficulties)

            difficulty_result = self.difficulty_estimator.estimate_detection_difficulty(
                predictions_tensor, labels_tensor, difficulties_tensor
            )

            basic_metrics = difficulty_result['basic_metrics']
            train_auc = basic_metrics['auc']
            precision = basic_metrics['precision']
            recall = basic_metrics['recall']
            f1 = basic_metrics['f1']
        else:
            train_auc = 0.0
            precision = 0.0
            recall = 0.0
            f1 = 0.0

        return {
            'loss': avg_loss,
            'auc': train_auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'sample_count': len(all_predictions)
        }
    
    def _validate_epoch(self, val_loader) -> Dict:
        """验证一个epoch"""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # 获取批次数据
                if self.with_pid:
                    q, s, pid = batch.get("q", "s", "pid")
                else:
                    q, s = batch.get("q", "s")
                    pid = None
                
                q = q.to(self.device)
                s = s.to(self.device)
                if pid is not None:
                    pid = pid.to(self.device)
                
                # 生成验证异常数据（使用固定策略）
                s_anomaly, anomaly_labels = self.baseline_generator.generate_baseline_anomalies(
                    q, s, strategy='random_flip', anomaly_ratio=0.1
                )
                
                # 前向传播
                logits = self.model(q, s_anomaly, pid)
                loss = self.model.get_loss(q, s_anomaly, anomaly_labels, pid)
                
                total_loss += loss.item()
                
                # 收集预测结果
                pred_probs = torch.sigmoid(logits)
                mask = (s_anomaly >= 0)
                
                all_predictions.extend(pred_probs[mask].cpu().numpy())
                all_labels.extend(anomaly_labels[mask].cpu().numpy())
        
        # 计算验证指标
        avg_loss = total_loss / len(val_loader)

        if all_predictions:
            predictions_tensor = torch.tensor(all_predictions)
            labels_tensor = torch.tensor(all_labels)

            difficulty_result = self.difficulty_estimator.estimate_detection_difficulty(
                predictions_tensor, labels_tensor
            )

            basic_metrics = difficulty_result['basic_metrics']
            val_auc = basic_metrics['auc']
            precision = basic_metrics['precision']
            recall = basic_metrics['recall']
            f1 = basic_metrics['f1']
        else:
            val_auc = 0.0
            precision = 0.0
            recall = 0.0
            f1 = 0.0

        return {
            'loss': avg_loss,
            'auc': val_auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'sample_count': len(all_predictions)
        }
    
    def _generate_curriculum_data(self, q: torch.Tensor, s: torch.Tensor,
                                 curriculum_state: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """根据课程状态生成训练数据"""
        difficulty_levels = [1, 2, 3, 4]
        level_weights = curriculum_state['level_weights']
        anomaly_ratio = curriculum_state['anomaly_ratio']
        
        # 生成课程异常
        s_anomaly, anomaly_labels, difficulty_scores = self.curriculum_generator.generate_curriculum_anomalies(
            q, s,
            difficulty_levels=difficulty_levels,
            level_weights=level_weights,
            anomaly_ratio=anomaly_ratio,
            include_baseline=True,
            baseline_ratio=0.3
        )
        
        return s_anomaly, anomaly_labels, difficulty_scores
    
    def _save_model(self, epoch: int, metrics: Dict):
        """保存模型"""
        model_path = os.path.join(self.save_dir, 'best_anomaly_detector.pt')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_auc': self.best_auc
        }, model_path)
    
    def load_model(self, model_path: str) -> Dict:
        """加载模型"""
        # PyTorch 2.6+ 兼容性：禁用 weights_only 以支持旧模型文件
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_auc = checkpoint.get('best_auc', 0.0)
        
        return checkpoint.get('metrics', {})
