"""
异常感知知识追踪训练器

实现渐进式训练策略：
1. 冻结预训练模型，只训练融合层
2. 解冻异常检测器，联合训练
3. 端到端微调（可选）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

from .model import AnomalyAwareKT


class AnomalyAwareTrainer:
    """异常感知知识追踪训练器"""
    
    def __init__(self,
                 model: AnomalyAwareKT,
                 device: str = 'cuda',
                 learning_rate: float = 0.001,
                 save_dir: str = 'output/stage3',
                 patience: int = 10):
        """
        初始化训练器
        
        Args:
            model: 异常感知知识追踪模型
            device: 训练设备
            learning_rate: 学习率
            save_dir: 保存目录
            patience: 早停耐心值
        """
        self.model = model
        self.device = device
        self.save_dir = save_dir
        self.patience = patience
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 优化器（只优化可训练参数）
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(trainable_params, lr=learning_rate)
        
        # 训练状态
        self.best_auc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.training_history = []
        
    def progressive_train(self,
                         train_loader,
                         val_loader,
                         stage1_epochs: int = 10,
                         stage2_epochs: int = 20,
                         stage3_epochs: int = 10,
                         lambda_anomaly: float = 0.1) -> Dict:
        """
        渐进式训练
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            stage1_epochs: 阶段1训练轮数（只训练融合层）
            stage2_epochs: 阶段2训练轮数（联合训练）
            stage3_epochs: 阶段3训练轮数（端到端微调）
            lambda_anomaly: 异常一致性损失权重
            
        Returns:
            训练结果
        """
        print("🚀 开始渐进式训练")
        
        total_epochs = stage1_epochs + stage2_epochs + stage3_epochs
        current_epoch = 0
        
        # 阶段1：只训练融合层
        print(f"\n📚 阶段1：融合层训练 ({stage1_epochs} epochs)")
        print("🔒 基线模型和异常检测器已冻结")
        
        for epoch in range(1, stage1_epochs + 1):
            current_epoch += 1
            stage_info = {'stage': 1, 'stage_epoch': epoch, 'total_epoch': current_epoch}
            
            train_metrics = self._train_epoch(train_loader, lambda_anomaly, stage_info)
            val_metrics = self._validate_epoch(val_loader, lambda_anomaly)
            
            self._log_epoch_results(current_epoch, train_metrics, val_metrics, stage_info)
            
            # 早停检查
            if self._early_stopping_check(val_metrics['auc'], current_epoch):
                break
        
        # 阶段2：解冻异常检测器，联合训练
        if stage2_epochs > 0:
            print(f"\n🔓 阶段2：联合训练 ({stage2_epochs} epochs)")
            print("🔓 异常检测器已解冻")
            
            # 解冻异常检测器
            for param in self.model.anomaly_detector.parameters():
                param.requires_grad = True
            
            # 更新优化器
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = optim.Adam(trainable_params, lr=self.optimizer.param_groups[0]['lr'] * 0.5)
            
            for epoch in range(1, stage2_epochs + 1):
                current_epoch += 1
                stage_info = {'stage': 2, 'stage_epoch': epoch, 'total_epoch': current_epoch}
                
                train_metrics = self._train_epoch(train_loader, lambda_anomaly, stage_info)
                val_metrics = self._validate_epoch(val_loader, lambda_anomaly)
                
                self._log_epoch_results(current_epoch, train_metrics, val_metrics, stage_info)
                
                if self._early_stopping_check(val_metrics['auc'], current_epoch):
                    break
        
        # 阶段3：端到端微调
        if stage3_epochs > 0:
            print(f"\n🎯 阶段3：端到端微调 ({stage3_epochs} epochs)")
            print("🔓 所有模型已解冻")
            
            # 解冻所有模型
            self.model.unfreeze_pretrained_models()
            
            # 更新优化器（使用更小的学习率）
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = optim.Adam(trainable_params, lr=self.optimizer.param_groups[0]['lr'] * 0.1)
            
            for epoch in range(1, stage3_epochs + 1):
                current_epoch += 1
                stage_info = {'stage': 3, 'stage_epoch': epoch, 'total_epoch': current_epoch}
                
                train_metrics = self._train_epoch(train_loader, lambda_anomaly, stage_info)
                val_metrics = self._validate_epoch(val_loader, lambda_anomaly)
                
                self._log_epoch_results(current_epoch, train_metrics, val_metrics, stage_info)
                
                if self._early_stopping_check(val_metrics['auc'], current_epoch):
                    break
        
        # 训练完成
        final_result = {
            'best_auc': self.best_auc,
            'best_epoch': self.best_epoch,
            'total_epochs': current_epoch,
            'training_history': self.training_history,
            'model_info': self.model.get_model_info()
        }
        
        print(f"\n🎉 渐进式训练完成！")
        print(f"📈 最佳AUC: {self.best_auc:.4f} (Epoch {self.best_epoch})")
        
        return final_result
    
    def _train_epoch(self, train_loader, lambda_anomaly: float, stage_info: Dict) -> Dict:
        """训练一个epoch"""
        self.model.train()
        
        total_losses = {'total': 0.0, 'kt': 0.0, 'anomaly': 0.0}
        all_predictions = []
        all_targets = []
        
        desc = f"Stage {stage_info['stage']} Training"
        for batch in tqdm(train_loader, desc=desc):
            # 获取批次数据
            q, s = batch.get("q", "s")
            
            # 安全地获取pid字段
            try:
                pid = batch.get("pid")
            except (KeyError, AttributeError):
                pid = None
            
            q = q.to(self.device)
            s = s.to(self.device)
            if pid is not None:
                pid = pid.to(self.device)
            
            # 构建目标序列（下一题的答案）
            target = s[:, 1:].clone()  # 向前移动一位
            q_input = q[:, :-1]        # 去掉最后一位
            s_input = s[:, :-1]        # 去掉最后一位
            if pid is not None:
                pid_input = pid[:, :-1]
            else:
                pid_input = None
            
            # 前向传播
            losses = self.model.get_loss(q_input, s_input, target, pid_input, lambda_anomaly)
            
            # 反向传播
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 记录损失
            total_losses['total'] += losses['total_loss'].item()
            total_losses['kt'] += losses['kt_loss'].item()
            total_losses['anomaly'] += losses['anomaly_consistency_loss'].item()
            
            # 收集预测结果用于AUC计算
            with torch.no_grad():
                predictions = self.model(q_input, s_input, pid_input)
                mask = (target >= 0)
                
                all_predictions.extend(predictions[mask].cpu().numpy())
                all_targets.extend(target[mask].cpu().numpy())
        
        # 计算平均损失
        avg_losses = {k: v / len(train_loader) for k, v in total_losses.items()}
        
        # 计算AUC
        if all_predictions:
            try:
                train_auc = roc_auc_score(all_targets, all_predictions)
            except:
                train_auc = 0.5
        else:
            train_auc = 0.5
        
        return {
            'auc': train_auc,
            'total_loss': avg_losses['total'],
            'kt_loss': avg_losses['kt'],
            'anomaly_loss': avg_losses['anomaly'],
            'sample_count': len(all_predictions)
        }
    
    def _validate_epoch(self, val_loader, lambda_anomaly: float) -> Dict:
        """验证一个epoch"""
        self.model.eval()
        
        total_losses = {'total': 0.0, 'kt': 0.0, 'anomaly': 0.0}
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # 获取批次数据
                q, s = batch.get("q", "s")
                
                try:
                    pid = batch.get("pid")
                except (KeyError, AttributeError):
                    pid = None
                
                q = q.to(self.device)
                s = s.to(self.device)
                if pid is not None:
                    pid = pid.to(self.device)
                
                # 构建目标序列
                target = s[:, 1:].clone()
                q_input = q[:, :-1]
                s_input = s[:, :-1]
                if pid is not None:
                    pid_input = pid[:, :-1]
                else:
                    pid_input = None
                
                # 前向传播
                losses = self.model.get_loss(q_input, s_input, target, pid_input, lambda_anomaly)
                predictions = self.model(q_input, s_input, pid_input)
                
                # 记录损失
                total_losses['total'] += losses['total_loss'].item()
                total_losses['kt'] += losses['kt_loss'].item()
                total_losses['anomaly'] += losses['anomaly_consistency_loss'].item()
                
                # 收集预测结果
                mask = (target >= 0)
                all_predictions.extend(predictions[mask].cpu().numpy())
                all_targets.extend(target[mask].cpu().numpy())
        
        # 计算平均损失
        avg_losses = {k: v / len(val_loader) for k, v in total_losses.items()}
        
        # 计算AUC
        if all_predictions:
            try:
                val_auc = roc_auc_score(all_targets, all_predictions)
            except:
                val_auc = 0.5
        else:
            val_auc = 0.5
        
        return {
            'auc': val_auc,
            'total_loss': avg_losses['total'],
            'kt_loss': avg_losses['kt'],
            'anomaly_loss': avg_losses['anomaly'],
            'sample_count': len(all_predictions)
        }
    
    def _log_epoch_results(self, epoch: int, train_metrics: Dict, val_metrics: Dict, stage_info: Dict):
        """记录epoch结果"""
        print(f"\nEpoch {epoch} (Stage {stage_info['stage']}-{stage_info['stage_epoch']})")
        print(f"训练 - AUC: {train_metrics['auc']:.4f}, "
              f"Total Loss: {train_metrics['total_loss']:.4f}, "
              f"KT Loss: {train_metrics['kt_loss']:.4f}")
        print(f"验证 - AUC: {val_metrics['auc']:.4f}, "
              f"Total Loss: {val_metrics['total_loss']:.4f}, "
              f"样本: {val_metrics['sample_count']}")
        
        # 记录训练历史
        epoch_result = {
            'epoch': epoch,
            'stage_info': stage_info,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
        self.training_history.append(epoch_result)
    
    def _early_stopping_check(self, val_auc: float, epoch: int) -> bool:
        """早停检查"""
        if val_auc > self.best_auc:
            self.best_auc = val_auc
            self.best_epoch = epoch
            self.patience_counter = 0
            
            # 保存最佳模型
            self._save_model(epoch, val_auc)
            print(f"✅ 新的最佳模型 (AUC: {self.best_auc:.4f})")
            return False
        else:
            self.patience_counter += 1
            print(f"⏳ 无改进 ({self.patience_counter}/{self.patience})")
            
            if self.patience_counter >= self.patience:
                print(f"🛑 早停触发！最佳AUC: {self.best_auc:.4f} (Epoch {self.best_epoch})")
                return True
        
        return False
    
    def _save_model(self, epoch: int, auc: float):
        """保存模型"""
        model_path = os.path.join(self.save_dir, 'best_anomaly_aware_kt.pt')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'auc': auc,
            'best_auc': self.best_auc,
            'model_info': self.model.get_model_info()
        }, model_path)
    
    def load_model(self, model_path: str) -> Dict:
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_auc = checkpoint.get('best_auc', 0.0)
        
        return checkpoint.get('model_info', {})
