"""
异常感知知识追踪模型

将基线知识追踪模型与异常检测器融合，
实现异常感知的知识追踪，目标提升AUC 0.05-0.1。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Union
import sys
import os

# 添加DTransformer路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .fusion import AnomalyAwareFusion, ContextEnhancer


class AnomalyAwareKT(nn.Module):
    """
    异常感知知识追踪模型
    
    核心架构：
    1. 基线知识追踪模型（冻结）
    2. 异常检测器（冻结）  
    3. 异常感知融合层（可训练）
    """
    
    def __init__(self,
                 baseline_model: nn.Module,
                 anomaly_detector: nn.Module,
                 d_model: int,
                 fusion_type: str = 'attention',
                 enable_context_enhancement: bool = True,
                 freeze_pretrained: bool = True,
                 dropout: float = 0.1):
        """
        初始化异常感知知识追踪模型
        
        Args:
            baseline_model: 第一阶段训练的基线模型
            anomaly_detector: 第二阶段训练的异常检测器
            d_model: 模型维度
            fusion_type: 融合类型
            enable_context_enhancement: 是否启用上下文增强
            freeze_pretrained: 是否冻结预训练模型
            dropout: Dropout率
        """
        super().__init__()
        
        self.baseline_model = baseline_model
        self.anomaly_detector = anomaly_detector
        self.d_model = d_model
        self.enable_context_enhancement = enable_context_enhancement
        
        # 冻结预训练模型
        if freeze_pretrained:
            self._freeze_pretrained_models()
        
        # 异常感知融合层
        self.anomaly_fusion = AnomalyAwareFusion(
            d_model=d_model,
            fusion_type=fusion_type,
            dropout=dropout
        )
        
        # 上下文增强器（可选）
        if enable_context_enhancement:
            self.context_enhancer = ContextEnhancer(d_model=d_model)
        
        # 最终预测层
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # 多任务学习的辅助损失
        self.anomaly_consistency_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
    def _freeze_pretrained_models(self):
        """冻结预训练模型的参数"""
        for param in self.baseline_model.parameters():
            param.requires_grad = False
            
        for param in self.anomaly_detector.parameters():
            param.requires_grad = False
            
        print("✅ 预训练模型已冻结")
        
    def unfreeze_pretrained_models(self):
        """解冻预训练模型（用于端到端微调）"""
        for param in self.baseline_model.parameters():
            param.requires_grad = True
            
        for param in self.anomaly_detector.parameters():
            param.requires_grad = True
            
        print("🔓 预训练模型已解冻，开始端到端训练")
    
    def forward(self, 
                q: torch.Tensor, 
                s: torch.Tensor, 
                pid: Optional[torch.Tensor] = None,
                return_anomaly_info: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        前向传播
        
        Args:
            q: 问题序列 [batch_size, seq_len]
            s: 答案序列 [batch_size, seq_len]
            pid: 问题ID序列 [batch_size, seq_len]
            return_anomaly_info: 是否返回异常信息
            
        Returns:
            predictions: 知识追踪预测 [batch_size, seq_len]
            anomaly_info: 异常相关信息（可选）
        """
        batch_size, seq_len = q.shape
        
        # 1. 基线知识追踪模型
        with torch.no_grad() if hasattr(self.baseline_model, 'training') and not self.baseline_model.training else torch.enable_grad():
            baseline_output = self.baseline_model.get_hidden_states(q, s, pid)  # [batch_size, seq_len, d_model]
        
        # 2. 异常检测
        with torch.no_grad() if hasattr(self.anomaly_detector, 'training') and not self.anomaly_detector.training else torch.enable_grad():
            anomaly_logits = self.anomaly_detector(q, s, pid)  # [batch_size, seq_len]
            anomaly_probs = torch.sigmoid(anomaly_logits)
        
        # 3. 上下文增强（可选）
        if self.enable_context_enhancement:
            baseline_output = self.context_enhancer(baseline_output, anomaly_probs)
        
        # 4. 异常感知融合
        fused_output, adjustment_weights = self.anomaly_fusion(
            baseline_output, anomaly_probs
        )
        
        # 5. 最终预测
        predictions = self.prediction_head(fused_output).squeeze(-1)  # [batch_size, seq_len]
        
        if return_anomaly_info:
            # 异常一致性预测（用于多任务学习）
            anomaly_consistency = self.anomaly_consistency_head(fused_output).squeeze(-1)
            
            anomaly_info = {
                'anomaly_probs': anomaly_probs,
                'adjustment_weights': adjustment_weights,
                'anomaly_consistency': anomaly_consistency,
                'baseline_output': baseline_output,
                'fused_output': fused_output
            }
            return predictions, anomaly_info
        
        return predictions
    
    def get_loss(self, 
                 q: torch.Tensor, 
                 s: torch.Tensor, 
                 target: torch.Tensor,
                 pid: Optional[torch.Tensor] = None,
                 lambda_anomaly: float = 0.1) -> Dict[str, torch.Tensor]:
        """
        计算损失函数
        
        Args:
            q: 问题序列
            s: 答案序列  
            target: 目标序列（下一题的答案）
            pid: 问题ID序列
            lambda_anomaly: 异常一致性损失权重
            
        Returns:
            losses: 损失字典
        """
        # 前向传播
        predictions, anomaly_info = self.forward(q, s, pid, return_anomaly_info=True)
        
        # 创建有效位置的掩码
        mask = (target >= 0)
        
        # 主要知识追踪损失
        kt_loss = F.binary_cross_entropy(
            predictions[mask], 
            target[mask].float()
        )
        
        # 异常一致性损失（多任务学习）
        anomaly_consistency_loss = F.binary_cross_entropy(
            anomaly_info['anomaly_consistency'][mask],
            anomaly_info['anomaly_probs'][mask]
        )
        
        # 总损失
        total_loss = kt_loss + lambda_anomaly * anomaly_consistency_loss
        
        return {
            'total_loss': total_loss,
            'kt_loss': kt_loss,
            'anomaly_consistency_loss': anomaly_consistency_loss,
            'lambda_anomaly': lambda_anomaly
        }
    
    def get_trainable_parameters(self) -> Dict[str, torch.nn.Parameter]:
        """获取可训练参数"""
        trainable_params = {}
        
        # 融合层参数
        for name, param in self.anomaly_fusion.named_parameters():
            if param.requires_grad:
                trainable_params[f'fusion.{name}'] = param
        
        # 上下文增强器参数
        if self.enable_context_enhancement:
            for name, param in self.context_enhancer.named_parameters():
                if param.requires_grad:
                    trainable_params[f'context.{name}'] = param
        
        # 预测头参数
        for name, param in self.prediction_head.named_parameters():
            if param.requires_grad:
                trainable_params[f'prediction.{name}'] = param
        
        # 异常一致性头参数
        for name, param in self.anomaly_consistency_head.named_parameters():
            if param.requires_grad:
                trainable_params[f'anomaly_consistency.{name}'] = param
        
        return trainable_params
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': frozen_params,
            'trainable_ratio': trainable_params / total_params,
            'fusion_type': self.anomaly_fusion.fusion_type,
            'context_enhancement': self.enable_context_enhancement,
            'd_model': self.d_model
        }
