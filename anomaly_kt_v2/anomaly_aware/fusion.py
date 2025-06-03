"""
异常感知融合策略

实现多种融合策略，重点是异常权重调整机制，
目标是通过降低异常数据的影响来提升知识追踪性能。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class AnomalyWeightAdjuster(nn.Module):
    """
    异常权重调整器
    
    核心思想：降低异常数据对知识状态更新的影响
    预期AUC提升：+0.02-0.04
    """
    
    def __init__(self, 
                 anomaly_discount_factor: float = 0.7,
                 min_weight: float = 0.1,
                 adjustment_type: str = 'linear'):
        """
        初始化异常权重调整器
        
        Args:
            anomaly_discount_factor: 异常折扣因子，越高异常影响越小
            min_weight: 最小权重，避免完全忽略异常数据
            adjustment_type: 调整类型 ('linear', 'exponential', 'sigmoid')
        """
        super().__init__()
        self.anomaly_discount_factor = anomaly_discount_factor
        self.min_weight = min_weight
        self.adjustment_type = adjustment_type
        
        # 可学习的调整参数
        self.learnable_discount = nn.Parameter(torch.tensor(anomaly_discount_factor))
        
    def forward(self, anomaly_probs: torch.Tensor) -> torch.Tensor:
        """
        计算异常权重
        
        Args:
            anomaly_probs: 异常概率 [batch_size, seq_len]
            
        Returns:
            weights: 调整权重 [batch_size, seq_len]
        """
        if self.adjustment_type == 'linear':
            # 线性调整：weight = 1 - discount * anomaly_prob
            weights = 1.0 - self.learnable_discount * anomaly_probs
            
        elif self.adjustment_type == 'exponential':
            # 指数调整：weight = exp(-discount * anomaly_prob)
            weights = torch.exp(-self.learnable_discount * anomaly_probs)
            
        elif self.adjustment_type == 'sigmoid':
            # Sigmoid调整：更平滑的过渡
            weights = torch.sigmoid(-self.learnable_discount * (anomaly_probs - 0.5))
            
        else:
            raise ValueError(f"Unknown adjustment_type: {self.adjustment_type}")
        
        # 确保权重在合理范围内
        weights = torch.clamp(weights, min=self.min_weight, max=1.0)
        
        return weights


class AnomalyAwareFusion(nn.Module):
    """
    异常感知融合层
    
    将基线知识追踪的输出与异常检测信息融合，
    实现异常感知的知识状态表示。
    """
    
    def __init__(self,
                 d_model: int,
                 fusion_type: str = 'attention',
                 dropout: float = 0.1):
        """
        初始化融合层
        
        Args:
            d_model: 模型维度
            fusion_type: 融合类型 ('attention', 'gating', 'weighted')
            dropout: Dropout率
        """
        super().__init__()
        self.d_model = d_model
        self.fusion_type = fusion_type
        
        # 异常权重调整器
        self.weight_adjuster = AnomalyWeightAdjuster()
        
        if fusion_type == 'attention':
            # 注意力融合
            self.attention = nn.MultiheadAttention(d_model, num_heads=8, dropout=dropout)
            self.anomaly_proj = nn.Linear(1, d_model)
            
        elif fusion_type == 'gating':
            # 门控融合
            self.gate_net = nn.Sequential(
                nn.Linear(d_model + 1, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.Sigmoid()
            )
            
        elif fusion_type == 'weighted':
            # 加权融合
            self.weight_net = nn.Sequential(
                nn.Linear(1, d_model // 4),
                nn.ReLU(),
                nn.Linear(d_model // 4, 1),
                nn.Sigmoid()
            )
            
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,
                baseline_output: torch.Tensor,
                anomaly_probs: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        异常感知融合
        
        Args:
            baseline_output: 基线模型输出 [batch_size, seq_len, d_model]
            anomaly_probs: 异常概率 [batch_size, seq_len]
            mask: 序列掩码 [batch_size, seq_len]
            
        Returns:
            fused_output: 融合后的输出 [batch_size, seq_len, d_model]
            adjustment_weights: 调整权重 [batch_size, seq_len]
        """
        batch_size, seq_len, d_model = baseline_output.shape
        
        # 计算异常调整权重
        adjustment_weights = self.weight_adjuster(anomaly_probs)
        
        if self.fusion_type == 'attention':
            # 注意力融合
            anomaly_features = self.anomaly_proj(anomaly_probs.unsqueeze(-1))
            
            # 使用异常信息作为query，基线输出作为key和value
            fused_output, _ = self.attention(
                query=anomaly_features.transpose(0, 1),
                key=baseline_output.transpose(0, 1),
                value=baseline_output.transpose(0, 1),
                key_padding_mask=mask if mask is not None else None
            )
            fused_output = fused_output.transpose(0, 1)
            
        elif self.fusion_type == 'gating':
            # 门控融合
            anomaly_input = torch.cat([
                baseline_output, 
                anomaly_probs.unsqueeze(-1).expand(-1, -1, d_model)
            ], dim=-1)
            
            gate = self.gate_net(anomaly_input)
            fused_output = gate * baseline_output
            
        elif self.fusion_type == 'weighted':
            # 加权融合
            fusion_weights = self.weight_net(anomaly_probs.unsqueeze(-1))
            
            # 应用异常调整权重
            adjusted_baseline = baseline_output * adjustment_weights.unsqueeze(-1)
            fused_output = fusion_weights * baseline_output + (1 - fusion_weights) * adjusted_baseline
            
        else:
            # 默认：直接应用权重调整
            fused_output = baseline_output * adjustment_weights.unsqueeze(-1)
        
        # 残差连接和层归一化
        fused_output = self.layer_norm(fused_output + baseline_output)
        fused_output = self.dropout(fused_output)
        
        return fused_output, adjustment_weights


class ContextEnhancer(nn.Module):
    """
    异常上下文增强器
    
    利用异常检测信息增强序列的上下文表示
    预期AUC提升：+0.03-0.05
    """
    
    def __init__(self, d_model: int, context_window: int = 5):
        """
        初始化上下文增强器
        
        Args:
            d_model: 模型维度
            context_window: 上下文窗口大小
        """
        super().__init__()
        self.d_model = d_model
        self.context_window = context_window
        
        # 异常上下文编码器
        self.anomaly_encoder = nn.Sequential(
            nn.Linear(context_window, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
            nn.Tanh()
        )
        
        # 上下文融合层
        self.context_fusion = nn.MultiheadAttention(d_model, num_heads=4)
        
    def forward(self, 
                sequence: torch.Tensor,
                anomaly_probs: torch.Tensor) -> torch.Tensor:
        """
        增强序列上下文
        
        Args:
            sequence: 输入序列 [batch_size, seq_len, d_model]
            anomaly_probs: 异常概率 [batch_size, seq_len]
            
        Returns:
            enhanced_sequence: 增强后的序列 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = sequence.shape
        
        # 构建异常上下文窗口
        anomaly_context = []
        for i in range(seq_len):
            start_idx = max(0, i - self.context_window // 2)
            end_idx = min(seq_len, i + self.context_window // 2 + 1)
            
            # 填充到固定窗口大小
            window = torch.zeros(batch_size, self.context_window, device=anomaly_probs.device)
            actual_window = anomaly_probs[:, start_idx:end_idx]
            window[:, :actual_window.size(1)] = actual_window
            
            anomaly_context.append(window)
        
        anomaly_context = torch.stack(anomaly_context, dim=1)  # [batch_size, seq_len, context_window]
        
        # 编码异常上下文
        anomaly_features = self.anomaly_encoder(anomaly_context)  # [batch_size, seq_len, d_model]
        
        # 融合上下文信息
        enhanced_sequence, _ = self.context_fusion(
            query=sequence.transpose(0, 1),
            key=anomaly_features.transpose(0, 1),
            value=anomaly_features.transpose(0, 1)
        )
        
        enhanced_sequence = enhanced_sequence.transpose(0, 1)
        
        return enhanced_sequence + sequence  # 残差连接
