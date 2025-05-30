"""
异常感知的DTransformer模型

集成异常检测器，动态调整知识状态更新。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from DTransformer.model import DTransformer


class AnomalyAwareDTransformer(DTransformer):
    """异常感知的知识追踪模型"""
    
    def __init__(self, 
                 n_questions: int,
                 n_pid: int = 0,
                 d_model: int = 128,
                 d_fc: int = 256,
                 n_heads: int = 8,
                 n_know: int = 16,
                 n_layers: int = 1,
                 dropout: float = 0.05,
                 lambda_cl: float = 0.1,
                 proj: bool = False,
                 hard_neg: bool = True,
                 window: int = 1,
                 shortcut: bool = False,
                 anomaly_detector: Optional[nn.Module] = None,
                 anomaly_weight: float = 0.5):
        """
        Args:
            anomaly_detector: 预训练的异常检测器
            anomaly_weight: 异常权重 (0-1)，控制异常对知识状态的影响
        """
        super().__init__(
            n_questions, n_pid, d_model, d_fc, n_heads, n_know,
            n_layers, dropout, lambda_cl, proj, hard_neg, window, shortcut
        )
        
        self.anomaly_detector = anomaly_detector
        self.anomaly_weight = anomaly_weight
        
        # 异常调节层
        self.anomaly_gate = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.Sigmoid()
        )
        
        # 异常感知的知识更新
        self.anomaly_modulator = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
            nn.Tanh()
        )
    
    def forward_with_anomaly(self, q_emb: torch.Tensor, s_emb: torch.Tensor, 
                            lens: torch.Tensor, anomaly_scores: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """带异常感知的前向传播"""
        batch_size, seq_len, d_model = q_emb.shape
        
        # 扩展异常分数
        anomaly_expanded = anomaly_scores.unsqueeze(-1)
        
        # 调节答案嵌入
        anomaly_mod = self.anomaly_modulator(anomaly_expanded)
        s_emb_adjusted = s_emb * (1 - self.anomaly_weight * anomaly_expanded) + \
                        s_emb * anomaly_mod * self.anomaly_weight
        
        # 使用调整后的嵌入进行前向传播
        if self.shortcut:
            hq, _ = self.block1(q_emb, q_emb, q_emb, lens, peek_cur=True)
            hs, scores = self.block2(s_emb_adjusted, s_emb_adjusted, s_emb_adjusted, lens, peek_cur=True)
            h = self.block3(hq, hq, hs, lens, peek_cur=False)
            return h, scores, None
        
        # 标准DTransformer流程
        if self.n_layers == 1:
            hq = q_emb
            p, q_scores = self.block1(q_emb, q_emb, s_emb_adjusted, lens, peek_cur=True)
        elif self.n_layers == 2:
            hq = q_emb
            hs, _ = self.block1(s_emb_adjusted, s_emb_adjusted, s_emb_adjusted, lens, peek_cur=True)
            p, q_scores = self.block2(hq, hq, hs, lens, peek_cur=True)
        else:
            hq, _ = self.block1(q_emb, q_emb, q_emb, lens, peek_cur=True)
            hs, _ = self.block2(s_emb_adjusted, s_emb_adjusted, s_emb_adjusted, lens, peek_cur=True)
            p, q_scores = self.block3(hq, hq, hs, lens, peek_cur=True)
        
        # 知识参数交互
        bs, seqlen, d_model = p.size()
        n_know = self.n_know
        
        query = (self.know_params[None, :, None, :]
                .expand(bs, -1, seqlen, -1)
                .contiguous()
                .view(bs * n_know, seqlen, d_model))
        
        hq = hq.unsqueeze(1).expand(-1, n_know, -1, -1).reshape_as(query)
        p = p.unsqueeze(1).expand(-1, n_know, -1, -1).reshape_as(query)
        
        # 根据异常分数调节知识更新
        anomaly_expanded_know = anomaly_scores.unsqueeze(1).expand(-1, n_know, -1)
        anomaly_expanded_know = anomaly_expanded_know.reshape(bs * n_know, seqlen, 1)
        p_modulated = p * (1 - self.anomaly_weight * anomaly_expanded_know)
        
        z, k_scores = self.block4(
            query, hq, p_modulated, 
            torch.repeat_interleave(lens, n_know), 
            peek_cur=False
        )
        
        z = (z.view(bs, n_know, seqlen, d_model)
             .transpose(1, 2)
             .contiguous()
             .view(bs, seqlen, -1))
        
        k_scores = (k_scores.view(bs, n_know, self.n_heads, seqlen, seqlen)
                   .permute(0, 2, 3, 1, 4)
                   .contiguous())
        
        return z, q_scores, k_scores
    
    def predict_with_anomaly(self, q: torch.Tensor, s: torch.Tensor, 
                           pid: Optional[torch.Tensor] = None, n: int = 1):
        """带异常检测的预测"""
        # 获取异常分数
        if self.anomaly_detector is not None:
            with torch.no_grad():
                self.anomaly_detector.eval()
                anomaly_scores = self.anomaly_detector(q, s, pid)
        else:
            anomaly_scores = torch.zeros_like(s).float()
        
        # 嵌入
        q_emb, s_emb, lens, p_diff = self.embedding(q, s, pid)
        
        # 异常感知的前向传播
        z, q_scores, k_scores = self.forward_with_anomaly(
            q_emb, s_emb, lens, anomaly_scores
        )
        
        # 预测
        if self.shortcut:
            assert n == 1, "AKT does not support T+N prediction"
            h = z
            query = q_emb
        else:
            query = q_emb[:, n - 1:, :]
            h = self.readout(z[:, :query.size(1), :], query)
        
        # 输出预测
        combined = torch.cat([query, h], dim=-1)
        y = self.out(combined).squeeze(-1)
        
        # 对异常时刻降低预测置信度
        if n == 1:
            anomaly_mask = anomaly_scores[:, n-1:] > 0.5
            y = torch.where(
                anomaly_mask,
                y * (1 - self.anomaly_weight * 0.5),  # 降低极端预测
                y
            )
        
        reg_loss = (p_diff**2).mean() * 1e-3 if pid is not None else 0.0
        
        return y, z, q_emb, reg_loss, (q_scores, k_scores), anomaly_scores
    
    def get_loss(self, q: torch.Tensor, s: torch.Tensor, 
                 pid: Optional[torch.Tensor] = None):
        """计算损失（考虑异常）"""
        logits, _, _, reg_loss, _, anomaly_scores = self.predict_with_anomaly(q, s, pid)
        
        masked_labels = s[s >= 0].float()
        masked_logits = logits[s >= 0]
        masked_anomaly = anomaly_scores[s >= 0]
        
        # 降低异常样本的权重
        weights = 1.0 - masked_anomaly * self.anomaly_weight
        
        # 加权BCE损失
        loss = F.binary_cross_entropy_with_logits(
            masked_logits, masked_labels, reduction='none'
        )
        weighted_loss = (loss * weights).mean()
        
        return weighted_loss + reg_loss
    
    def get_cl_loss(self, q: torch.Tensor, s: torch.Tensor, 
                    pid: Optional[torch.Tensor] = None):
        """计算对比学习损失（考虑异常）"""
        # 获取异常分数
        if self.anomaly_detector is not None:
            with torch.no_grad():
                self.anomaly_detector.eval()
                anomaly_scores = self.anomaly_detector(q, s, pid)
        else:
            anomaly_scores = torch.zeros_like(s).float()
        
        # 过滤掉异常样本进行对比学习
        mask = (s >= 0) & (anomaly_scores < 0.5)
        
        # 如果正常样本太少，使用标准损失
        if mask.sum() < s.size(0) * s.size(1) * 0.5:
            return self.get_loss(q, s, pid), None, None
        
        # 调用父类的对比学习方法
        return super().get_cl_loss(q, s, pid)