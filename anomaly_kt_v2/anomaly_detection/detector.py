"""
因果异常检测器模块

实现严格遵守时序因果性的异常检测模型。
完全基于DTransformer原始代码，不依赖任何anomaly_kt模块。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import random


class CausalAnomalyDetector(nn.Module):
    """因果异常检测器 - 检测答题序列中的异常行为"""

    def __init__(self,
                 n_questions: int,
                 n_pid: int = 0,
                 d_model: int = 128,
                 n_heads: int = 8,
                 n_layers: int = 2,
                 dropout: float = 0.1,
                 window_size: int = 10):
        """
        Args:
            n_questions: 问题总数
            n_pid: 问题部分ID总数
            d_model: 模型隐藏层维度
            n_heads: 注意力头数
            n_layers: Transformer层数
            dropout: Dropout率
            window_size: 统计特征窗口大小
        """
        super().__init__()

        self.n_questions = n_questions
        self.window_size = window_size
        self.d_model = d_model

        # 嵌入层
        self.q_embed = nn.Embedding(n_questions + 1, d_model)
        self.s_embed = nn.Embedding(2, d_model)
        self.position_embed = nn.Embedding(512, d_model)

        if n_pid > 0:
            self.pid_embed = nn.Embedding(n_pid + 1, d_model // 2)
        else:
            self.pid_embed = None

        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Transformer编码器
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # 统计特征提取
        self.stat_features = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, d_model // 4)
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model + d_model // 4, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, s: torch.Tensor,
                pid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""
        batch_size, seq_len = q.shape
        device = q.device

        # 处理mask
        mask = (s >= 0)
        q_masked = q.masked_fill(~mask, 0)
        s_masked = s.masked_fill(~mask, 0)

        # 嵌入
        q_emb = self.q_embed(q_masked)
        s_emb = self.s_embed(s_masked)

        # 位置编码
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embed(positions)

        # 融合特征
        combined = torch.cat([q_emb, s_emb], dim=-1)
        h = self.feature_fusion(combined) + pos_emb

        if self.pid_embed is not None and pid is not None:
            pid_masked = pid.masked_fill(~mask, 0)
            pid_emb = self.pid_embed(pid_masked)
            h[:, :, :self.d_model//2] += pid_emb

        h = self.dropout(h)

        # 创建因果掩码
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()

        # Transformer编码
        for layer in self.transformer_layers:
            h = layer(h, causal_mask, mask)

        # 提取统计特征
        stat_feat = self._extract_statistics(s_masked, mask)
        stat_emb = self.stat_features(stat_feat)

        # 组合特征并分类
        combined = torch.cat([h, stat_emb], dim=-1)
        logits = self.classifier(combined).squeeze(-1)

        # 应用mask
        logits = logits.masked_fill(~mask, -1e9)

        # 直接返回logits，让损失函数处理sigmoid
        return logits

    def _extract_statistics(self, s: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """提取统计特征（只基于历史）"""
        batch_size, seq_len = s.shape
        device = s.device
        features = []

        for t in range(seq_len):
            # 历史窗口
            start = max(0, t - self.window_size + 1)
            end = t + 1

            window_s = s[:, start:end]
            window_mask = mask[:, start:end]

            # 计算特征
            feat = self._compute_window_features(window_s, window_mask, t, seq_len)
            features.append(feat)

        return torch.stack(features, dim=1)

    def _compute_window_features(self, window_s: torch.Tensor, window_mask: torch.Tensor,
                                t: int, seq_len: int) -> torch.Tensor:
        """计算窗口统计特征"""
        batch_size = window_s.shape[0]
        device = window_s.device

        # 有效数量
        valid_count = window_mask.sum(dim=1, keepdim=True).float()

        # 1. 正确率
        correct_rate = (window_s * window_mask).sum(dim=1, keepdim=True).float() / (valid_count + 1e-6)

        # 2. 最近5个的正确率
        recent_k = min(5, window_s.shape[1])
        if recent_k > 0:
            recent_s = window_s[:, -recent_k:]
            recent_mask = window_mask[:, -recent_k:]
            recent_count = recent_mask.sum(dim=1, keepdim=True).float()
            recent_correct = (recent_s * recent_mask).sum(dim=1, keepdim=True).float() / (recent_count + 1e-6)
        else:
            recent_correct = torch.zeros(batch_size, 1, device=device)

        # 3. 变化率
        change_count = torch.zeros(batch_size, 1, device=device)
        if window_s.shape[1] > 1:
            for i in range(1, window_s.shape[1]):
                valid_trans = window_mask[:, i] * window_mask[:, i-1]
                changes = (window_s[:, i] != window_s[:, i-1]) * valid_trans
                change_count += changes.unsqueeze(1).float()
        change_rate = change_count / (valid_count + 1e-6)

        # 4. 最大连续长度
        max_consecutive = self._compute_max_consecutive(window_s, window_mask)

        # 5. 相对位置
        relative_position = torch.full((batch_size, 1), t / max(seq_len, 100), device=device)

        # 6. 窗口覆盖率
        window_coverage = valid_count / self.window_size

        return torch.cat([
            correct_rate,
            recent_correct,
            change_rate,
            max_consecutive / self.window_size,
            relative_position,
            window_coverage
        ], dim=1)

    def _compute_max_consecutive(self, window_s: torch.Tensor,
                                window_mask: torch.Tensor) -> torch.Tensor:
        """计算最大连续相同答案长度"""
        batch_size = window_s.shape[0]
        device = window_s.device
        max_consecutive = torch.zeros(batch_size, 1, device=device)

        for b in range(batch_size):
            if window_mask[b].sum() == 0:
                continue

            max_len = 0
            current_len = 0
            prev_val = -1

            for i in range(window_s.shape[1]):
                if window_mask[b, i]:
                    if window_s[b, i] == prev_val:
                        current_len += 1
                    else:
                        max_len = max(max_len, current_len)
                        current_len = 1
                        prev_val = window_s[b, i].item()

            max_consecutive[b] = max(max_len, current_len)

        return max_consecutive

    def get_loss(self, q: torch.Tensor, s: torch.Tensor,
                 labels: torch.Tensor, pid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算损失函数"""
        logits = self(q, s, pid)

        mask = (s >= 0)
        valid_logits = logits[mask]
        valid_labels = labels[mask].float()

        # 计算类别权重
        pos_count = valid_labels.sum()
        neg_count = len(valid_labels) - pos_count

        # 使用类别权重
        if pos_count > 0:
            pos_weight = neg_count / pos_count
            pos_weight = torch.clamp(pos_weight, min=1.0, max=20.0)
        else:
            pos_weight = torch.tensor(10.0)

        pos_weight = pos_weight.to(logits.device)

        # 使用 BCEWithLogitsLoss
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = criterion(valid_logits, valid_labels)

        return loss


class TransformerLayer(nn.Module):
    """因果Transformer层"""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()

        # 多头注意力
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor,
                padding_mask: torch.Tensor) -> torch.Tensor:
        # 准备注意力掩码
        attn_mask = ~causal_mask  # MultiheadAttention需要True表示被mask的位置

        # 自注意力
        attn_out, _ = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=~padding_mask
        )
        x = self.norm1(x + self.dropout(attn_out))

        # 前馈
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x
