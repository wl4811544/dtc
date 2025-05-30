"""
增强的异常数据生成器

解决原生成器的问题：
1. 异常密度不够
2. 策略权重不合理
3. 缺乏渐进式难度
"""

import numpy as np
import torch
from typing import Tuple, List, Dict, Optional
import random
from .generator import AnomalyGenerator


class EnhancedAnomalyGenerator(AnomalyGenerator):
    """增强的异常生成器"""

    def __init__(self, strategies: Optional[List[str]] = None, seed: Optional[int] = None):
        super().__init__(strategies, seed)

        # 改进的策略权重（基于检测难度）
        self.default_strategy_weights = {
            'consecutive': 0.4,      # 最容易检测，权重高
            'difficulty_based': 0.3, # 中等难度
            'pattern': 0.2,          # 较难检测
            'random_burst': 0.1      # 最难检测，权重低
        }

    def generate_anomalies_enhanced(self,
                                  q: torch.Tensor,
                                  s: torch.Tensor,
                                  anomaly_ratio: float = 0.1,
                                  strategy_weights: Optional[Dict[str, float]] = None,
                                  min_anomaly_density: float = 0.3,
                                  progressive_difficulty: bool = True,
                                  epoch: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        增强的异常生成方法

        Args:
            q: 问题ID序列
            s: 答案序列
            anomaly_ratio: 异常序列比例
            strategy_weights: 策略权重
            min_anomaly_density: 每个异常序列的最小异常密度
            progressive_difficulty: 是否使用渐进式难度
            epoch: 当前训练轮次
        """
        batch_size, seq_len = s.shape
        s_anomaly = s.clone()
        anomaly_labels = torch.zeros_like(s)

        # 使用改进的策略权重
        if strategy_weights is None:
            strategy_weights = self.default_strategy_weights.copy()

            # 渐进式难度调整
            if progressive_difficulty and epoch > 0:
                strategy_weights = self._adjust_weights_by_epoch(strategy_weights, epoch)

        # 确定异常序列数量（至少保证一定数量）
        min_anomaly_seqs = max(2, int(batch_size * 0.3))  # 至少30%
        target_anomaly_seqs = max(min_anomaly_seqs, int(batch_size * anomaly_ratio))
        target_anomaly_seqs = min(target_anomaly_seqs, batch_size)  # 不能超过批次大小
        anomaly_seq_indices = random.sample(range(batch_size), target_anomaly_seqs)

        # 归一化权重
        total_weight = sum(strategy_weights.values())
        strategy_probs = {k: v/total_weight for k, v in strategy_weights.items()}

        for idx in anomaly_seq_indices:
            # 选择策略
            strategy = np.random.choice(
                list(strategy_probs.keys()),
                p=list(strategy_probs.values())
            )

            # 生成异常，确保最小密度
            s_anomaly[idx], anomaly_labels[idx] = self._generate_dense_anomaly(
                s[idx], q[idx], strategy, min_anomaly_density
            )

        return s_anomaly, anomaly_labels

    def _adjust_weights_by_epoch(self, weights: Dict[str, float], epoch: int) -> Dict[str, float]:
        """根据训练轮次调整策略权重"""
        adjusted_weights = weights.copy()

        if epoch < 5:
            # 早期：更多简单异常
            adjusted_weights['consecutive'] = 0.6
            adjusted_weights['difficulty_based'] = 0.25
            adjusted_weights['pattern'] = 0.1
            adjusted_weights['random_burst'] = 0.05
        elif epoch < 15:
            # 中期：平衡
            adjusted_weights['consecutive'] = 0.4
            adjusted_weights['difficulty_based'] = 0.3
            adjusted_weights['pattern'] = 0.2
            adjusted_weights['random_burst'] = 0.1
        else:
            # 后期：更多复杂异常
            adjusted_weights['consecutive'] = 0.3
            adjusted_weights['difficulty_based'] = 0.3
            adjusted_weights['pattern'] = 0.25
            adjusted_weights['random_burst'] = 0.15

        return adjusted_weights

    def _generate_dense_anomaly(self, s: torch.Tensor, q: torch.Tensor,
                               strategy: str, min_density: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成密度足够的异常"""
        # 先用原方法生成
        if strategy == 'consecutive':
            s_anomaly, labels = self._consecutive_flip_enhanced(s, q, min_density)
        elif strategy == 'pattern':
            s_anomaly, labels = self._pattern_flip_enhanced(s, q, min_density)
        elif strategy == 'random_burst':
            s_anomaly, labels = self._random_burst_flip_enhanced(s, q, min_density)
        elif strategy == 'difficulty_based':
            s_anomaly, labels = self._difficulty_based_flip_enhanced(s, q, min_density)
        else:
            s_anomaly, labels = self._consecutive_flip_enhanced(s, q, min_density)

        # 检查并补充异常密度
        current_density = labels.float().mean().item()
        if current_density < min_density:
            s_anomaly, labels = self._supplement_anomalies(s_anomaly, labels, s, min_density)

        return s_anomaly, labels

    def _consecutive_flip_enhanced(self, s: torch.Tensor, q: torch.Tensor,
                                  min_density: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """增强的连续翻转"""
        s_flip = s.clone()
        labels = torch.zeros_like(s)

        valid_mask = (s >= 0)
        valid_indices = torch.where(valid_mask)[0]

        if len(valid_indices) > 3:
            # 根据最小密度确定异常长度
            target_anomalies = max(5, int(len(valid_indices) * min_density))  # 至少5个异常

            # 可能生成多个连续段
            remaining_anomalies = target_anomalies
            attempts = 0

            while remaining_anomalies > 1 and attempts < 5:  # 增加尝试次数
                segment_len = min(remaining_anomalies, random.randint(2, min(8, remaining_anomalies + 1)))

                if segment_len <= len(valid_indices):
                    max_start = len(valid_indices) - segment_len
                    if max_start >= 0:
                        start_idx = random.randint(0, max_start)
                        segment_indices = valid_indices[start_idx:start_idx + segment_len]

                        # 翻转这一段
                        s_flip[segment_indices] = 1 - s_flip[segment_indices]
                        labels[segment_indices] = 1

                        remaining_anomalies -= segment_len

                attempts += 1

        return s_flip, labels

    def _pattern_flip_enhanced(self, s: torch.Tensor, q: torch.Tensor,
                              min_density: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """增强的模式翻转"""
        s_flip = s.clone()
        labels = torch.zeros_like(s)

        valid_mask = (s >= 0)
        valid_indices = torch.where(valid_mask)[0]

        if len(valid_indices) > 5:
            target_anomalies = max(3, int(len(valid_indices) * min_density))
            pattern_type = random.choice(['all_correct', 'all_wrong', 'alternating', 'mixed'])

            pattern_len = min(target_anomalies, len(valid_indices))
            start_idx = random.randint(0, len(valid_indices) - pattern_len)
            pattern_indices = valid_indices[start_idx:start_idx + pattern_len]

            if pattern_type == 'all_correct':
                s_flip[pattern_indices] = 1
            elif pattern_type == 'all_wrong':
                s_flip[pattern_indices] = 0
            elif pattern_type == 'alternating':
                for i, idx in enumerate(pattern_indices):
                    s_flip[idx] = i % 2
            else:  # mixed - 更复杂的模式
                for i, idx in enumerate(pattern_indices):
                    if i % 3 == 0:
                        s_flip[idx] = 1
                    elif i % 3 == 1:
                        s_flip[idx] = 0
                    # else keep original

            labels[pattern_indices] = 1

        return s_flip, labels

    def _random_burst_flip_enhanced(self, s: torch.Tensor, q: torch.Tensor,
                                   min_density: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """增强的随机突发"""
        s_flip = s.clone()
        labels = torch.zeros_like(s)

        valid_mask = (s >= 0)
        valid_indices = torch.where(valid_mask)[0]

        if len(valid_indices) > 3:
            target_anomalies = max(2, int(len(valid_indices) * min_density))

            # 生成多个小的随机突发
            remaining = target_anomalies
            while remaining > 0:
                burst_len = min(remaining, random.randint(1, min(5, remaining + 1)))

                if burst_len <= len(valid_indices):
                    start_idx = random.randint(0, len(valid_indices) - burst_len)
                    burst_indices = valid_indices[start_idx:start_idx + burst_len]

                    for idx in burst_indices:
                        if random.random() < 0.9:  # 90%概率随机化
                            s_flip[idx] = random.randint(0, 1)
                            labels[idx] = 1
                            remaining -= 1

                if remaining <= 0:
                    break

        return s_flip, labels

    def _difficulty_based_flip_enhanced(self, s: torch.Tensor, q: torch.Tensor,
                                       min_density: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """增强的基于难度的翻转"""
        s_flip = s.clone()
        labels = torch.zeros_like(s)

        valid_mask = (s >= 0)
        valid_indices = torch.where(valid_mask)[0]

        if len(valid_indices) > 5:
            target_anomalies = max(3, int(len(valid_indices) * min_density))

            # 简化的难度估计：基于答案分布
            correct_ratio = s[valid_mask].float().mean().item()

            # 创建违反学习规律的模式
            anomaly_count = 0
            for i, idx in enumerate(valid_indices):
                if anomaly_count >= target_anomalies:
                    break

                # 在"应该对"的地方错，在"应该错"的地方对
                if random.random() < min_density * 2:  # 增加概率
                    if correct_ratio > 0.7:  # 如果整体正确率高
                        s_flip[idx] = 0  # 强制错误
                    else:  # 如果整体正确率低
                        s_flip[idx] = 1  # 强制正确

                    labels[idx] = 1
                    anomaly_count += 1

        return s_flip, labels

    def _supplement_anomalies(self, s_anomaly: torch.Tensor, labels: torch.Tensor,
                             s_original: torch.Tensor, min_density: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """补充异常以达到最小密度"""
        valid_mask = (s_original >= 0)
        valid_indices = torch.where(valid_mask)[0]
        current_anomalies = labels.sum().item()
        target_anomalies = max(3, int(len(valid_indices) * min_density))  # 至少3个异常

        if current_anomalies < target_anomalies:
            need_more = target_anomalies - current_anomalies
            available_indices = valid_indices[labels[valid_indices] == 0]

            if len(available_indices) > 0:
                supplement_count = min(need_more, len(available_indices))
                if supplement_count > 0:
                    supplement_indices = np.random.choice(
                        available_indices.cpu().numpy(),
                        supplement_count,
                        replace=False
                    )

                    for idx in supplement_indices:
                        s_anomaly[idx] = 1 - s_anomaly[idx]  # 简单翻转
                        labels[idx] = 1

        return s_anomaly, labels
