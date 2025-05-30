"""
异常数据生成器模块

提供多种策略生成异常答题行为，用于训练异常检测器。
"""

import numpy as np
import torch
from typing import Tuple, List, Dict, Optional
import random


class AnomalyGenerator:
    """异常答题序列生成器"""
    
    STRATEGIES = ['consecutive', 'pattern', 'random_burst', 'difficulty_based']
    
    def __init__(self, strategies: Optional[List[str]] = None, seed: Optional[int] = None):
        """
        Args:
            strategies: 使用的异常生成策略列表
            seed: 随机种子
        """
        self.strategies = strategies or self.STRATEGIES
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
    
    def generate_anomalies(self, 
                          q: torch.Tensor, 
                          s: torch.Tensor, 
                          anomaly_ratio: float = 0.1,
                          strategy_weights: Optional[Dict[str, float]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成异常序列和标签
        
        Args:
            q: 问题ID序列 (batch_size, seq_len)
            s: 答案序列 (batch_size, seq_len)
            anomaly_ratio: 异常样本比例
            strategy_weights: 各策略的权重
        
        Returns:
            s_anomaly: 包含异常的答案序列
            anomaly_labels: 异常标签 (0=正常, 1=异常)
        """
        batch_size, seq_len = s.shape
        s_anomaly = s.clone()
        anomaly_labels = torch.zeros_like(s)
        
        # 确定要添加异常的序列数
        n_anomaly_seqs = max(1, int(batch_size * anomaly_ratio))
        anomaly_seq_indices = random.sample(range(batch_size), n_anomaly_seqs)
        
        # 默认策略权重
        if strategy_weights is None:
            strategy_weights = {strategy: 1.0 for strategy in self.strategies}
        
        # 归一化权重
        total_weight = sum(strategy_weights.values())
        strategy_probs = {k: v/total_weight for k, v in strategy_weights.items()}
        
        for idx in anomaly_seq_indices:
            # 根据权重选择策略
            strategy = np.random.choice(
                list(strategy_probs.keys()), 
                p=list(strategy_probs.values())
            )
            
            # 应用策略
            if strategy == 'consecutive':
                s_anomaly[idx], anomaly_labels[idx] = self._consecutive_flip(s[idx], q[idx])
            elif strategy == 'pattern':
                s_anomaly[idx], anomaly_labels[idx] = self._pattern_flip(s[idx], q[idx])
            elif strategy == 'random_burst':
                s_anomaly[idx], anomaly_labels[idx] = self._random_burst_flip(s[idx], q[idx])
            elif strategy == 'difficulty_based':
                s_anomaly[idx], anomaly_labels[idx] = self._difficulty_based_flip(s[idx], q[idx])
        
        return s_anomaly, anomaly_labels
    
    def _consecutive_flip(self, s: torch.Tensor, q: torch.Tensor, 
                         min_len: int = 5, max_len: int = 15) -> Tuple[torch.Tensor, torch.Tensor]:
        """连续翻转策略：模拟持续的异常行为"""
        s_flip = s.clone()
        labels = torch.zeros_like(s)
        
        valid_mask = (s >= 0)
        valid_indices = torch.where(valid_mask)[0]
        
        if len(valid_indices) > min_len:
            # 确保翻转长度合理
            actual_min_len = min(min_len, len(valid_indices) // 3)
            actual_max_len = min(max_len, len(valid_indices) // 2)
            
            if actual_max_len > actual_min_len:
                flip_len = random.randint(actual_min_len, actual_max_len)
            else:
                flip_len = actual_min_len
                
            if flip_len > 0 and flip_len < len(valid_indices):
                start_idx = random.randint(0, len(valid_indices) - flip_len)
                
                # 执行翻转
                flip_indices = valid_indices[start_idx:start_idx + flip_len]
                s_flip[flip_indices] = 1 - s_flip[flip_indices]
                labels[flip_indices] = 1
        
        return s_flip, labels
    
    def _pattern_flip(self, s: torch.Tensor, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """模式翻转策略：创建不自然的答题模式"""
        s_flip = s.clone()
        labels = torch.zeros_like(s)
        
        valid_mask = (s >= 0)
        valid_indices = torch.where(valid_mask)[0]
        
        if len(valid_indices) > 10:
            pattern_type = random.choice(['all_correct', 'all_wrong', 'alternating'])
            # 确保 pattern_len 的范围合理
            min_len = min(8, len(valid_indices) // 3)  # 至少是序列的1/3，但不超过8
            max_len = min(20, len(valid_indices) // 2)  # 最多是序列的一半，但不超过20
            
            if max_len > min_len:
                pattern_len = random.randint(min_len, max_len)
            else:
                pattern_len = min_len
                
            if pattern_len > 0 and pattern_len <= len(valid_indices):
                start_idx = random.randint(0, len(valid_indices) - pattern_len)
                pattern_indices = valid_indices[start_idx:start_idx + pattern_len]
            
                if pattern_type == 'all_correct':
                    s_flip[pattern_indices] = 1
                elif pattern_type == 'all_wrong':
                    s_flip[pattern_indices] = 0
                else:  # alternating
                    for i, idx in enumerate(pattern_indices):
                        s_flip[idx] = i % 2
                
                labels[pattern_indices] = 1
        
        return s_flip, labels
    
    def _random_burst_flip(self, s: torch.Tensor, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """随机突发翻转：模拟短时间内的随机行为"""
        s_flip = s.clone()
        labels = torch.zeros_like(s)
        
        valid_mask = (s >= 0)
        valid_indices = torch.where(valid_mask)[0]
        
        if len(valid_indices) > 5:
            # 确保burst长度合理
            min_burst = min(3, len(valid_indices) // 4)
            max_burst = min(10, len(valid_indices) // 3)
            
            if max_burst > min_burst:
                burst_len = random.randint(min_burst, max_burst)
            else:
                burst_len = min_burst
                
            if burst_len > 0 and burst_len < len(valid_indices):
                start_idx = random.randint(0, len(valid_indices) - burst_len)
                burst_indices = valid_indices[start_idx:start_idx + burst_len]
                
                for idx in burst_indices:
                    if random.random() < 0.8:  # 80%概率随机
                        s_flip[idx] = random.randint(0, 1)
                        labels[idx] = 1
        
        return s_flip, labels
    
    def _difficulty_based_flip(self, s: torch.Tensor, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """基于难度的翻转：违反正常学习规律"""
        s_flip = s.clone()
        labels = torch.zeros_like(s)
        
        valid_mask = (s >= 0)
        valid_indices = torch.where(valid_mask)[0]
        
        if len(valid_indices) > 10:
            # 估计问题难度
            unique_questions = torch.unique(q[valid_mask])
            q_difficulty = {}
            
            for q_id in unique_questions:
                q_mask = (q == q_id) & valid_mask
                if q_mask.sum() > 0:
                    q_difficulty[q_id.item()] = 1 - s[q_mask].float().mean().item()
            
            # 选择异常区间
            min_flip = min(5, len(valid_indices) // 4)
            max_flip = min(15, len(valid_indices) // 2)
            
            if max_flip > min_flip:
                flip_len = random.randint(min_flip, max_flip)
            else:
                flip_len = min_flip
                
            if flip_len > 0 and flip_len < len(valid_indices):
                start_idx = random.randint(0, len(valid_indices) - flip_len)
                flip_indices = valid_indices[start_idx:start_idx + flip_len]
                
                for idx in flip_indices:
                    q_id = q[idx].item()
                    if q_id in q_difficulty:
                        difficulty = q_difficulty[q_id]
                        if difficulty < 0.3:  # 简单题
                            s_flip[idx] = 0  # 错误
                            labels[idx] = 1
                        elif difficulty > 0.7:  # 难题
                            s_flip[idx] = 1  # 正确
                            labels[idx] = 1
        
        return s_flip, labels
    
    def get_strategy_stats(self, anomaly_labels: torch.Tensor) -> Dict[str, int]:
        """统计异常标签的分布"""
        total_anomalies = (anomaly_labels == 1).sum().item()
        total_normal = (anomaly_labels == 0).sum().item()
        total_invalid = (anomaly_labels < 0).sum().item()
        
        return {
            'total_anomalies': total_anomalies,
            'total_normal': total_normal,
            'total_invalid': total_invalid,
            'anomaly_ratio': total_anomalies / (total_anomalies + total_normal) if (total_anomalies + total_normal) > 0 else 0
        }