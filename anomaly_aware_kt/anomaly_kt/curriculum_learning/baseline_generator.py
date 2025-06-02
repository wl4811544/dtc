"""
基线异常生成器

用于技术验证和性能基准的基线异常生成器。
包含随机翻转、均匀随机等技术验证用的异常策略。

基于我们的分析：
- 随机翻转异常属于统计异常，教育价值有限但技术价值存在
- 主要用于验证检测器基本功能和建立性能下界基准
"""

import torch
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


class BaselineAnomalyGenerator:
    """基线异常生成器 - 用于技术验证和性能基准"""
    
    BASELINE_STRATEGIES = [
        'random_flip',          # 随机翻转
        'uniform_random',       # 均匀随机
        'gaussian_noise',       # 高斯噪声
        'systematic_bias',      # 系统性偏差
        'label_corruption'      # 标签损坏
    ]
    
    def __init__(self, purpose='technical_validation'):
        """
        初始化基线异常生成器
        
        Args:
            purpose: 使用目的 ('technical_validation', 'performance_baseline')
        """
        self.purpose = purpose
        self.strategies = self.BASELINE_STRATEGIES
        
    def generate_baseline_anomalies(self, q: torch.Tensor, s: torch.Tensor, 
                                   strategy: str = 'random_flip',
                                   anomaly_ratio: float = 0.1,
                                   **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成基线异常用于技术验证
        
        Args:
            q: 问题序列 [batch_size, seq_len]
            s: 答案序列 [batch_size, seq_len] 
            strategy: 异常生成策略
            anomaly_ratio: 异常序列的比例
            **kwargs: 策略特定参数
            
        Returns:
            s_anomaly: 包含异常的答案序列
            anomaly_labels: 异常标签 [batch_size, seq_len]
        """
        if strategy == 'random_flip':
            return self._random_flip_anomaly(q, s, anomaly_ratio, **kwargs)
        elif strategy == 'uniform_random':
            return self._uniform_random_anomaly(q, s, anomaly_ratio, **kwargs)
        elif strategy == 'gaussian_noise':
            return self._gaussian_noise_anomaly(q, s, anomaly_ratio, **kwargs)
        elif strategy == 'systematic_bias':
            return self._systematic_bias_anomaly(q, s, anomaly_ratio, **kwargs)
        elif strategy == 'label_corruption':
            return self._label_corruption_anomaly(q, s, anomaly_ratio, **kwargs)
        else:
            raise ValueError(f"未知的基线策略: {strategy}")
    
    def _random_flip_anomaly(self, q: torch.Tensor, s: torch.Tensor,
                           anomaly_ratio: float = 0.1,
                           flip_probability: float = 0.5,
                           context_aware: bool = False,
                           min_anomaly_length: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        随机翻转异常生成
        
        基于我们的分析：
        - 属于统计异常，高熵、零自相关
        - 检测难度低，主要用于验证基本功能
        - 可以添加上下文感知提升真实性
        
        Args:
            flip_probability: 每个位置翻转的概率
            context_aware: 是否考虑上下文(题目难度等)
            min_anomaly_length: 最小异常长度
        """
        batch_size, seq_len = s.shape
        s_anomaly = s.clone()
        anomaly_labels = torch.zeros_like(s, dtype=torch.float)
        
        # 选择要添加异常的序列
        n_anomaly_seqs = max(1, int(batch_size * anomaly_ratio))
        anomaly_seq_indices = random.sample(range(batch_size), n_anomaly_seqs)
        
        for idx in anomaly_seq_indices:
            # 找到有效位置（非padding）
            valid_mask = (s[idx] >= 0)
            valid_indices = torch.where(valid_mask)[0]
            
            if len(valid_indices) < min_anomaly_length:
                continue
            
            # 随机选择连续的异常区间
            max_start = len(valid_indices) - min_anomaly_length
            if max_start <= 0:
                continue
                
            start_idx = random.randint(0, max_start)
            anomaly_length = min(
                random.randint(min_anomaly_length, min(len(valid_indices) - start_idx, 10)),
                int(len(valid_indices) * flip_probability)
            )
            
            anomaly_positions = valid_indices[start_idx:start_idx + anomaly_length]
            
            for pos in anomaly_positions:
                if context_aware:
                    # 基于上下文的智能随机（考虑题目难度）
                    s_anomaly[idx, pos] = self._context_aware_flip(
                        s[idx, pos], q[idx, pos] if q is not None else None
                    )
                else:
                    # 纯随机翻转
                    s_anomaly[idx, pos] = 1 - s[idx, pos]  # 0->1, 1->0
                
                anomaly_labels[idx, pos] = 1
        
        return s_anomaly, anomaly_labels
    
    def _context_aware_flip(self, original_answer: torch.Tensor, 
                          question_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """基于上下文的智能随机翻转"""
        # 简单的上下文感知：基于问题ID的伪随机
        if question_id is not None:
            # 使用问题ID作为随机种子，增加一些确定性
            random.seed(int(question_id.item()) % 1000)
            flip_prob = random.uniform(0.3, 0.8)  # 不完全随机
            random.seed()  # 重置随机种子
            
            if random.random() < flip_prob:
                return 1 - original_answer
        
        return 1 - original_answer  # 默认翻转
    
    def _uniform_random_anomaly(self, q: torch.Tensor, s: torch.Tensor,
                               anomaly_ratio: float = 0.1,
                               **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """均匀随机异常：完全随机的0/1序列"""
        batch_size, seq_len = s.shape
        s_anomaly = s.clone()
        anomaly_labels = torch.zeros_like(s, dtype=torch.float)
        
        n_anomaly_seqs = max(1, int(batch_size * anomaly_ratio))
        anomaly_seq_indices = random.sample(range(batch_size), n_anomaly_seqs)
        
        for idx in anomaly_seq_indices:
            valid_mask = (s[idx] >= 0)
            valid_indices = torch.where(valid_mask)[0]
            
            # 随机选择一段区间进行均匀随机
            if len(valid_indices) > 3:
                start = random.randint(0, len(valid_indices) - 3)
                end = min(start + random.randint(3, 8), len(valid_indices))
                
                for pos in valid_indices[start:end]:
                    s_anomaly[idx, pos] = random.randint(0, 1)
                    anomaly_labels[idx, pos] = 1
        
        return s_anomaly, anomaly_labels
    
    def _gaussian_noise_anomaly(self, q: torch.Tensor, s: torch.Tensor,
                               anomaly_ratio: float = 0.1,
                               noise_std: float = 0.3,
                               **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """高斯噪声异常：在原答案基础上添加高斯噪声"""
        batch_size, seq_len = s.shape
        s_anomaly = s.clone().float()
        anomaly_labels = torch.zeros_like(s, dtype=torch.float)
        
        n_anomaly_seqs = max(1, int(batch_size * anomaly_ratio))
        anomaly_seq_indices = random.sample(range(batch_size), n_anomaly_seqs)
        
        for idx in anomaly_seq_indices:
            valid_mask = (s[idx] >= 0)
            
            # 添加高斯噪声
            noise = torch.normal(0, noise_std, size=s[idx].shape)
            s_anomaly[idx] = s[idx].float() + noise
            
            # 限制在[0,1]范围内并四舍五入到0/1
            s_anomaly[idx] = torch.clamp(s_anomaly[idx], 0, 1)
            s_anomaly[idx] = torch.round(s_anomaly[idx])
            
            # 标记发生变化的位置为异常
            changed_mask = (s_anomaly[idx] != s[idx].float()) & valid_mask
            anomaly_labels[idx] = changed_mask.float()
        
        return s_anomaly.long(), anomaly_labels
    
    def _systematic_bias_anomaly(self, q: torch.Tensor, s: torch.Tensor,
                                anomaly_ratio: float = 0.1,
                                bias_probability: float = 0.8,
                                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """系统性偏差异常：倾向于选择某个答案"""
        batch_size, seq_len = s.shape
        s_anomaly = s.clone()
        anomaly_labels = torch.zeros_like(s, dtype=torch.float)
        
        n_anomaly_seqs = max(1, int(batch_size * anomaly_ratio))
        anomaly_seq_indices = random.sample(range(batch_size), n_anomaly_seqs)
        
        for idx in anomaly_seq_indices:
            valid_mask = (s[idx] >= 0)
            valid_indices = torch.where(valid_mask)[0]
            
            # 随机选择偏好答案（0或1）
            preferred_answer = random.randint(0, 1)
            
            # 在一段区间内系统性地偏向某个答案
            if len(valid_indices) > 5:
                start = random.randint(0, len(valid_indices) - 5)
                end = min(start + random.randint(5, 10), len(valid_indices))
                
                for pos in valid_indices[start:end]:
                    if random.random() < bias_probability:
                        s_anomaly[idx, pos] = preferred_answer
                        if s_anomaly[idx, pos] != s[idx, pos]:
                            anomaly_labels[idx, pos] = 1
        
        return s_anomaly, anomaly_labels
    
    def _label_corruption_anomaly(self, q: torch.Tensor, s: torch.Tensor,
                                 anomaly_ratio: float = 0.1,
                                 corruption_rate: float = 0.3,
                                 **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """标签损坏异常：随机损坏部分标签"""
        batch_size, seq_len = s.shape
        s_anomaly = s.clone()
        anomaly_labels = torch.zeros_like(s, dtype=torch.float)
        
        n_anomaly_seqs = max(1, int(batch_size * anomaly_ratio))
        anomaly_seq_indices = random.sample(range(batch_size), n_anomaly_seqs)
        
        for idx in anomaly_seq_indices:
            valid_mask = (s[idx] >= 0)
            valid_indices = torch.where(valid_mask)[0]
            
            # 随机选择要损坏的位置
            n_corrupt = max(1, int(len(valid_indices) * corruption_rate))
            corrupt_positions = random.sample(valid_indices.tolist(), 
                                            min(n_corrupt, len(valid_indices)))
            
            for pos in corrupt_positions:
                s_anomaly[idx, pos] = random.randint(0, 1)
                if s_anomaly[idx, pos] != s[idx, pos]:
                    anomaly_labels[idx, pos] = 1
        
        return s_anomaly, anomaly_labels
    
    def estimate_difficulty(self, strategy: str, **kwargs) -> float:
        """
        评估基线异常的检测难度
        
        基于我们的分析：
        - random_flip: 检测难度低 (0.1-0.3)
        - uniform_random: 检测难度低 (0.1-0.2)  
        - gaussian_noise: 检测难度中等 (0.3-0.5)
        - systematic_bias: 检测难度中等 (0.4-0.6)
        - label_corruption: 检测难度中等 (0.3-0.5)
        """
        difficulty_map = {
            'random_flip': 0.2,
            'uniform_random': 0.15,
            'gaussian_noise': 0.4,
            'systematic_bias': 0.5,
            'label_corruption': 0.4
        }
        
        return difficulty_map.get(strategy, 0.3)
    
    def get_strategy_info(self, strategy: str) -> Dict:
        """获取策略详细信息"""
        info_map = {
            'random_flip': {
                'description': '随机翻转异常',
                'type': '统计异常',
                'educational_value': '低',
                'technical_value': '中',
                'detection_difficulty': '低',
                'use_case': '基本功能验证'
            },
            'uniform_random': {
                'description': '均匀随机异常',
                'type': '统计异常',
                'educational_value': '极低',
                'technical_value': '低',
                'detection_difficulty': '极低',
                'use_case': '最简单基准'
            },
            'gaussian_noise': {
                'description': '高斯噪声异常',
                'type': '噪声异常',
                'educational_value': '低',
                'technical_value': '中',
                'detection_difficulty': '中',
                'use_case': '鲁棒性测试'
            },
            'systematic_bias': {
                'description': '系统性偏差异常',
                'type': '偏差异常',
                'educational_value': '中',
                'technical_value': '中',
                'detection_difficulty': '中',
                'use_case': '偏差检测验证'
            },
            'label_corruption': {
                'description': '标签损坏异常',
                'type': '噪声异常',
                'educational_value': '低',
                'technical_value': '中',
                'detection_difficulty': '中',
                'use_case': '噪声鲁棒性测试'
            }
        }
        
        return info_map.get(strategy, {})
