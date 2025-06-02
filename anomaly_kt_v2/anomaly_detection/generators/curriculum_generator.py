"""
课程学习异常生成器

实现基于课程学习的异常生成策略，支持4级难度体系。
完全基于DTransformer原始代码，不依赖任何anomaly_kt模块。
"""

import torch
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from .baseline_generator import BaselineAnomalyGenerator


class CurriculumAnomalyGenerator:
    """课程学习异常生成器"""
    
    # 4级难度体系
    DIFFICULTY_LEVELS = [1, 2, 3, 4]
    
    # 各级别对应的策略
    LEVEL_STRATEGIES = {
        1: ['obvious_consecutive', 'simple_random'],           # 简单：明显异常
        2: ['pattern_anomaly', 'burst_errors'],               # 中等：模式异常
        3: ['ability_mismatch', 'subtle_temporal'],           # 困难：微妙异常
        4: ['advanced_masking', 'intelligent_deception']      # 极难：高级异常
    }
    
    def __init__(self, dataset_name: str = 'assist17'):
        """
        初始化课程异常生成器
        
        Args:
            dataset_name: 数据集名称，用于自适应配置
        """
        self.dataset_name = dataset_name
        self.baseline_generator = BaselineAnomalyGenerator()
        
        # 数据集特定配置
        self.dataset_config = self._get_dataset_config()
        
    def _get_dataset_config(self) -> Dict:
        """获取数据集特定配置"""
        configs = {
            'assist09': {
                'avg_seq_length': 50,
                'difficulty_distribution': [0.4, 0.3, 0.2, 0.1],
                'anomaly_patterns': ['consecutive', 'burst', 'gradual']
            },
            'assist17': {
                'avg_seq_length': 100,
                'difficulty_distribution': [0.3, 0.3, 0.3, 0.1],
                'anomaly_patterns': ['consecutive', 'pattern', 'temporal', 'complex']
            }
        }
        return configs.get(self.dataset_name, configs['assist17'])
    
    def generate_curriculum_anomalies(self, q: torch.Tensor, s: torch.Tensor,
                                    difficulty_levels: List[int] = [1, 2],
                                    level_weights: Dict[int, float] = None,
                                    anomaly_ratio: float = 0.1,
                                    include_baseline: bool = True,
                                    baseline_ratio: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        生成课程学习异常
        
        Args:
            q: 问题序列 [batch_size, seq_len]
            s: 答案序列 [batch_size, seq_len]
            difficulty_levels: 要生成的难度级别列表
            level_weights: 各级别权重分配
            anomaly_ratio: 异常序列比例
            include_baseline: 是否包含基线异常
            baseline_ratio: 基线异常占比
            
        Returns:
            s_anomaly: 包含异常的答案序列
            anomaly_labels: 异常标签 [batch_size, seq_len]
            difficulty_scores: 难度分数 [batch_size, seq_len]
        """
        batch_size, seq_len = s.shape
        s_anomaly = s.clone()
        anomaly_labels = torch.zeros_like(s, dtype=torch.float)
        difficulty_scores = torch.zeros_like(s, dtype=torch.float)
        
        # 计算各级别的异常数量
        total_anomaly_seqs = max(1, int(batch_size * anomaly_ratio))
        
        if include_baseline:
            # 分配基线异常和教育异常
            baseline_seqs = max(1, int(total_anomaly_seqs * baseline_ratio))
            educational_seqs = total_anomaly_seqs - baseline_seqs
        else:
            baseline_seqs = 0
            educational_seqs = total_anomaly_seqs
        
        # 默认权重
        if level_weights is None:
            level_weights = {level: 1.0/len(difficulty_levels) for level in difficulty_levels}
        
        # 生成基线异常
        if baseline_seqs > 0:
            baseline_indices = random.sample(range(batch_size), baseline_seqs)
            for idx in baseline_indices:
                s_baseline, labels_baseline = self.baseline_generator.generate_baseline_anomalies(
                    q[idx:idx+1], s[idx:idx+1], strategy='random_flip', anomaly_ratio=1.0
                )
                s_anomaly[idx] = s_baseline[0]
                anomaly_labels[idx] = labels_baseline[0]
                difficulty_scores[idx] = labels_baseline[0] * 0.1  # 基线难度低
        
        # 生成教育异常
        if educational_seqs > 0:
            remaining_indices = [i for i in range(batch_size) if i not in (baseline_indices if baseline_seqs > 0 else [])]
            educational_indices = random.sample(remaining_indices, min(educational_seqs, len(remaining_indices)))
            
            for idx in educational_indices:
                # 随机选择难度级别
                level = random.choices(difficulty_levels, weights=[level_weights.get(l, 0) for l in difficulty_levels])[0]
                
                s_level, labels_level, diff_level = self._generate_level_anomaly(
                    q[idx:idx+1], s[idx:idx+1], level
                )
                
                s_anomaly[idx] = s_level[0]
                anomaly_labels[idx] = labels_level[0]
                difficulty_scores[idx] = diff_level[0]
        
        return s_anomaly, anomaly_labels, difficulty_scores
    
    def _generate_level_anomaly(self, q: Optional[torch.Tensor], 
                               s: torch.Tensor,
                               level: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """生成指定难度级别的异常"""
        strategies = self.LEVEL_STRATEGIES.get(level, self.LEVEL_STRATEGIES[1])
        strategy = random.choice(strategies)
        
        if level == 1:
            return self._generate_level1_anomaly(q, s, strategy)
        elif level == 2:
            return self._generate_level2_anomaly(q, s, strategy)
        elif level == 3:
            return self._generate_level3_anomaly(q, s, strategy)
        else:  # level == 4
            return self._generate_level4_anomaly(q, s, strategy)
    
    def _generate_level1_anomaly(self, q: Optional[torch.Tensor], s: torch.Tensor, 
                                strategy: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Level 1: 简单异常 - 明显可检测"""
        s_anomaly = s.clone()
        anomaly_labels = torch.zeros_like(s, dtype=torch.float)
        
        valid_mask = (s[0] >= 0)
        valid_indices = torch.where(valid_mask)[0]
        
        if len(valid_indices) < 3:
            return s_anomaly, anomaly_labels, anomaly_labels
        
        if strategy == 'obvious_consecutive':
            # 明显的连续错误（5-8个连续的错误答案）
            start_idx = random.randint(0, max(0, len(valid_indices) - 8))
            length = random.randint(5, min(8, len(valid_indices) - start_idx))
            
            for i in range(start_idx, start_idx + length):
                pos = valid_indices[i]
                s_anomaly[0, pos] = 0  # 强制错误
                anomaly_labels[0, pos] = 1
                
        elif strategy == 'simple_random':
            # 简单随机区间
            start_idx = random.randint(0, max(0, len(valid_indices) - 5))
            length = random.randint(3, min(6, len(valid_indices) - start_idx))
            
            for i in range(start_idx, start_idx + length):
                pos = valid_indices[i]
                s_anomaly[0, pos] = random.randint(0, 1)
                anomaly_labels[0, pos] = 1
        
        # Level 1 难度分数: 0.1-0.3
        difficulty_scores = anomaly_labels * random.uniform(0.1, 0.3)
        
        return s_anomaly, anomaly_labels, difficulty_scores
    
    def _generate_level2_anomaly(self, q: Optional[torch.Tensor], s: torch.Tensor,
                                strategy: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Level 2: 中等异常 - 需要模式识别"""
        s_anomaly = s.clone()
        anomaly_labels = torch.zeros_like(s, dtype=torch.float)
        
        valid_mask = (s[0] >= 0)
        valid_indices = torch.where(valid_mask)[0]
        
        if len(valid_indices) < 5:
            return s_anomaly, anomaly_labels, anomaly_labels
        
        if strategy == 'pattern_anomaly':
            # 周期性模式异常
            pattern_length = random.randint(3, 5)
            pattern = [random.randint(0, 1) for _ in range(pattern_length)]
            
            start_idx = random.randint(0, max(0, len(valid_indices) - pattern_length * 3))
            for cycle in range(3):  # 重复3次模式
                for i, val in enumerate(pattern):
                    pos_idx = start_idx + cycle * pattern_length + i
                    if pos_idx < len(valid_indices):
                        pos = valid_indices[pos_idx]
                        s_anomaly[0, pos] = val
                        anomaly_labels[0, pos] = 1
                        
        elif strategy == 'burst_errors':
            # 突发错误模式
            n_bursts = random.randint(2, 4)
            burst_length = random.randint(2, 4)
            
            for _ in range(n_bursts):
                start_idx = random.randint(0, max(0, len(valid_indices) - burst_length))
                for i in range(burst_length):
                    if start_idx + i < len(valid_indices):
                        pos = valid_indices[start_idx + i]
                        s_anomaly[0, pos] = 0  # 错误答案
                        anomaly_labels[0, pos] = 1
        
        # Level 2 难度分数: 0.3-0.5
        difficulty_scores = anomaly_labels * random.uniform(0.3, 0.5)
        
        return s_anomaly, anomaly_labels, difficulty_scores
    
    def _generate_level3_anomaly(self, q: Optional[torch.Tensor], s: torch.Tensor,
                                strategy: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Level 3: 困难异常 - 微妙的时序异常"""
        s_anomaly = s.clone()
        anomaly_labels = torch.zeros_like(s, dtype=torch.float)
        
        valid_mask = (s[0] >= 0)
        valid_indices = torch.where(valid_mask)[0]
        
        if len(valid_indices) < 8:
            return s_anomaly, anomaly_labels, anomaly_labels
        
        if strategy == 'ability_mismatch':
            # 能力不匹配：简单题错误，难题正确
            # 简化实现：前半部分多错误，后半部分多正确
            mid_point = len(valid_indices) // 2
            
            # 前半部分增加错误
            for i in range(0, mid_point):
                if random.random() < 0.4:
                    pos = valid_indices[i]
                    s_anomaly[0, pos] = 0
                    anomaly_labels[0, pos] = 1
            
            # 后半部分增加正确
            for i in range(mid_point, len(valid_indices)):
                if random.random() < 0.3:
                    pos = valid_indices[i]
                    s_anomaly[0, pos] = 1
                    anomaly_labels[0, pos] = 1
                    
        elif strategy == 'subtle_temporal':
            # 微妙的时序异常：渐进式变化
            change_rate = 0.1
            for i, pos in enumerate(valid_indices):
                if random.random() < change_rate:
                    s_anomaly[0, pos] = 1 - s[0, pos]  # 翻转
                    anomaly_labels[0, pos] = 1
                change_rate = min(0.5, change_rate + 0.02)  # 逐渐增加
        
        # Level 3 难度分数: 0.5-0.7
        difficulty_scores = anomaly_labels * random.uniform(0.5, 0.7)
        
        return s_anomaly, anomaly_labels, difficulty_scores
    
    def _generate_level4_anomaly(self, q: Optional[torch.Tensor], s: torch.Tensor,
                                strategy: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Level 4: 极难异常 - 高级欺骗性异常"""
        s_anomaly = s.clone()
        anomaly_labels = torch.zeros_like(s, dtype=torch.float)
        
        valid_mask = (s[0] >= 0)
        valid_indices = torch.where(valid_mask)[0]
        
        if len(valid_indices) < 10:
            return s_anomaly, anomaly_labels, anomaly_labels
        
        if strategy == 'advanced_masking':
            # 高级掩蔽：在正常模式中隐藏异常
            # 大部分保持正常，少量关键位置异常
            anomaly_positions = random.sample(valid_indices.tolist(), 
                                            max(1, len(valid_indices) // 8))
            
            for pos in anomaly_positions:
                s_anomaly[0, pos] = random.randint(0, 1)
                anomaly_labels[0, pos] = 1
                
        elif strategy == 'intelligent_deception':
            # 智能欺骗：模拟真实但异常的学习模式
            # 创建看似合理但实际异常的答题模式
            for i, pos in enumerate(valid_indices):
                if i % 7 == 0:  # 每7个位置一个异常
                    s_anomaly[0, pos] = 1 - s[0, pos]
                    anomaly_labels[0, pos] = 1
        
        # Level 4 难度分数: 0.7-0.9
        difficulty_scores = anomaly_labels * random.uniform(0.7, 0.9)
        
        return s_anomaly, anomaly_labels, difficulty_scores
    
    def get_difficulty_info(self, level: int) -> Dict:
        """获取难度级别信息"""
        info_map = {
            1: {
                'name': '简单异常',
                'description': '明显可检测的异常模式',
                'detection_difficulty': '低',
                'educational_value': '中',
                'strategies': self.LEVEL_STRATEGIES[1]
            },
            2: {
                'name': '中等异常',
                'description': '需要模式识别的异常',
                'detection_difficulty': '中',
                'educational_value': '高',
                'strategies': self.LEVEL_STRATEGIES[2]
            },
            3: {
                'name': '困难异常',
                'description': '微妙的时序异常',
                'detection_difficulty': '高',
                'educational_value': '高',
                'strategies': self.LEVEL_STRATEGIES[3]
            },
            4: {
                'name': '极难异常',
                'description': '高级欺骗性异常',
                'detection_difficulty': '极高',
                'educational_value': '极高',
                'strategies': self.LEVEL_STRATEGIES[4]
            }
        }
        
        return info_map.get(level, {})
