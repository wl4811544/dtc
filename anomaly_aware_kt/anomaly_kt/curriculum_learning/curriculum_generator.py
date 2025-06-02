"""
课程学习异常生成器

基于课程学习的分级异常生成器。
结合教育理论和认知科学，生成不同难度级别的异常样本。

核心创新：
1. 基于教育理论的异常分级
2. 时序因果约束下的异常生成
3. 自适应的难度调整机制
4. 与课程调度器的协同工作
"""

import torch
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from .difficulty_estimator import DifficultyEstimator
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
        self.difficulty_estimator = DifficultyEstimator(dataset_name)
        self.baseline_generator = BaselineAnomalyGenerator()
        
        # 数据集特定配置
        self.dataset_config = self._get_dataset_config()
        
    def _get_dataset_config(self) -> Dict:
        """获取数据集特定配置"""
        configs = {
            'assist17': {
                'has_temporal': True,
                'has_pid': True,
                'complexity_factor': 1.0,
                'max_sequence_length': 200
            },
            'assist09': {
                'has_temporal': True,
                'has_pid': True,
                'complexity_factor': 0.9,
                'max_sequence_length': 150
            },
            'statics': {
                'has_temporal': False,
                'has_pid': False,
                'complexity_factor': 0.6,
                'max_sequence_length': 100
            },
            'algebra05': {
                'has_temporal': False,
                'has_pid': True,
                'complexity_factor': 0.7,
                'max_sequence_length': 80
            }
        }
        return configs.get(self.dataset_name, configs['assist17'])
    
    def generate_curriculum_anomalies(self, 
                                    q: torch.Tensor, 
                                    s: torch.Tensor,
                                    difficulty_levels: List[int],
                                    level_weights: Dict[int, float],
                                    anomaly_ratio: float = 0.1,
                                    include_baseline: bool = False,
                                    baseline_ratio: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        生成课程学习异常
        
        Args:
            q: 问题序列 [batch_size, seq_len]
            s: 答案序列 [batch_size, seq_len]
            difficulty_levels: 当前阶段的难度级别列表
            level_weights: 各难度级别的权重
            anomaly_ratio: 总异常比例
            include_baseline: 是否包含基线异常
            baseline_ratio: 基线异常比例
            
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
        
        # 生成基线异常
        if baseline_seqs > 0:
            baseline_indices = random.sample(range(batch_size), baseline_seqs)
            for idx in baseline_indices:
                s_baseline, labels_baseline = self.baseline_generator.generate_baseline_anomalies(
                    q[idx:idx+1] if q is not None else None,
                    s[idx:idx+1],
                    strategy='random_flip',
                    anomaly_ratio=1.0  # 对选中的序列100%添加异常
                )
                s_anomaly[idx] = s_baseline[0]
                anomaly_labels[idx] = labels_baseline[0]
                # 基线异常难度较低
                difficulty_scores[idx] = labels_baseline[0] * 0.2
        
        # 生成教育异常
        if educational_seqs > 0:
            remaining_indices = [i for i in range(batch_size) if i not in (baseline_indices if baseline_seqs > 0 else [])]
            educational_indices = random.sample(remaining_indices, min(educational_seqs, len(remaining_indices)))
            
            # 按权重分配各难度级别
            level_assignments = self._assign_difficulty_levels(educational_indices, difficulty_levels, level_weights)
            
            for level, indices in level_assignments.items():
                for idx in indices:
                    s_level, labels_level, diff_level = self._generate_level_anomaly(
                        q[idx] if q is not None else None,
                        s[idx],
                        level
                    )
                    s_anomaly[idx] = s_level
                    anomaly_labels[idx] = labels_level
                    difficulty_scores[idx] = diff_level
        
        return s_anomaly, anomaly_labels, difficulty_scores
    
    def _assign_difficulty_levels(self, indices: List[int], 
                                 difficulty_levels: List[int],
                                 level_weights: Dict[int, float]) -> Dict[int, List[int]]:
        """按权重分配难度级别"""
        assignments = {level: [] for level in difficulty_levels}
        
        # 根据权重计算每个级别的数量
        total_indices = len(indices)
        for level in difficulty_levels:
            weight = level_weights.get(level, 0)
            count = max(1, int(total_indices * weight))
            assignments[level] = indices[:count]
            indices = indices[count:]
        
        # 分配剩余的索引
        for i, idx in enumerate(indices):
            level = difficulty_levels[i % len(difficulty_levels)]
            assignments[level].append(idx)
        
        return assignments
    
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
    
    def _generate_level1_anomaly(self, q: Optional[torch.Tensor], 
                                s: torch.Tensor,
                                strategy: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Level 1: 简单异常 - 明显的连续错误、完全随机答案
        目标Recall: >90%
        """
        s_anomaly = s.clone()
        anomaly_labels = torch.zeros_like(s, dtype=torch.float)
        
        valid_mask = (s >= 0)
        valid_indices = torch.where(valid_mask)[0]
        
        if len(valid_indices) < 3:
            difficulty_scores = torch.zeros_like(s, dtype=torch.float)
            return s_anomaly, anomaly_labels, difficulty_scores
        
        if strategy == 'obvious_consecutive':
            # 明显的连续错误（5-8个连续的错误答案）
            start_idx = random.randint(0, max(0, len(valid_indices) - 8))
            length = random.randint(5, min(8, len(valid_indices) - start_idx))
            
            for i in range(start_idx, start_idx + length):
                pos = valid_indices[i]
                s_anomaly[pos] = 0  # 强制错误
                anomaly_labels[pos] = 1
                
        elif strategy == 'simple_random':
            # 简单随机区间
            start_idx = random.randint(0, max(0, len(valid_indices) - 5))
            length = random.randint(3, min(6, len(valid_indices) - start_idx))
            
            for i in range(start_idx, start_idx + length):
                pos = valid_indices[i]
                s_anomaly[pos] = random.randint(0, 1)
                anomaly_labels[pos] = 1
        
        # Level 1 难度分数: 0.1-0.3
        difficulty_scores = anomaly_labels * random.uniform(0.1, 0.3)
        
        return s_anomaly, anomaly_labels, difficulty_scores
    
    def _generate_level2_anomaly(self, q: Optional[torch.Tensor], 
                                s: torch.Tensor,
                                strategy: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Level 2: 中等异常 - 模式性异常、局部突发异常
        目标Recall: >80%
        """
        s_anomaly = s.clone()
        anomaly_labels = torch.zeros_like(s, dtype=torch.float)
        
        valid_mask = (s >= 0)
        valid_indices = torch.where(valid_mask)[0]
        
        if len(valid_indices) < 5:
            difficulty_scores = torch.zeros_like(s, dtype=torch.float)
            return s_anomaly, anomaly_labels, difficulty_scores
        
        if strategy == 'pattern_anomaly':
            # 周期性模式异常（如ABAB模式）
            pattern_length = random.choice([2, 3])
            pattern = [random.randint(0, 1) for _ in range(pattern_length)]
            
            start_idx = random.randint(0, max(0, len(valid_indices) - pattern_length * 3))
            for i in range(start_idx, min(start_idx + pattern_length * 4, len(valid_indices))):
                pos = valid_indices[i]
                pattern_value = pattern[i % pattern_length]
                if s_anomaly[pos] != pattern_value:
                    s_anomaly[pos] = pattern_value
                    anomaly_labels[pos] = 1
                    
        elif strategy == 'burst_errors':
            # 突发性错误集群
            n_bursts = random.randint(2, 3)
            burst_size = random.randint(2, 4)
            
            for _ in range(n_bursts):
                if len(valid_indices) > burst_size:
                    start_idx = random.randint(0, len(valid_indices) - burst_size)
                    for i in range(start_idx, start_idx + burst_size):
                        pos = valid_indices[i]
                        s_anomaly[pos] = 0  # 错误答案
                        anomaly_labels[pos] = 1
        
        # Level 2 难度分数: 0.3-0.5
        difficulty_scores = anomaly_labels * random.uniform(0.3, 0.5)
        
        return s_anomaly, anomaly_labels, difficulty_scores
    
    def _generate_level3_anomaly(self, q: Optional[torch.Tensor], 
                                s: torch.Tensor,
                                strategy: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Level 3: 困难异常 - 微妙的能力不匹配、复杂的作弊模式
        目标Recall: >70%
        """
        s_anomaly = s.clone()
        anomaly_labels = torch.zeros_like(s, dtype=torch.float)
        
        valid_mask = (s >= 0)
        valid_indices = torch.where(valid_mask)[0]
        
        if len(valid_indices) < 8:
            difficulty_scores = torch.zeros_like(s, dtype=torch.float)
            return s_anomaly, anomaly_labels, difficulty_scores
        
        if strategy == 'ability_mismatch':
            # 能力水平与表现不匹配
            # 前半段表现很差，后半段突然变好（或相反）
            mid_point = len(valid_indices) // 2
            
            if random.random() < 0.5:
                # 前差后好
                for i in range(mid_point):
                    if random.random() < 0.7:
                        pos = valid_indices[i]
                        s_anomaly[pos] = 0
                        anomaly_labels[pos] = 1
                        
                for i in range(mid_point, len(valid_indices)):
                    if random.random() < 0.8:
                        pos = valid_indices[i]
                        s_anomaly[pos] = 1
                        if s[pos] != 1:
                            anomaly_labels[pos] = 1
            else:
                # 前好后差
                for i in range(mid_point):
                    if random.random() < 0.8:
                        pos = valid_indices[i]
                        s_anomaly[pos] = 1
                        if s[pos] != 1:
                            anomaly_labels[pos] = 1
                            
                for i in range(mid_point, len(valid_indices)):
                    if random.random() < 0.7:
                        pos = valid_indices[i]
                        s_anomaly[pos] = 0
                        anomaly_labels[pos] = 1
                        
        elif strategy == 'subtle_temporal':
            # 微妙的时序异常
            if self.dataset_config['has_temporal']:
                # 基于位置的微妙变化
                for i, pos in enumerate(valid_indices):
                    # 在特定位置引入微妙的异常
                    if i % 7 == 0 and random.random() < 0.6:  # 不规律的间隔
                        s_anomaly[pos] = 1 - s[pos]
                        anomaly_labels[pos] = 1
        
        # Level 3 难度分数: 0.5-0.7
        difficulty_scores = anomaly_labels * random.uniform(0.5, 0.7)
        
        return s_anomaly, anomaly_labels, difficulty_scores
    
    def _generate_level4_anomaly(self, q: Optional[torch.Tensor], 
                                s: torch.Tensor,
                                strategy: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Level 4: 极难异常 - 高度伪装的异常、智能作弊行为
        目标Recall: >60%
        """
        s_anomaly = s.clone()
        anomaly_labels = torch.zeros_like(s, dtype=torch.float)
        
        valid_mask = (s >= 0)
        valid_indices = torch.where(valid_mask)[0]
        
        if len(valid_indices) < 10:
            difficulty_scores = torch.zeros_like(s, dtype=torch.float)
            return s_anomaly, anomaly_labels, difficulty_scores
        
        if strategy == 'advanced_masking':
            # 高级掩盖：在正常行为中隐藏异常
            # 大部分保持正常，只在关键位置做微调
            critical_positions = random.sample(
                valid_indices.tolist(), 
                max(2, len(valid_indices) // 8)
            )
            
            for pos in critical_positions:
                # 微妙的改变，不太明显
                if random.random() < 0.7:
                    s_anomaly[pos] = 1 - s[pos]
                    anomaly_labels[pos] = 1
                    
        elif strategy == 'intelligent_deception':
            # 智能欺骗：模拟真实的学习曲线但有异常
            # 创建看似合理的学习进步，但实际上是异常的
            
            # 生成伪学习曲线
            progress_rate = random.uniform(0.1, 0.3)
            for i, pos in enumerate(valid_indices):
                expected_performance = min(0.9, 0.3 + i * progress_rate / len(valid_indices))
                
                if random.random() < expected_performance:
                    target_answer = 1
                else:
                    target_answer = 0
                
                # 在某些位置引入微妙的异常
                if i % 11 == 0 and random.random() < 0.4:  # 很不规律的异常
                    s_anomaly[pos] = 1 - target_answer
                    anomaly_labels[pos] = 1
                else:
                    s_anomaly[pos] = target_answer
                    if target_answer != s[pos]:
                        anomaly_labels[pos] = 1
        
        # Level 4 难度分数: 0.7-0.9
        difficulty_scores = anomaly_labels * random.uniform(0.7, 0.9)
        
        return s_anomaly, anomaly_labels, difficulty_scores
    
    def estimate_generation_difficulty(self, level: int, strategy: str) -> float:
        """估算生成的异常检测难度"""
        base_difficulties = {1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8}
        base_difficulty = base_difficulties.get(level, 0.5)
        
        # 根据数据集复杂度调整
        complexity_factor = self.dataset_config['complexity_factor']
        
        return base_difficulty * complexity_factor
