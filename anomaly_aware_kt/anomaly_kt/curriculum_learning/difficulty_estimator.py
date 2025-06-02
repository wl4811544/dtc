"""
难度评估器

基于教育理论和统计分析的异常检测难度评估器。
结合认知负荷理论、IRT理论和时序分析来评估异常样本的检测难度。

核心创新：
1. 时序因果约束下的难度评估（只使用历史信息）
2. 认知理论驱动的多维度评估
3. 自适应的难度校准机制
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
import math


class DifficultyEstimator:
    """异常检测难度评估器"""
    
    def __init__(self, dataset_name: str = 'assist17'):
        """
        初始化难度评估器
        
        Args:
            dataset_name: 数据集名称，用于自适应配置
        """
        self.dataset_name = dataset_name
        self.difficulty_factors = self._get_dataset_factors()
        self.calibration_history = []
        
    def _get_dataset_factors(self) -> Dict[str, float]:
        """获取数据集特定的难度因子权重"""
        factors = {
            'assist17': {
                'temporal_weight': 0.3,      # 时序复杂度权重
                'statistical_weight': 0.3,   # 统计复杂度权重
                'cognitive_weight': 0.4,     # 认知复杂度权重
                'context_weight': 0.2        # 上下文复杂度权重
            },
            'assist09': {
                'temporal_weight': 0.2,
                'statistical_weight': 0.4,
                'cognitive_weight': 0.4,
                'context_weight': 0.2
            },
            'statics': {
                'temporal_weight': 0.1,
                'statistical_weight': 0.7,   # 主要依赖统计模式
                'cognitive_weight': 0.2,
                'context_weight': 0.1
            },
            'algebra05': {
                'temporal_weight': 0.2,
                'statistical_weight': 0.5,
                'cognitive_weight': 0.3,
                'context_weight': 0.2
            }
        }
        return factors.get(self.dataset_name, factors['assist17'])
    
    def estimate_sample_difficulty(self, q: torch.Tensor, s: torch.Tensor, 
                                 anomaly_labels: torch.Tensor,
                                 position: int,
                                 window_size: int = 10) -> float:
        """
        评估单个样本在指定位置的异常检测难度
        
        核心创新：严格遵循时序因果约束，只使用历史信息
        
        Args:
            q: 问题序列 [seq_len]
            s: 答案序列 [seq_len]
            anomaly_labels: 异常标签 [seq_len]
            position: 当前评估位置
            window_size: 历史窗口大小
            
        Returns:
            difficulty_score: 难度分数 [0, 1]，越高越难检测
        """
        # 确保只使用历史信息（时序因果约束）
        start_pos = max(0, position - window_size)
        historical_q = q[start_pos:position] if q is not None else None
        historical_s = s[start_pos:position]
        historical_labels = anomaly_labels[start_pos:position]
        
        if len(historical_s) == 0:
            return 0.5  # 无历史信息时返回中等难度
        
        # 多维度难度评估
        temporal_difficulty = self._compute_temporal_difficulty(
            historical_s, historical_labels, position
        )
        
        statistical_difficulty = self._compute_statistical_difficulty(
            historical_s, historical_labels
        )
        
        cognitive_difficulty = self._compute_cognitive_difficulty(
            historical_q, historical_s, historical_labels
        )
        
        context_difficulty = self._compute_context_difficulty(
            historical_q, historical_s, position
        )
        
        # 加权组合
        weights = self.difficulty_factors
        final_difficulty = (
            temporal_difficulty * weights['temporal_weight'] +
            statistical_difficulty * weights['statistical_weight'] +
            cognitive_difficulty * weights['cognitive_weight'] +
            context_difficulty * weights['context_weight']
        )
        
        # 归一化到[0, 1]
        return np.clip(final_difficulty, 0.0, 1.0)
    
    def _compute_temporal_difficulty(self, historical_s: torch.Tensor,
                                   historical_labels: torch.Tensor,
                                   position: int) -> float:
        """
        计算时序复杂度
        
        基于时间序列分析的难度评估：
        - 序列的自相关性
        - 趋势变化的复杂度
        - 异常模式的时序特征
        """
        if len(historical_s) < 3:
            return 0.5
        
        # 1. 计算序列自相关性
        s_np = historical_s.cpu().numpy().astype(float)
        if len(s_np) > 1:
            autocorr = np.corrcoef(s_np[:-1], s_np[1:])[0, 1]
            if np.isnan(autocorr):
                autocorr = 0.0
        else:
            autocorr = 0.0
        
        # 2. 计算趋势变化复杂度
        if len(s_np) > 2:
            diff = np.diff(s_np)
            trend_complexity = np.std(diff) if len(diff) > 0 else 0.0
        else:
            trend_complexity = 0.0
        
        # 3. 异常模式的时序分布
        labels_np = historical_labels.cpu().numpy()
        if np.sum(labels_np) > 0:
            # 异常位置的分布熵
            anomaly_positions = np.where(labels_np > 0)[0]
            if len(anomaly_positions) > 1:
                intervals = np.diff(anomaly_positions)
                interval_entropy = stats.entropy(np.histogram(intervals, bins=3)[0] + 1e-8)
                temporal_pattern_complexity = interval_entropy / np.log(3)  # 归一化
            else:
                temporal_pattern_complexity = 0.0
        else:
            temporal_pattern_complexity = 0.0
        
        # 组合时序难度指标
        temporal_difficulty = (
            abs(autocorr) * 0.4 +  # 高自相关增加预测难度
            trend_complexity * 0.3 +
            temporal_pattern_complexity * 0.3
        )
        
        return temporal_difficulty
    
    def _compute_statistical_difficulty(self, historical_s: torch.Tensor,
                                      historical_labels: torch.Tensor) -> float:
        """
        计算统计复杂度
        
        基于信息论和统计学的难度评估：
        - 序列熵
        - 异常密度
        - 模式复杂度
        """
        if len(historical_s) == 0:
            return 0.5
        
        s_np = historical_s.cpu().numpy()
        labels_np = historical_labels.cpu().numpy()
        
        # 1. 序列熵（信息复杂度）
        if len(s_np) > 0:
            unique, counts = np.unique(s_np, return_counts=True)
            probabilities = counts / len(s_np)
            sequence_entropy = stats.entropy(probabilities)
            normalized_entropy = sequence_entropy / np.log(len(unique)) if len(unique) > 1 else 0.0
        else:
            normalized_entropy = 0.0
        
        # 2. 异常密度
        anomaly_density = np.mean(labels_np) if len(labels_np) > 0 else 0.0
        
        # 3. 模式复杂度（基于游程长度）
        if len(s_np) > 1:
            runs = []
            current_run = 1
            for i in range(1, len(s_np)):
                if s_np[i] == s_np[i-1]:
                    current_run += 1
                else:
                    runs.append(current_run)
                    current_run = 1
            runs.append(current_run)
            
            pattern_complexity = np.std(runs) / np.mean(runs) if len(runs) > 0 and np.mean(runs) > 0 else 0.0
        else:
            pattern_complexity = 0.0
        
        # 组合统计难度指标
        statistical_difficulty = (
            normalized_entropy * 0.4 +
            anomaly_density * 0.3 +
            np.clip(pattern_complexity, 0, 1) * 0.3
        )
        
        return statistical_difficulty
    
    def _compute_cognitive_difficulty(self, historical_q: Optional[torch.Tensor],
                                    historical_s: torch.Tensor,
                                    historical_labels: torch.Tensor) -> float:
        """
        计算认知复杂度
        
        基于认知负荷理论的难度评估：
        - 内在认知负荷（题目本身难度）
        - 外在认知负荷（异常模式复杂度）
        - 相关认知负荷（学习过程复杂度）
        """
        if len(historical_s) == 0:
            return 0.5
        
        # 1. 内在认知负荷（基于答题表现推断题目难度）
        s_np = historical_s.cpu().numpy()
        if len(s_np) > 0:
            # 错误率作为题目难度的代理指标
            error_rate = 1 - np.mean(s_np)
            intrinsic_load = error_rate
        else:
            intrinsic_load = 0.5
        
        # 2. 外在认知负荷（异常模式的复杂度）
        labels_np = historical_labels.cpu().numpy()
        if np.sum(labels_np) > 0:
            # 异常模式的分散程度
            anomaly_positions = np.where(labels_np > 0)[0]
            if len(anomaly_positions) > 1:
                # 计算异常位置的分散度
                position_variance = np.var(anomaly_positions) / (len(historical_s) ** 2)
                extraneous_load = position_variance
            else:
                extraneous_load = 0.0
        else:
            extraneous_load = 0.0
        
        # 3. 相关认知负荷（学习过程的复杂度）
        if len(s_np) > 2:
            # 基于学习曲线的复杂度
            learning_trend = np.polyfit(range(len(s_np)), s_np, 1)[0]  # 线性趋势
            learning_complexity = abs(learning_trend)  # 趋势变化的绝对值
            germane_load = np.clip(learning_complexity, 0, 1)
        else:
            germane_load = 0.0
        
        # 组合认知难度指标（基于认知负荷理论）
        cognitive_difficulty = (
            intrinsic_load * 0.4 +
            extraneous_load * 0.3 +
            germane_load * 0.3
        )
        
        return cognitive_difficulty
    
    def _compute_context_difficulty(self, historical_q: Optional[torch.Tensor],
                                  historical_s: torch.Tensor,
                                  position: int) -> float:
        """
        计算上下文复杂度
        
        基于问题上下文和序列位置的难度评估：
        - 问题类型多样性
        - 序列位置效应
        - 上下文切换复杂度
        """
        if len(historical_s) == 0:
            return 0.5
        
        # 1. 序列位置效应（序列越长，上下文越复杂）
        position_effect = min(position / 50.0, 1.0)  # 归一化到[0,1]
        
        # 2. 问题类型多样性（如果有问题ID）
        if historical_q is not None and len(historical_q) > 0:
            q_np = historical_q.cpu().numpy()
            unique_questions = len(np.unique(q_np))
            question_diversity = min(unique_questions / len(q_np), 1.0)
            
            # 3. 上下文切换复杂度
            if len(q_np) > 1:
                switches = np.sum(q_np[1:] != q_np[:-1])
                switch_complexity = switches / (len(q_np) - 1)
            else:
                switch_complexity = 0.0
        else:
            question_diversity = 0.5  # 无问题信息时使用中等值
            switch_complexity = 0.5
        
        # 组合上下文难度指标
        context_difficulty = (
            position_effect * 0.4 +
            question_diversity * 0.3 +
            switch_complexity * 0.3
        )
        
        return context_difficulty
    
    def calibrate_difficulty(self, predicted_difficulties: List[float],
                           actual_detection_results: List[bool]) -> None:
        """
        基于实际检测结果校准难度评估
        
        Args:
            predicted_difficulties: 预测的难度分数列表
            actual_detection_results: 实际检测结果列表（True=检测成功，False=检测失败）
        """
        if len(predicted_difficulties) != len(actual_detection_results):
            return
        
        # 记录校准历史
        calibration_data = {
            'predicted': predicted_difficulties,
            'actual': actual_detection_results,
            'correlation': np.corrcoef(predicted_difficulties, 
                                     [1-x for x in actual_detection_results])[0, 1]
        }
        
        self.calibration_history.append(calibration_data)
        
        # 如果校准历史过长，保留最近的记录
        if len(self.calibration_history) > 10:
            self.calibration_history = self.calibration_history[-10:]
    
    def get_calibration_stats(self) -> Dict:
        """获取校准统计信息"""
        if not self.calibration_history:
            return {}
        
        recent_correlations = [data['correlation'] for data in self.calibration_history 
                             if not np.isnan(data['correlation'])]
        
        return {
            'num_calibrations': len(self.calibration_history),
            'avg_correlation': np.mean(recent_correlations) if recent_correlations else 0.0,
            'calibration_trend': 'improving' if len(recent_correlations) > 1 and 
                               recent_correlations[-1] > recent_correlations[0] else 'stable'
        }
    
    def estimate_batch_difficulty(self, q: torch.Tensor, s: torch.Tensor,
                                anomaly_labels: torch.Tensor) -> torch.Tensor:
        """
        批量评估异常检测难度
        
        Args:
            q: 问题序列 [batch_size, seq_len]
            s: 答案序列 [batch_size, seq_len]
            anomaly_labels: 异常标签 [batch_size, seq_len]
            
        Returns:
            difficulty_scores: 难度分数 [batch_size, seq_len]
        """
        batch_size, seq_len = s.shape
        difficulty_scores = torch.zeros_like(s, dtype=torch.float)
        
        for batch_idx in range(batch_size):
            for pos in range(seq_len):
                if s[batch_idx, pos] >= 0:  # 有效位置
                    difficulty = self.estimate_sample_difficulty(
                        q[batch_idx] if q is not None else None,
                        s[batch_idx],
                        anomaly_labels[batch_idx],
                        pos
                    )
                    difficulty_scores[batch_idx, pos] = difficulty
        
        return difficulty_scores
