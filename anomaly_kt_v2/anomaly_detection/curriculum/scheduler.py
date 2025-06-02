"""
课程调度器

管理课程学习的进度和难度调度。
完全基于DTransformer原始代码，不依赖任何anomaly_kt模块。
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


class CurriculumScheduler:
    """课程学习调度器"""
    
    def __init__(self, 
                 total_epochs: int = 50,
                 initial_difficulty: float = 0.1,
                 final_difficulty: float = 0.8,
                 schedule_type: str = 'linear',
                 warmup_epochs: int = 5):
        """
        初始化课程调度器
        
        Args:
            total_epochs: 总训练轮数
            initial_difficulty: 初始难度
            final_difficulty: 最终难度
            schedule_type: 调度类型 ('linear', 'exponential', 'cosine', 'step')
            warmup_epochs: 预热轮数
        """
        self.total_epochs = total_epochs
        self.initial_difficulty = initial_difficulty
        self.final_difficulty = final_difficulty
        self.schedule_type = schedule_type
        self.warmup_epochs = warmup_epochs
        
        self.current_epoch = 0
        self.current_difficulty = initial_difficulty
        
    def step(self, epoch: int) -> Dict:
        """
        更新课程进度
        
        Args:
            epoch: 当前轮数
            
        Returns:
            课程状态字典
        """
        self.current_epoch = epoch
        
        # 计算难度进度
        if epoch < self.warmup_epochs:
            # 预热阶段：保持初始难度
            progress = 0.0
        else:
            # 正常调度阶段
            adjusted_epoch = epoch - self.warmup_epochs
            adjusted_total = self.total_epochs - self.warmup_epochs
            progress = min(1.0, adjusted_epoch / max(1, adjusted_total))
        
        # 根据调度类型计算难度
        self.current_difficulty = self._compute_difficulty(progress)
        
        # 计算难度级别分布
        level_weights = self._compute_level_weights(self.current_difficulty)
        
        # 计算异常比例
        anomaly_ratio = self._compute_anomaly_ratio(progress)
        
        return {
            'epoch': epoch,
            'progress': progress,
            'difficulty': self.current_difficulty,
            'level_weights': level_weights,
            'anomaly_ratio': anomaly_ratio,
            'phase': self._get_current_phase(progress)
        }
    
    def _compute_difficulty(self, progress: float) -> float:
        """根据进度计算当前难度"""
        if self.schedule_type == 'linear':
            return self.initial_difficulty + progress * (self.final_difficulty - self.initial_difficulty)
        
        elif self.schedule_type == 'exponential':
            # 指数增长
            alpha = math.log(self.final_difficulty / self.initial_difficulty)
            return self.initial_difficulty * math.exp(alpha * progress)
        
        elif self.schedule_type == 'cosine':
            # 余弦退火
            return self.initial_difficulty + 0.5 * (self.final_difficulty - self.initial_difficulty) * \
                   (1 - math.cos(math.pi * progress))
        
        elif self.schedule_type == 'step':
            # 阶梯式
            if progress < 0.25:
                return self.initial_difficulty
            elif progress < 0.5:
                return self.initial_difficulty + 0.3 * (self.final_difficulty - self.initial_difficulty)
            elif progress < 0.75:
                return self.initial_difficulty + 0.6 * (self.final_difficulty - self.initial_difficulty)
            else:
                return self.final_difficulty
        
        else:
            return self.initial_difficulty + progress * (self.final_difficulty - self.initial_difficulty)
    
    def _compute_level_weights(self, difficulty: float) -> Dict[int, float]:
        """根据当前难度计算各级别权重"""
        # 基础权重分布
        if difficulty < 0.2:
            # 早期：主要是简单异常
            weights = {1: 0.7, 2: 0.3, 3: 0.0, 4: 0.0}
        elif difficulty < 0.4:
            # 初期：简单和中等
            weights = {1: 0.5, 2: 0.4, 3: 0.1, 4: 0.0}
        elif difficulty < 0.6:
            # 中期：中等为主
            weights = {1: 0.2, 2: 0.5, 3: 0.3, 4: 0.0}
        elif difficulty < 0.8:
            # 后期：困难为主
            weights = {1: 0.1, 2: 0.3, 3: 0.5, 4: 0.1}
        else:
            # 最终：包含所有级别
            weights = {1: 0.1, 2: 0.2, 3: 0.4, 4: 0.3}
        
        return weights
    
    def _compute_anomaly_ratio(self, progress: float) -> float:
        """计算异常数据比例"""
        # 异常比例随进度逐渐增加
        min_ratio = 0.05  # 最小5%
        max_ratio = 0.2   # 最大20%
        
        return min_ratio + progress * (max_ratio - min_ratio)
    
    def _get_current_phase(self, progress: float) -> str:
        """获取当前训练阶段"""
        if progress < 0.25:
            return 'warmup'
        elif progress < 0.5:
            return 'early'
        elif progress < 0.75:
            return 'middle'
        else:
            return 'advanced'
    
    def get_schedule_info(self) -> Dict:
        """获取调度器信息"""
        return {
            'total_epochs': self.total_epochs,
            'current_epoch': self.current_epoch,
            'schedule_type': self.schedule_type,
            'initial_difficulty': self.initial_difficulty,
            'final_difficulty': self.final_difficulty,
            'current_difficulty': self.current_difficulty,
            'warmup_epochs': self.warmup_epochs
        }
    
    def reset(self):
        """重置调度器"""
        self.current_epoch = 0
        self.current_difficulty = self.initial_difficulty
    
    def preview_schedule(self, num_points: int = 20) -> List[Dict]:
        """预览整个调度过程"""
        preview = []
        original_epoch = self.current_epoch
        
        for i in range(num_points):
            epoch = int(i * self.total_epochs / (num_points - 1))
            state = self.step(epoch)
            preview.append(state)
        
        # 恢复原始状态
        self.current_epoch = original_epoch
        
        return preview


class AdaptiveCurriculumScheduler(CurriculumScheduler):
    """自适应课程调度器 - 根据模型性能动态调整"""
    
    def __init__(self, 
                 total_epochs: int = 50,
                 initial_difficulty: float = 0.1,
                 final_difficulty: float = 0.8,
                 adaptation_rate: float = 0.1,
                 performance_threshold: float = 0.8):
        """
        初始化自适应调度器
        
        Args:
            adaptation_rate: 自适应调整率
            performance_threshold: 性能阈值
        """
        super().__init__(total_epochs, initial_difficulty, final_difficulty)
        self.adaptation_rate = adaptation_rate
        self.performance_threshold = performance_threshold
        self.performance_history = []
        
    def step_with_performance(self, epoch: int, performance: float) -> Dict:
        """
        根据性能更新课程
        
        Args:
            epoch: 当前轮数
            performance: 当前性能指标 (如AUC)
            
        Returns:
            课程状态字典
        """
        self.performance_history.append(performance)
        
        # 基础调度
        base_state = self.step(epoch)
        
        # 自适应调整
        if len(self.performance_history) >= 3:
            recent_performance = np.mean(self.performance_history[-3:])
            
            if recent_performance > self.performance_threshold:
                # 性能良好，可以增加难度
                adjustment = self.adaptation_rate
            else:
                # 性能不佳，降低难度
                adjustment = -self.adaptation_rate
            
            # 调整当前难度
            self.current_difficulty = np.clip(
                self.current_difficulty + adjustment,
                self.initial_difficulty,
                self.final_difficulty
            )
            
            # 重新计算权重
            base_state['difficulty'] = self.current_difficulty
            base_state['level_weights'] = self._compute_level_weights(self.current_difficulty)
            base_state['adaptation'] = adjustment
        
        return base_state
