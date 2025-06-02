"""
课程调度器

智能调度课程学习进度的核心组件。
支持多种调度策略：性能驱动、时间驱动、混合策略。

核心功能：
1. 自适应课程进度调整
2. 多维度性能监控
3. 数据集感知的调度配置
4. 课程回退机制
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum


class SchedulingStrategy(Enum):
    """调度策略枚举"""
    PERFORMANCE_DRIVEN = "performance_driven"  # 性能驱动
    TIME_DRIVEN = "time_driven"               # 时间驱动
    HYBRID = "hybrid"                         # 混合策略


@dataclass
class CurriculumPhase:
    """课程阶段配置"""
    phase_id: int
    difficulty_levels: List[int]
    level_weights: Dict[int, float]
    target_metrics: Dict[str, float]
    min_epochs: int
    max_epochs: int
    advancement_threshold: float


class CurriculumScheduler:
    """课程学习调度器"""
    
    def __init__(self, 
                 strategy: str = 'hybrid',
                 dataset_name: str = 'assist17',
                 total_epochs: int = 100):
        """
        初始化课程调度器
        
        Args:
            strategy: 调度策略 ('performance_driven', 'time_driven', 'hybrid')
            dataset_name: 数据集名称，用于自适应配置
            total_epochs: 总训练轮数
        """
        self.strategy = SchedulingStrategy(strategy)
        self.dataset_name = dataset_name
        self.total_epochs = total_epochs
        
        # 初始化课程配置
        self.curriculum_phases = self._initialize_curriculum_phases()
        self.current_phase = 0
        self.current_epoch = 0
        
        # 性能监控
        self.performance_history = []
        self.phase_history = []
        
        # 调度参数
        self.patience_counter = 0
        self.max_patience = 5
        self.regression_threshold = 0.02  # 性能回退阈值
        
    def _initialize_curriculum_phases(self) -> List[CurriculumPhase]:
        """
        初始化课程阶段配置
        
        基于数据集特征自适应配置课程阶段
        """
        if self.dataset_name in ['assist17', 'assist09']:
            # 大中型数据集：4阶段完整课程
            phases = [
                CurriculumPhase(
                    phase_id=1,
                    difficulty_levels=[1],
                    level_weights={1: 1.0},
                    target_metrics={'recall': 0.90, 'precision': 0.80},
                    min_epochs=5,
                    max_epochs=15,
                    advancement_threshold=0.85
                ),
                CurriculumPhase(
                    phase_id=2,
                    difficulty_levels=[1, 2],
                    level_weights={1: 0.7, 2: 0.3},
                    target_metrics={'recall': 0.85, 'precision': 0.75},
                    min_epochs=8,
                    max_epochs=20,
                    advancement_threshold=0.80
                ),
                CurriculumPhase(
                    phase_id=3,
                    difficulty_levels=[1, 2, 3],
                    level_weights={1: 0.3, 2: 0.5, 3: 0.2},
                    target_metrics={'recall': 0.75, 'precision': 0.70},
                    min_epochs=10,
                    max_epochs=25,
                    advancement_threshold=0.75
                ),
                CurriculumPhase(
                    phase_id=4,
                    difficulty_levels=[1, 2, 3, 4],
                    level_weights={1: 0.2, 2: 0.3, 3: 0.3, 4: 0.2},
                    target_metrics={'recall': 0.65, 'precision': 0.65},
                    min_epochs=15,
                    max_epochs=40,
                    advancement_threshold=0.70
                )
            ]
        else:
            # 小型数据集：2阶段简化课程
            phases = [
                CurriculumPhase(
                    phase_id=1,
                    difficulty_levels=[1],
                    level_weights={1: 1.0},
                    target_metrics={'recall': 0.85, 'precision': 0.75},
                    min_epochs=8,
                    max_epochs=20,
                    advancement_threshold=0.80
                ),
                CurriculumPhase(
                    phase_id=2,
                    difficulty_levels=[1, 2],
                    level_weights={1: 0.6, 2: 0.4},
                    target_metrics={'recall': 0.75, 'precision': 0.70},
                    min_epochs=15,
                    max_epochs=40,
                    advancement_threshold=0.70
                )
            ]
        
        return phases
    
    def should_advance_phase(self, current_metrics: Dict[str, float]) -> bool:
        """
        判断是否应该进入下一阶段
        
        Args:
            current_metrics: 当前性能指标 {'recall': x, 'precision': y, 'f1': z, 'auc': w}
            
        Returns:
            bool: 是否应该进入下一阶段
        """
        if self.current_phase >= len(self.curriculum_phases):
            return False
        
        current_phase_config = self.curriculum_phases[self.current_phase]
        phase_epochs = self.current_epoch - sum(
            phase.min_epochs for phase in self.curriculum_phases[:self.current_phase]
        )
        
        # 检查最小轮数要求
        if phase_epochs < current_phase_config.min_epochs:
            return False
        
        # 检查最大轮数限制
        if phase_epochs >= current_phase_config.max_epochs:
            return True
        
        # 根据调度策略判断
        if self.strategy == SchedulingStrategy.PERFORMANCE_DRIVEN:
            return self._performance_driven_advancement(current_metrics, current_phase_config)
        elif self.strategy == SchedulingStrategy.TIME_DRIVEN:
            return self._time_driven_advancement(phase_epochs, current_phase_config)
        else:  # HYBRID
            return self._hybrid_advancement(current_metrics, current_phase_config, phase_epochs)
    
    def _performance_driven_advancement(self, metrics: Dict[str, float],
                                      phase_config: CurriculumPhase) -> bool:
        """性能驱动的阶段推进判断"""
        # 检查是否达到目标性能
        target_met = True
        for metric_name, target_value in phase_config.target_metrics.items():
            if metric_name in metrics:
                if metrics[metric_name] < target_value:
                    target_met = False
                    break
        
        if target_met:
            self.patience_counter = 0
            return True
        
        # 检查性能是否停滞
        if len(self.performance_history) >= 3:
            recent_f1 = [h.get('f1', 0) for h in self.performance_history[-3:]]
            if max(recent_f1) - min(recent_f1) < 0.01:  # 性能停滞
                self.patience_counter += 1
                if self.patience_counter >= self.max_patience:
                    self.patience_counter = 0
                    return True
        
        return False
    
    def _time_driven_advancement(self, phase_epochs: int,
                               phase_config: CurriculumPhase) -> bool:
        """时间驱动的阶段推进判断"""
        # 基于时间的线性推进
        progress_ratio = phase_epochs / phase_config.max_epochs
        return progress_ratio >= 0.8  # 80%时间后推进
    
    def _hybrid_advancement(self, metrics: Dict[str, float],
                          phase_config: CurriculumPhase,
                          phase_epochs: int) -> bool:
        """混合策略的阶段推进判断"""
        # 结合性能和时间的判断
        performance_ready = self._performance_driven_advancement(metrics, phase_config)
        time_ready = phase_epochs >= phase_config.min_epochs * 1.5
        
        # 性能达标或时间充足时推进
        return performance_ready or time_ready
    
    def update(self, epoch: int, metrics: Dict[str, float]) -> Dict[str, any]:
        """
        更新调度器状态
        
        Args:
            epoch: 当前轮次
            metrics: 当前性能指标
            
        Returns:
            调度信息字典
        """
        self.current_epoch = epoch
        self.performance_history.append(metrics.copy())
        
        # 检查是否需要推进阶段
        should_advance = self.should_advance_phase(metrics)
        
        schedule_info = {
            'current_phase': self.current_phase + 1,  # 1-indexed for display
            'total_phases': len(self.curriculum_phases),
            'should_advance': should_advance,
            'phase_progress': self._calculate_phase_progress(),
            'curriculum_config': self.get_current_curriculum_config(),
            'performance_trend': self._analyze_performance_trend(),
            'recommendation': self._get_recommendation(metrics)
        }
        
        # 执行阶段推进
        if should_advance and self.current_phase < len(self.curriculum_phases) - 1:
            self.current_phase += 1
            self.phase_history.append({
                'epoch': epoch,
                'old_phase': self.current_phase - 1,
                'new_phase': self.current_phase,
                'metrics': metrics.copy()
            })
            schedule_info['phase_advanced'] = True
            schedule_info['new_phase'] = self.current_phase + 1
        else:
            schedule_info['phase_advanced'] = False
        
        return schedule_info
    
    def _calculate_phase_progress(self) -> float:
        """计算当前阶段的进度"""
        if self.current_phase >= len(self.curriculum_phases):
            return 1.0
        
        current_phase_config = self.curriculum_phases[self.current_phase]
        phase_start_epoch = sum(
            phase.min_epochs for phase in self.curriculum_phases[:self.current_phase]
        )
        phase_epochs = self.current_epoch - phase_start_epoch
        
        return min(phase_epochs / current_phase_config.max_epochs, 1.0)
    
    def _analyze_performance_trend(self) -> str:
        """分析性能趋势"""
        if len(self.performance_history) < 3:
            return 'insufficient_data'
        
        recent_f1 = [h.get('f1', 0) for h in self.performance_history[-5:]]
        
        if len(recent_f1) >= 3:
            # 计算趋势
            x = np.arange(len(recent_f1))
            slope = np.polyfit(x, recent_f1, 1)[0]
            
            if slope > 0.01:
                return 'improving'
            elif slope < -0.01:
                return 'declining'
            else:
                return 'stable'
        
        return 'stable'
    
    def _get_recommendation(self, metrics: Dict[str, float]) -> str:
        """获取调度建议"""
        if self.current_phase >= len(self.curriculum_phases):
            return 'curriculum_completed'
        
        current_phase_config = self.curriculum_phases[self.current_phase]
        
        # 检查性能是否达标
        performance_issues = []
        for metric_name, target_value in current_phase_config.target_metrics.items():
            if metric_name in metrics:
                if metrics[metric_name] < target_value * 0.9:  # 90%阈值
                    performance_issues.append(metric_name)
        
        if performance_issues:
            return f'improve_{",".join(performance_issues)}'
        
        trend = self._analyze_performance_trend()
        if trend == 'declining':
            return 'check_overfitting'
        elif trend == 'stable':
            return 'consider_advancement'
        else:
            return 'continue_training'
    
    def get_current_curriculum_config(self) -> Dict:
        """获取当前课程配置"""
        if self.current_phase >= len(self.curriculum_phases):
            # 返回最后阶段的配置
            phase_config = self.curriculum_phases[-1]
        else:
            phase_config = self.curriculum_phases[self.current_phase]
        
        return {
            'phase_id': phase_config.phase_id,
            'difficulty_levels': phase_config.difficulty_levels,
            'level_weights': phase_config.level_weights,
            'target_metrics': phase_config.target_metrics,
            'focus': self._get_phase_focus(phase_config.phase_id)
        }
    
    def _get_phase_focus(self, phase_id: int) -> str:
        """获取阶段重点"""
        focus_map = {
            1: 'building_confidence',
            2: 'gradual_complexity',
            3: 'advanced_patterns',
            4: 'expert_level'
        }
        return focus_map.get(phase_id, 'advanced_training')
    
    def should_use_baseline_anomalies(self) -> Tuple[bool, float]:
        """
        判断是否应该使用基线异常
        
        Returns:
            (should_use, ratio): 是否使用基线异常，使用比例
        """
        if self.current_phase == 0:  # 第一阶段
            return True, 0.2  # 20%基线异常用于建立信心
        elif self.current_phase == len(self.curriculum_phases) - 1:  # 最后阶段
            return True, 0.05  # 5%基线异常用于鲁棒性
        else:
            return False, 0.0
    
    def get_schedule_summary(self) -> Dict:
        """获取调度总结"""
        return {
            'strategy': self.strategy.value,
            'dataset': self.dataset_name,
            'current_phase': self.current_phase + 1,
            'total_phases': len(self.curriculum_phases),
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'phase_history': self.phase_history,
            'performance_trend': self._analyze_performance_trend(),
            'completion_ratio': self.current_epoch / self.total_epochs
        }
