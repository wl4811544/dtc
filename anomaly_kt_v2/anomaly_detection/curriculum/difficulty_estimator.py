"""
难度评估器

评估异常检测的难度和模型性能。
完全基于DTransformer原始代码，不依赖任何anomaly_kt模块。
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


class DifficultyEstimator:
    """异常检测难度评估器"""
    
    def __init__(self, window_size: int = 100):
        """
        初始化难度评估器
        
        Args:
            window_size: 滑动窗口大小
        """
        self.window_size = window_size
        self.performance_history = []
        self.difficulty_history = []
        
    def estimate_detection_difficulty(self, 
                                    predictions: torch.Tensor,
                                    labels: torch.Tensor,
                                    difficulty_scores: Optional[torch.Tensor] = None) -> Dict:
        """
        评估检测难度
        
        Args:
            predictions: 模型预测 [batch_size, seq_len]
            labels: 真实标签 [batch_size, seq_len]
            difficulty_scores: 预设难度分数 [batch_size, seq_len]
            
        Returns:
            难度评估结果
        """
        # 展平并过滤有效数据
        pred_flat = predictions.view(-1)
        labels_flat = labels.view(-1)
        
        # 只考虑有标签的位置
        valid_mask = (labels_flat >= 0)
        pred_valid = pred_flat[valid_mask]
        labels_valid = labels_flat[valid_mask]
        
        if len(labels_valid) == 0:
            return self._empty_result()
        
        # 基础性能指标
        basic_metrics = self._compute_basic_metrics(pred_valid, labels_valid)
        
        # 难度相关指标
        difficulty_metrics = self._compute_difficulty_metrics(pred_valid, labels_valid)
        
        # 如果有预设难度分数，计算相关性
        correlation_metrics = {}
        if difficulty_scores is not None:
            diff_flat = difficulty_scores.view(-1)[valid_mask]
            correlation_metrics = self._compute_correlation_metrics(
                pred_valid, labels_valid, diff_flat
            )
        
        # 综合难度评估
        overall_difficulty = self._compute_overall_difficulty(
            basic_metrics, difficulty_metrics, correlation_metrics
        )
        
        result = {
            'overall_difficulty': overall_difficulty,
            'basic_metrics': basic_metrics,
            'difficulty_metrics': difficulty_metrics,
            'correlation_metrics': correlation_metrics,
            'sample_count': len(labels_valid),
            'anomaly_ratio': labels_valid.float().mean().item()
        }
        
        # 更新历史记录
        self.performance_history.append(basic_metrics['auc'])
        self.difficulty_history.append(overall_difficulty)
        
        return result
    
    def _compute_basic_metrics(self, predictions: torch.Tensor, 
                              labels: torch.Tensor) -> Dict:
        """计算基础性能指标"""
        pred_np = predictions.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        # AUC
        try:
            auc_score = roc_auc_score(labels_np, pred_np)
        except:
            auc_score = 0.5
        
        # Precision-Recall AUC
        try:
            precision, recall, _ = precision_recall_curve(labels_np, pred_np)
            pr_auc = auc(recall, precision)
        except:
            pr_auc = 0.0
        
        # 准确率（使用0.5阈值）
        pred_binary = (pred_np > 0.5).astype(int)
        accuracy = (pred_binary == labels_np).mean()

        # 精确率和召回率
        tp = ((pred_binary == 1) & (labels_np == 1)).sum()
        fp = ((pred_binary == 1) & (labels_np == 0)).sum()
        fn = ((pred_binary == 0) & (labels_np == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'auc': auc_score,
            'pr_auc': pr_auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _compute_difficulty_metrics(self, predictions: torch.Tensor,
                                   labels: torch.Tensor) -> Dict:
        """计算难度相关指标"""
        pred_np = predictions.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        # 预测置信度分析
        confidence_scores = np.abs(pred_np - 0.5) * 2  # 转换到[0,1]
        
        # 异常样本的平均置信度
        anomaly_mask = (labels_np == 1)
        normal_mask = (labels_np == 0)
        
        anomaly_confidence = confidence_scores[anomaly_mask].mean() if anomaly_mask.sum() > 0 else 0.0
        normal_confidence = confidence_scores[normal_mask].mean() if normal_mask.sum() > 0 else 0.0
        
        # 置信度差异（越小说明越难区分）
        confidence_gap = abs(anomaly_confidence - normal_confidence)
        
        # 预测分布分析
        pred_std = pred_np.std()
        pred_entropy = self._compute_entropy(pred_np)
        
        # 分类边界分析
        boundary_samples = np.sum((pred_np > 0.4) & (pred_np < 0.6)) / len(pred_np)
        
        return {
            'anomaly_confidence': anomaly_confidence,
            'normal_confidence': normal_confidence,
            'confidence_gap': confidence_gap,
            'prediction_std': pred_std,
            'prediction_entropy': pred_entropy,
            'boundary_ratio': boundary_samples
        }
    
    def _compute_correlation_metrics(self, predictions: torch.Tensor,
                                    labels: torch.Tensor,
                                    difficulty_scores: torch.Tensor) -> Dict:
        """计算与预设难度的相关性"""
        pred_np = predictions.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        diff_np = difficulty_scores.detach().cpu().numpy()
        
        # 只考虑异常样本
        anomaly_mask = (labels_np == 1)
        if anomaly_mask.sum() == 0:
            return {}
        
        anomaly_pred = pred_np[anomaly_mask]
        anomaly_diff = diff_np[anomaly_mask]
        
        # 计算相关性 - 正确处理标准差为0的情况
        if len(anomaly_pred) <= 1:
            correlation = 0.0
        else:
            # 检查数据的变异性
            pred_std = np.std(anomaly_pred)
            diff_std = np.std(anomaly_diff)

            if pred_std == 0 and diff_std == 0:
                # 两个变量都没有变异 - 完全相关
                correlation = 1.0
            elif pred_std == 0 or diff_std == 0:
                # 其中一个变量没有变异 - 无法计算相关性
                correlation = 0.0
            else:
                # 正常计算相关性
                try:
                    correlation = np.corrcoef(anomaly_pred, anomaly_diff)[0, 1]
                    if np.isnan(correlation):
                        correlation = 0.0
                except:
                    correlation = 0.0
        
        # 难度分层分析
        difficulty_levels = self._analyze_by_difficulty_levels(anomaly_pred, anomaly_diff)
        
        return {
            'difficulty_correlation': correlation,
            'difficulty_levels': difficulty_levels
        }
    
    def _analyze_by_difficulty_levels(self, predictions: np.ndarray,
                                     difficulty_scores: np.ndarray) -> Dict:
        """按难度级别分析性能"""
        if len(predictions) == 0:
            return {}
        
        # 将难度分为4个级别
        quartiles = np.percentile(difficulty_scores, [25, 50, 75])
        
        levels = {}
        for i, (low, high) in enumerate([(0, quartiles[0]), 
                                        (quartiles[0], quartiles[1]),
                                        (quartiles[1], quartiles[2]),
                                        (quartiles[2], 1.0)]):
            mask = (difficulty_scores >= low) & (difficulty_scores < high)
            if mask.sum() > 0:
                level_pred = predictions[mask]
                levels[f'level_{i+1}'] = {
                    'count': mask.sum(),
                    'mean_prediction': level_pred.mean(),
                    'std_prediction': level_pred.std(),
                    'difficulty_range': (low, high)
                }
        
        return levels
    
    def _compute_entropy(self, predictions: np.ndarray) -> float:
        """计算预测熵"""
        # 将预测转换为概率分布
        p = np.clip(predictions, 1e-7, 1-1e-7)
        entropy = -(p * np.log(p) + (1-p) * np.log(1-p)).mean()
        return entropy
    
    def _compute_overall_difficulty(self, basic_metrics: Dict,
                                   difficulty_metrics: Dict,
                                   correlation_metrics: Dict) -> float:
        """计算综合难度分数"""
        # 基于多个指标的加权组合
        
        # AUC越低，难度越高
        auc_difficulty = 1.0 - basic_metrics['auc']
        
        # 置信度差异越小，难度越高
        confidence_difficulty = 1.0 - difficulty_metrics['confidence_gap']
        
        # 边界样本比例越高，难度越高
        boundary_difficulty = difficulty_metrics['boundary_ratio']
        
        # 预测熵越高，难度越高
        entropy_difficulty = difficulty_metrics['prediction_entropy'] / np.log(2)  # 归一化
        
        # 加权平均
        weights = [0.4, 0.3, 0.2, 0.1]
        difficulties = [auc_difficulty, confidence_difficulty, boundary_difficulty, entropy_difficulty]
        
        overall = sum(w * d for w, d in zip(weights, difficulties))
        
        return np.clip(overall, 0.0, 1.0)
    
    def _empty_result(self) -> Dict:
        """返回空结果"""
        return {
            'overall_difficulty': 0.5,
            'basic_metrics': {'auc': 0.5, 'accuracy': 0.5, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
            'difficulty_metrics': {},
            'correlation_metrics': {},
            'sample_count': 0,
            'anomaly_ratio': 0.0
        }
    
    def get_performance_trend(self, window: Optional[int] = None) -> Dict:
        """获取性能趋势"""
        if not self.performance_history:
            return {}
        
        window = window or self.window_size
        recent_performance = self.performance_history[-window:]
        recent_difficulty = self.difficulty_history[-window:]
        
        return {
            'recent_auc_mean': np.mean(recent_performance),
            'recent_auc_std': np.std(recent_performance),
            'recent_difficulty_mean': np.mean(recent_difficulty),
            'auc_trend': np.polyfit(range(len(recent_performance)), recent_performance, 1)[0] if len(recent_performance) > 1 else 0.0,
            'total_samples': len(self.performance_history)
        }
