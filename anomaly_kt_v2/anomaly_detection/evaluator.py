"""
异常检测评估器

提供全面的异常检测性能评估功能。
完全基于DTransformer原始代码，不依赖任何anomaly_kt模块。
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


class AnomalyDetectionEvaluator:
    """异常检测评估器"""
    
    def __init__(self):
        """初始化评估器"""
        self.evaluation_history = []
    
    def evaluate_model(self,
                      model: torch.nn.Module,
                      data_loader,
                      device: str = 'cuda',
                      anomaly_strategies: List[str] = None,
                      with_pid: bool = True) -> Dict:
        """
        全面评估异常检测模型
        
        Args:
            model: 异常检测模型
            data_loader: 数据加载器
            device: 设备
            anomaly_strategies: 要测试的异常策略列表
            
        Returns:
            评估结果字典
        """
        if anomaly_strategies is None:
            anomaly_strategies = ['random_flip', 'uniform_random', 'systematic_bias']
        
        model.eval()
        results = {}
        
        # 对每种异常策略进行评估
        for strategy in anomaly_strategies:
            print(f"评估异常策略: {strategy}")
            strategy_result = self._evaluate_strategy(model, data_loader, device, strategy, with_pid)
            results[strategy] = strategy_result
        
        # 计算综合指标
        overall_result = self._compute_overall_metrics(results)
        results['overall'] = overall_result
        
        # 保存评估历史
        self.evaluation_history.append(results)
        
        return results
    
    def _evaluate_strategy(self,
                          model: torch.nn.Module,
                          data_loader,
                          device: str,
                          strategy: str,
                          with_pid: bool = True) -> Dict:
        """评估特定异常策略"""
        from .generators.baseline_generator import BaselineAnomalyGenerator
        
        generator = BaselineAnomalyGenerator()
        
        all_predictions = []
        all_labels = []
        all_losses = []
        
        with torch.no_grad():
            for batch in data_loader:
                # 获取批次数据
                if with_pid:
                    q, s, pid = batch.get("q", "s", "pid")
                else:
                    q, s = batch.get("q", "s")
                    pid = None
                
                q = q.to(device)
                s = s.to(device)
                if pid is not None:
                    pid = pid.to(device)
                
                # 生成异常数据
                s_anomaly, anomaly_labels = generator.generate_baseline_anomalies(
                    q, s, strategy=strategy, anomaly_ratio=0.2
                )
                
                # 模型预测
                logits = model(q, s_anomaly, pid)
                loss = model.get_loss(q, s_anomaly, anomaly_labels, pid)
                
                # 收集结果
                pred_probs = torch.sigmoid(logits)
                mask = (s_anomaly >= 0)
                
                all_predictions.extend(pred_probs[mask].cpu().numpy())
                all_labels.extend(anomaly_labels[mask].cpu().numpy())
                all_losses.append(loss.item())
        
        # 计算指标
        if all_predictions:
            metrics = self._compute_metrics(
                np.array(all_predictions),
                np.array(all_labels)
            )
            metrics['avg_loss'] = np.mean(all_losses)
            metrics['sample_count'] = len(all_predictions)
            metrics['anomaly_ratio'] = np.mean(all_labels)
        else:
            metrics = self._empty_metrics()
        
        return metrics
    
    def _compute_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> Dict:
        """计算详细的评估指标"""
        # 基础指标
        try:
            auc_score = roc_auc_score(labels, predictions)
        except:
            auc_score = 0.5
        
        try:
            precision, recall, _ = precision_recall_curve(labels, predictions)
            pr_auc = auc(recall, precision)
        except:
            pr_auc = 0.0
        
        # 多个阈值下的指标
        thresholds = [0.3, 0.5, 0.7]
        threshold_metrics = {}
        
        for threshold in thresholds:
            pred_binary = (predictions > threshold).astype(int)
            
            # 混淆矩阵
            tn, fp, fn, tp = confusion_matrix(labels, pred_binary).ravel()
            
            # 计算指标
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            threshold_metrics[f'threshold_{threshold}'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1': f1,
                'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
            }
        
        # 预测分布分析
        distribution_metrics = self._analyze_prediction_distribution(predictions, labels)
        
        return {
            'auc': auc_score,
            'pr_auc': pr_auc,
            'threshold_metrics': threshold_metrics,
            'distribution_metrics': distribution_metrics
        }
    
    def _analyze_prediction_distribution(self, predictions: np.ndarray, labels: np.ndarray) -> Dict:
        """分析预测分布"""
        # 分离正常和异常样本的预测
        normal_pred = predictions[labels == 0]
        anomaly_pred = predictions[labels == 1]
        
        # 统计指标
        normal_stats = {
            'mean': normal_pred.mean() if len(normal_pred) > 0 else 0.0,
            'std': normal_pred.std() if len(normal_pred) > 0 else 0.0,
            'median': np.median(normal_pred) if len(normal_pred) > 0 else 0.0
        }
        
        anomaly_stats = {
            'mean': anomaly_pred.mean() if len(anomaly_pred) > 0 else 0.0,
            'std': anomaly_pred.std() if len(anomaly_pred) > 0 else 0.0,
            'median': np.median(anomaly_pred) if len(anomaly_pred) > 0 else 0.0
        }
        
        # 分离度指标
        separation = abs(anomaly_stats['mean'] - normal_stats['mean'])
        
        # 重叠度分析
        overlap_ratio = self._compute_overlap_ratio(normal_pred, anomaly_pred)
        
        return {
            'normal_stats': normal_stats,
            'anomaly_stats': anomaly_stats,
            'separation': separation,
            'overlap_ratio': overlap_ratio
        }
    
    def _compute_overlap_ratio(self, normal_pred: np.ndarray, anomaly_pred: np.ndarray) -> float:
        """计算预测分布的重叠比例"""
        if len(normal_pred) == 0 or len(anomaly_pred) == 0:
            return 0.0
        
        # 使用直方图估计重叠
        bins = np.linspace(0, 1, 21)
        normal_hist, _ = np.histogram(normal_pred, bins=bins, density=True)
        anomaly_hist, _ = np.histogram(anomaly_pred, bins=bins, density=True)
        
        # 计算重叠面积
        overlap = np.minimum(normal_hist, anomaly_hist).sum()
        total = (normal_hist.sum() + anomaly_hist.sum()) / 2
        
        return overlap / total if total > 0 else 0.0
    
    def _compute_overall_metrics(self, strategy_results: Dict) -> Dict:
        """计算综合指标"""
        if not strategy_results:
            return self._empty_metrics()
        
        # 收集所有策略的AUC
        aucs = []
        pr_aucs = []
        sample_counts = []
        
        for strategy, result in strategy_results.items():
            if 'auc' in result:
                aucs.append(result['auc'])
                pr_aucs.append(result.get('pr_auc', 0.0))
                sample_counts.append(result.get('sample_count', 0))
        
        if not aucs:
            return self._empty_metrics()
        
        # 计算平均指标
        avg_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        avg_pr_auc = np.mean(pr_aucs)
        total_samples = sum(sample_counts)
        
        # 性能等级评估
        performance_grade = self._assess_performance_grade(avg_auc)
        
        return {
            'avg_auc': avg_auc,
            'std_auc': std_auc,
            'avg_pr_auc': avg_pr_auc,
            'total_samples': total_samples,
            'num_strategies': len(aucs),
            'performance_grade': performance_grade,
            'strategy_aucs': dict(zip(strategy_results.keys(), aucs))
        }
    
    def _assess_performance_grade(self, auc: float) -> str:
        """评估性能等级"""
        if auc >= 0.9:
            return 'Excellent'
        elif auc >= 0.8:
            return 'Good'
        elif auc >= 0.7:
            return 'Fair'
        elif auc >= 0.6:
            return 'Poor'
        else:
            return 'Very Poor'
    
    def _empty_metrics(self) -> Dict:
        """返回空指标"""
        return {
            'auc': 0.5,
            'pr_auc': 0.0,
            'avg_loss': float('inf'),
            'sample_count': 0,
            'anomaly_ratio': 0.0
        }
    
    def generate_report(self, results: Dict, save_path: Optional[str] = None) -> str:
        """生成评估报告"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("异常检测模型评估报告")
        report_lines.append("=" * 60)
        
        # 综合指标
        if 'overall' in results:
            overall = results['overall']
            report_lines.append(f"\n📊 综合性能:")
            report_lines.append(f"  平均AUC: {overall.get('avg_auc', 0):.4f} ± {overall.get('std_auc', 0):.4f}")
            report_lines.append(f"  平均PR-AUC: {overall.get('avg_pr_auc', 0):.4f}")
            report_lines.append(f"  性能等级: {overall.get('performance_grade', 'Unknown')}")
            report_lines.append(f"  测试样本数: {overall.get('total_samples', 0)}")
        
        # 各策略详细结果
        report_lines.append(f"\n📋 各异常策略详细结果:")
        for strategy, result in results.items():
            if strategy == 'overall':
                continue
            
            report_lines.append(f"\n  {strategy}:")
            report_lines.append(f"    AUC: {result.get('auc', 0):.4f}")
            report_lines.append(f"    PR-AUC: {result.get('pr_auc', 0):.4f}")
            report_lines.append(f"    样本数: {result.get('sample_count', 0)}")
            report_lines.append(f"    异常比例: {result.get('anomaly_ratio', 0):.3f}")
            
            # 阈值指标
            if 'threshold_metrics' in result:
                report_lines.append(f"    阈值指标:")
                for threshold, metrics in result['threshold_metrics'].items():
                    report_lines.append(f"      {threshold}: F1={metrics.get('f1', 0):.3f}, "
                                      f"Precision={metrics.get('precision', 0):.3f}, "
                                      f"Recall={metrics.get('recall', 0):.3f}")
        
        report_text = "\n".join(report_lines)
        
        # 保存报告
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
        
        return report_text
