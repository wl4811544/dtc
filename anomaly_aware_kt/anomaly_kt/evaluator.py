"""
评估器模块

提供异常检测和知识追踪的评估功能。
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from DTransformer.eval import Evaluator as KTEvaluator


class AnomalyEvaluator:
    """异常检测评估器"""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """重置统计量"""
        self.predictions = []
        self.labels = []
        self.questions = []
        self.answers = []
    
    def update(self, predictions: torch.Tensor, labels: torch.Tensor,
               questions: Optional[torch.Tensor] = None,
               answers: Optional[torch.Tensor] = None):
        """更新统计量"""
        mask = (labels >= 0)
        
        self.predictions.extend(predictions[mask].cpu().numpy())
        self.labels.extend(labels[mask].cpu().numpy())
        
        if questions is not None:
            self.questions.extend(questions[mask].cpu().numpy())
        if answers is not None:
            self.answers.extend(answers[mask].cpu().numpy())
    
    def compute_metrics(self) -> Dict[str, float]:
        """计算评估指标"""
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        
        # 尝试多个阈值
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        best_f1 = 0
        best_threshold = 0.5
        best_metrics = {}
        
        for threshold in thresholds:
            pred_binary = (predictions >= threshold).astype(int)
            
            # 混淆矩阵
            if len(np.unique(labels)) > 1 and len(np.unique(pred_binary)) > 1:
                tn, fp, fn, tp = confusion_matrix(labels, pred_binary).ravel()
            else:
                # 处理所有预测为同一类的情况
                if pred_binary.sum() == 0:  # 全部预测为0
                    tp = fp = 0
                    tn = (labels == 0).sum()
                    fn = (labels == 1).sum()
                else:  # 全部预测为1
                    tn = fn = 0
                    tp = (labels == 1).sum()
                    fp = (labels == 0).sum()
            
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    'recall': recall,
                    'precision': precision,
                    'f1_score': f1,
                    'threshold': threshold
                }
        
        # 使用最佳阈值计算最终指标
        pred_binary = (predictions >= best_threshold).astype(int)
        
        if len(np.unique(labels)) > 1 and len(np.unique(pred_binary)) > 1:
            tn, fp, fn, tp = confusion_matrix(labels, pred_binary).ravel()
        else:
            if pred_binary.sum() == 0:
                tp = fp = 0
                tn = (labels == 0).sum()
                fn = (labels == 1).sum()
            else:
                tn = fn = 0
                tp = (labels == 1).sum()
                fp = (labels == 0).sum()
        
        # 基础指标
        metrics = {
            'recall': best_metrics.get('recall', 0),
            'precision': best_metrics.get('precision', 0),
            'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'f1_score': best_metrics.get('f1_score', 0),
            'best_threshold': best_threshold
        }
        
        # AUC-ROC
        try:
            metrics['auc_roc'] = roc_auc_score(labels, predictions)
        except:
            metrics['auc_roc'] = 0.0
        
        # 分布指标
        normal_scores = predictions[labels == 0]
        anomaly_scores = predictions[labels == 1]
        
        if len(normal_scores) > 0 and len(anomaly_scores) > 0:
            # 分离度
            metrics['score_separation'] = (
                (anomaly_scores.mean() - normal_scores.mean()) /
                (normal_scores.std() + anomaly_scores.std() + 1e-6)
            )
            
            # 重叠度
            overlap_min = max(normal_scores.min(), anomaly_scores.min())
            overlap_max = min(normal_scores.max(), anomaly_scores.max())
            if overlap_max > overlap_min:
                total_range = max(normal_scores.max(), anomaly_scores.max()) - \
                             min(normal_scores.min(), anomaly_scores.min())
                metrics['score_overlap'] = 1 - (overlap_max - overlap_min) / total_range
            else:
                metrics['score_overlap'] = 1.0
        
        # 误报率
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        return metrics
    
    def plot_score_distribution(self, save_path: Optional[str] = None):
        """绘制分数分布图"""
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        
        plt.figure(figsize=(10, 6))
        
        # 分别绘制正常和异常的分布
        plt.hist(predictions[labels == 0], bins=50, alpha=0.5, 
                label='Normal', density=True, color='blue')
        plt.hist(predictions[labels == 1], bins=50, alpha=0.5, 
                label='Anomaly', density=True, color='red')
        
        # 添加阈值线
        plt.axvline(x=self.threshold, color='green', linestyle='--', 
                   label=f'Threshold={self.threshold}')
        
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.title('Score Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def find_best_threshold(self, metric: str = 'f1_score') -> float:
        """找到最佳阈值"""
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        
        best_threshold = 0.5
        best_score = 0
        
        # 尝试不同阈值
        for threshold in np.linspace(0.1, 0.9, 17):
            pred_binary = (predictions >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(labels, pred_binary).ravel()
            
            if metric == 'f1_score':
                score = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            elif metric == 'recall':
                score = tp / (tp + fn) if (tp + fn) > 0 else 0
            elif metric == 'precision':
                score = tp / (tp + fp) if (tp + fp) > 0 else 0
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold


class ComparisonEvaluator:
    """比较评估器 - 用于比较基线和异常感知模型"""
    
    def __init__(self):
        self.baseline_evaluator = KTEvaluator()
        self.anomaly_aware_evaluator = KTEvaluator()
    
    def evaluate_models(self, test_loader, baseline_model, anomaly_aware_model, device='cpu'):
        """评估两个模型"""
        baseline_model.eval()
        anomaly_aware_model.eval()
        
        # 重置评估器
        self.baseline_evaluator = KTEvaluator()
        self.anomaly_aware_evaluator = KTEvaluator()
        
        with torch.no_grad():
            for batch in test_loader:
                # 获取数据
                if len(batch.data) == 2:
                    q, s = batch.get("q", "s")
                    pid = None
                else:
                    q, s, pid = batch.get("q", "s", "pid")
                
                q, s = q.to(device), s.to(device)
                if pid is not None:
                    pid = pid.to(device)
                
                # 基线模型预测
                y_baseline, *_ = baseline_model.predict(q, s, pid)
                self.baseline_evaluator.evaluate(s, torch.sigmoid(y_baseline))
                
                # 异常感知模型预测
                if hasattr(anomaly_aware_model, 'predict_with_anomaly'):
                    y_anomaly, *_ = anomaly_aware_model.predict_with_anomaly(q, s, pid)
                else:
                    y_anomaly, *_ = anomaly_aware_model.predict(q, s, pid)
                self.anomaly_aware_evaluator.evaluate(s, torch.sigmoid(y_anomaly))
        
        # 获取结果
        baseline_metrics = self.baseline_evaluator.report()
        anomaly_metrics = self.anomaly_aware_evaluator.report()
        
        # 计算提升
        improvements = {}
        for metric in ['acc', 'auc', 'mae', 'rmse']:
            baseline_val = baseline_metrics[metric]
            anomaly_val = anomaly_metrics[metric]
            
            if metric in ['mae', 'rmse']:
                # 误差指标，越小越好
                improvement = (baseline_val - anomaly_val) / baseline_val * 100
            else:
                # 准确率指标，越大越好
                improvement = (anomaly_val - baseline_val) / baseline_val * 100
            
            improvements[metric] = improvement
        
        return {
            'baseline': baseline_metrics,
            'anomaly_aware': anomaly_metrics,
            'improvements': improvements
        }
    
    def print_comparison(self, results: Dict):
        """打印比较结果"""
        print("\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        
        print("\nBaseline Model:")
        for metric, value in results['baseline'].items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        print("\nAnomaly-Aware Model:")
        for metric, value in results['anomaly_aware'].items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        print("\nImprovements:")
        for metric, improvement in results['improvements'].items():
            symbol = "↑" if improvement > 0 else "↓"
            print(f"  {metric.upper()}: {improvement:+.2f}% {symbol}")
        
        # 检查是否达到目标
        auc_improvement = results['improvements']['auc']
        if auc_improvement >= 1.0:
            print(f"\n✓ SUCCESS: AUC improved by {auc_improvement:.2f}% (target: ≥1%)")
        else:
            print(f"\n✗ Target not met: AUC improved by {auc_improvement:.2f}% (target: ≥1%)")
        
        print("="*60)


def plot_training_curves(history: Dict, save_dir: str):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss曲线
    if 'train_loss' in history:
        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 检出率和准确率
    if 'val_recall' in history and 'val_precision' in history:
        axes[0, 1].plot(history['val_recall'], label='Recall (检出率)', marker='o')
        axes[0, 1].plot(history['val_precision'], label='Precision (检出准确率)', marker='s')
        axes[0, 1].set_title('Detection Metrics')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # F1分数
    if 'val_f1_score' in history:
        axes[1, 0].plot(history['val_f1_score'], label='F1 Score', marker='o', color='green')
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # AUC-ROC
    if 'val_auc_roc' in history:
        axes[1, 1].plot(history['val_auc_roc'], label='AUC-ROC', marker='o', color='orange')
        axes[1, 1].set_title('AUC-ROC Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.png")
    plt.close()