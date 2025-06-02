"""
课程学习训练器

整合课程学习组件的训练器，实现基于课程学习的异常检测训练。

核心功能：
1. 与现有训练框架集成
2. 动态课程调度
3. 多阶段训练管理
4. 性能监控和分析
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path

from .curriculum_scheduler import CurriculumScheduler
from .curriculum_generator import CurriculumAnomalyGenerator
from .difficulty_estimator import DifficultyEstimator


class CurriculumTrainer:
    """课程学习训练器"""
    
    def __init__(self,
                 base_trainer,  # 现有的训练器实例
                 dataset_name: str = 'assist17',
                 strategy: str = 'hybrid',
                 total_epochs: int = 100,
                 output_dir: str = 'output/curriculum_training'):
        """
        初始化课程学习训练器

        Args:
            base_trainer: 现有的训练器实例 (EnhancedAnomalyTrainer等)
            dataset_name: 数据集名称
            strategy: 课程调度策略
            total_epochs: 总训练轮数
            output_dir: 输出目录
        """
        self.base_trainer = base_trainer
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化课程学习组件
        self.scheduler = CurriculumScheduler(strategy, dataset_name, total_epochs)
        self.generator = CurriculumAnomalyGenerator(dataset_name)
        self.difficulty_estimator = DifficultyEstimator(dataset_name)

        # 训练状态
        self.training_history = []
        self.phase_transitions = []

        # 设置日志
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('CurriculumTrainer')
        logger.setLevel(logging.INFO)
        
        # 文件处理器
        file_handler = logging.FileHandler(self.output_dir / 'curriculum_training.log')
        file_handler.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def train_with_curriculum(self, 
                            train_loader, 
                            val_loader,
                            model,
                            optimizer,
                            criterion,
                            **kwargs) -> Dict:
        """
        使用课程学习进行训练
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            model: 模型
            optimizer: 优化器
            criterion: 损失函数
            **kwargs: 其他训练参数
            
        Returns:
            训练结果字典
        """
        self.logger.info("开始课程学习训练")
        self.logger.info(f"数据集: {self.dataset_name}")
        self.logger.info(f"调度策略: {self.scheduler.strategy.value}")
        
        best_metrics = {'auc': 0.0, 'f1': 0.0}
        
        for epoch in range(self.scheduler.total_epochs):
            # 获取当前课程配置
            curriculum_config = self.scheduler.get_current_curriculum_config()
            
            # 生成课程异常数据
            train_data_with_anomalies = self._prepare_curriculum_data(
                train_loader, curriculum_config
            )
            
            # 训练一个epoch
            train_metrics = self._train_epoch(
                train_data_with_anomalies, model, optimizer, criterion, epoch
            )
            
            # 验证
            val_metrics = self._validate_epoch(val_loader, model, criterion, epoch)
            
            # 更新调度器
            schedule_info = self.scheduler.update(epoch, val_metrics)
            
            # 记录训练历史
            epoch_info = {
                'epoch': epoch,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'schedule_info': schedule_info,
                'curriculum_config': curriculum_config
            }
            self.training_history.append(epoch_info)
            
            # 记录阶段转换
            if schedule_info.get('phase_advanced', False):
                transition_info = {
                    'epoch': epoch,
                    'old_phase': schedule_info['current_phase'] - 1,
                    'new_phase': schedule_info['current_phase'],
                    'metrics': val_metrics.copy()
                }
                self.phase_transitions.append(transition_info)
                self.logger.info(f"阶段转换: Phase {transition_info['old_phase']} -> Phase {transition_info['new_phase']}")
            
            # 更新最佳指标
            if val_metrics.get('auc', 0) > best_metrics['auc']:
                best_metrics = val_metrics.copy()
                self._save_best_model(model, epoch, val_metrics)
            
            # 打印进度
            self._print_epoch_progress(epoch, train_metrics, val_metrics, schedule_info)
            
            # 早停检查
            if self._should_early_stop(epoch, val_metrics):
                self.logger.info(f"早停触发，在第 {epoch} 轮停止训练")
                break
        
        # 训练完成后的分析
        final_results = self._analyze_training_results(best_metrics)

        return final_results
    
    def _prepare_curriculum_data(self, train_loader, curriculum_config) -> List:
        """准备课程学习数据"""
        curriculum_data = []
        
        difficulty_levels = curriculum_config['difficulty_levels']
        level_weights = curriculum_config['level_weights']
        
        # 判断是否使用基线异常
        use_baseline, baseline_ratio = self.scheduler.should_use_baseline_anomalies()
        
        for batch_idx, batch in enumerate(train_loader):
            # 正确获取批次数据
            if len(batch.data) == 2:
                q, s = batch.get("q", "s")
                pid = None
            else:
                q, s, pid = batch.get("q", "s", "pid")
            
            # 生成课程异常
            s_anomaly, anomaly_labels, difficulty_scores = self.generator.generate_curriculum_anomalies(
                q, s,
                difficulty_levels=difficulty_levels,
                level_weights=level_weights,
                anomaly_ratio=0.1,  # 可配置
                include_baseline=use_baseline,
                baseline_ratio=baseline_ratio
            )
            
            curriculum_data.append({
                'questions': q,
                'answers': s_anomaly,
                'pid': pid,
                'anomaly_labels': anomaly_labels,
                'difficulty_scores': difficulty_scores,
                'original_answers': s
            })
        
        return curriculum_data
    
    def _train_epoch(self, curriculum_data, model, optimizer, criterion, epoch) -> Dict:
        """训练一个epoch - 专门用于异常检测器训练"""
        model.train()
        total_loss = 0.0
        total_samples = 0
        total_correct = 0
        total_predictions = 0

        for batch_data in curriculum_data:
            optimizer.zero_grad()

            # 获取批次数据
            questions = batch_data['questions'].to(model.device)
            answers = batch_data['answers'].to(model.device)
            pid = batch_data['pid']
            if pid is not None:
                pid = pid.to(model.device)
            anomaly_labels = batch_data['anomaly_labels'].to(model.device)

            # 异常检测器前向传播
            # 输入: questions, answers, pid (可选)
            # 输出: 异常概率 [batch_size, seq_len]
            if pid is not None:
                anomaly_probs = model(questions, answers, pid)
            else:
                anomaly_probs = model(questions, answers)

            # 创建有效位置的掩码（排除padding）
            valid_mask = (questions != -1) & (answers != -1)

            # 只在有效位置计算损失
            valid_probs = anomaly_probs[valid_mask]
            valid_labels = anomaly_labels[valid_mask].float()

            if valid_probs.numel() > 0:
                # 计算二分类损失
                loss = criterion(valid_probs, valid_labels)

                # 反向传播
                loss.backward()
                optimizer.step()

                # 统计
                total_loss += loss.item()
                total_samples += valid_labels.size(0)

                # 计算准确率
                predictions = (torch.sigmoid(valid_probs) > 0.5).float()
                total_correct += (predictions == valid_labels).sum().item()
                total_predictions += valid_labels.size(0)

        avg_loss = total_loss / len(curriculum_data) if len(curriculum_data) > 0 else 0.0
        accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'samples': total_samples
        }
    
    def _validate_epoch(self, val_loader, model, criterion, epoch) -> Dict:
        """验证一个epoch - 专门用于异常检测器验证"""
        model.eval()

        total_loss = 0.0
        all_probs = []
        all_labels = []
        total_correct = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in val_loader:
                # 获取批次数据
                if len(batch.data) == 2:
                    q, s = batch.get("q", "s")
                    pid = None
                else:
                    q, s, pid = batch.get("q", "s", "pid")

                q = q.to(model.device)
                s = s.to(model.device)
                if pid is not None:
                    pid = pid.to(model.device)

                # 生成验证用的异常数据
                s_anomaly, anomaly_labels, _ = self.generator.generate_curriculum_anomalies(
                    q, s,
                    difficulty_levels=[1, 2, 3, 4],  # 使用所有难度级别
                    level_weights={1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25},
                    anomaly_ratio=0.1,
                    include_baseline=False,
                    baseline_ratio=0.0
                )

                s_anomaly = s_anomaly.to(model.device)
                anomaly_labels = anomaly_labels.to(model.device)

                # 异常检测器前向传播
                if pid is not None:
                    anomaly_probs = model(q, s_anomaly, pid)
                else:
                    anomaly_probs = model(q, s_anomaly)

                # 创建有效位置的掩码
                valid_mask = (q != -1) & (s_anomaly != -1)

                # 只在有效位置计算指标
                valid_probs = anomaly_probs[valid_mask]
                valid_labels = anomaly_labels[valid_mask].float()

                if valid_probs.numel() > 0:
                    # 计算损失
                    loss = criterion(valid_probs, valid_labels)
                    total_loss += loss.item()

                    # 收集预测和标签用于计算AUC等指标
                    probs = torch.sigmoid(valid_probs).cpu().numpy()
                    labels = valid_labels.cpu().numpy()

                    all_probs.extend(probs)
                    all_labels.extend(labels)

                    # 计算准确率
                    predictions = (torch.sigmoid(valid_probs) > 0.5).float()
                    total_correct += (predictions == valid_labels).sum().item()
                    total_predictions += valid_labels.size(0)

        # 计算指标
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0

        # 计算AUC, F1等指标
        if len(all_probs) > 0:
            from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

            all_probs = np.array(all_probs)
            all_labels = np.array(all_labels)

            # 确保有正负样本
            if len(np.unique(all_labels)) > 1:
                auc = roc_auc_score(all_labels, all_probs)
                predictions = (all_probs > 0.5).astype(int)
                f1 = f1_score(all_labels, predictions)
                precision = precision_score(all_labels, predictions, zero_division=0)
                recall = recall_score(all_labels, predictions, zero_division=0)
            else:
                auc = f1 = precision = recall = 0.0
        else:
            auc = f1 = precision = recall = 0.0

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def _print_epoch_progress(self, epoch, train_metrics, val_metrics, schedule_info):
        """打印训练进度"""
        phase_info = f"Phase {schedule_info['current_phase']}/{schedule_info['total_phases']}"

        print(f"Epoch {epoch+1:3d} | {phase_info} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Train Acc: {train_metrics.get('accuracy', 0):.4f} | "
              f"Val AUC: {val_metrics['auc']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f} | "
              f"Val Acc: {val_metrics.get('accuracy', 0):.4f}")

        if schedule_info.get('phase_advanced', False):
            print(f"  🎓 进入新阶段: Phase {schedule_info['new_phase']}")
            print(f"  📊 详细指标: Precision={val_metrics.get('precision', 0):.4f}, "
                  f"Recall={val_metrics.get('recall', 0):.4f}")

        recommendation = schedule_info.get('recommendation', '')
        if recommendation and recommendation != 'continue_training':
            print(f"  💡 建议: {recommendation}")
    
    def _should_early_stop(self, epoch, val_metrics) -> bool:
        """早停判断"""
        # 简单的早停逻辑，可以根据需要扩展
        if len(self.training_history) < 10:
            return False
        
        recent_aucs = [h['val_metrics']['auc'] for h in self.training_history[-10:]]
        if max(recent_aucs) - min(recent_aucs) < 0.001:  # 性能停滞
            return True
        
        return False
    
    def _save_best_model(self, model, epoch, metrics):
        """保存最佳模型"""
        save_path = self.output_dir / 'best_curriculum_model.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'scheduler_state': self.scheduler.get_schedule_summary()
        }, save_path)
        
        self.logger.info(f"保存最佳模型: AUC={metrics['auc']:.4f}")
    
    def _analyze_training_results(self, best_metrics) -> Dict:
        """分析训练结果"""
        analysis = {
            'best_metrics': best_metrics,
            'total_epochs': len(self.training_history),
            'phase_transitions': self.phase_transitions,
            'final_phase': self.scheduler.current_phase + 1,
            'scheduler_summary': self.scheduler.get_schedule_summary()
        }
        
        # 保存分析结果
        import json
        with open(self.output_dir / 'training_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        self.logger.info("训练完成，结果已保存")
        
        return analysis
    
    def get_training_summary(self) -> Dict:
        """获取训练总结"""
        if not self.training_history:
            return {}
        
        return {
            'dataset': self.dataset_name,
            'strategy': self.scheduler.strategy.value,
            'total_epochs': len(self.training_history),
            'best_auc': max(h['val_metrics']['auc'] for h in self.training_history),
            'phase_transitions': len(self.phase_transitions),
            'final_phase': self.scheduler.current_phase + 1
        }
