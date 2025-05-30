"""
统一的异常检测器训练器

支持三种训练策略：Basic, Enhanced, Aggressive
"""

import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, Optional, Any

from .trainer import BaseTrainer
from .training_strategies import StrategyFactory, TrainingStrategy
from .evaluator import AnomalyEvaluator


class UnifiedAnomalyTrainer(BaseTrainer):
    """统一的异常检测器训练器"""

    def __init__(self,
                 model: nn.Module,
                 device: str = 'cpu',
                 save_dir: str = 'output/detector',
                 patience: int = 10,
                 strategy: str = 'basic'):
        """
        初始化统一训练器

        Args:
            model: 异常检测模型
            device: 设备
            save_dir: 保存目录
            patience: 早停耐心值
            strategy: 训练策略 ('basic', 'enhanced', 'aggressive')
        """
        super().__init__(model, device, save_dir, patience)

        # 创建训练策略
        self.strategy = StrategyFactory.create_strategy(
            strategy, model, device, save_dir, patience
        )
        self.strategy_name = strategy

        # 评估器
        self.evaluator = AnomalyEvaluator()

        print(f"🎯 使用训练策略: {self.strategy.get_strategy_name()}")

    def train(self,
              train_loader,
              val_loader,
              epochs: Optional[int] = None,
              learning_rate: Optional[float] = None,
              anomaly_ratio: Optional[float] = None,
              optimize_for: Optional[str] = None,
              **kwargs) -> Dict[str, float]:
        """
        训练异常检测器

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数 (None则使用策略默认值)
            learning_rate: 学习率 (None则使用策略默认值)
            anomaly_ratio: 异常比例 (None则使用策略默认值)
            optimize_for: 优化目标 (None则使用策略默认值)
            **kwargs: 其他参数
        """

        # 获取策略默认参数
        default_params = self.strategy.get_default_params()

        # 合并参数
        config = {
            'epochs': epochs or default_params.get('epochs', 30),
            'learning_rate': learning_rate or default_params.get('learning_rate', 1e-3),
            'anomaly_ratio': anomaly_ratio or default_params.get('anomaly_ratio', 0.1),
            'optimize_for': optimize_for or default_params.get('optimize_for', 'f1_score'),
            'strategy': self.strategy_name,
            **{k: v for k, v in default_params.items() if k not in ['epochs', 'learning_rate', 'anomaly_ratio', 'optimize_for']},
            **kwargs
        }

        # 保存配置
        with open(os.path.join(self.save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n📋 训练配置 ({self.strategy.get_strategy_name()} Strategy):")
        for key, value in config.items():
            if not key.startswith('_'):
                print(f"  {key}: {value}")

        # 显示优化目标建议
        self._print_optimization_advice(config['optimize_for'], self.strategy_name)

        # 创建优化器和调度器
        optimizer = self.strategy.create_optimizer(config['learning_rate'])
        scheduler = self.strategy.create_scheduler(optimizer, total_epochs=config['epochs'])

        # 训练循环
        no_improve = 0
        best_score = 0
        start_epoch = config.get('start_epoch', 0)  # 支持从指定轮次开始

        print(f"\n🚀 开始训练 ({self.strategy.get_strategy_name()} Strategy)")
        if start_epoch > 0:
            print(f"🔄 从第 {start_epoch} 轮继续训练")
        print(f"📊 总轮数: {config['epochs']}, 优化目标: {config['optimize_for']}")
        print(f"⚙️  学习率: {config['learning_rate']}, 异常比例: {config['anomaly_ratio']}")

        # 启用调试模式
        self._debug_anomaly_density = True

        for epoch in range(start_epoch, config['epochs']):
            print(f"\n{'='*20} Epoch {epoch+1}/{config['epochs']} {'='*20}")

            # 训练一个epoch
            print("🔄 训练中...")
            train_metrics = self.strategy.train_epoch(
                train_loader, optimizer, scheduler, epoch,
                total_epochs=config['epochs'],
                **config
            )

            # 验证
            print("📊 验证中...")
            self._density_samples = []  # 重置密度样本
            val_metrics = self._validate_epoch(val_loader, config['anomaly_ratio'], epoch)

            # 计算平均异常密度
            if hasattr(self, '_density_samples') and self._density_samples:
                avg_density = sum(self._density_samples) / len(self._density_samples)
                val_metrics['actual_anomaly_density'] = avg_density

            # 计算预测统计（如果有的话）
            pred_stats = None
            if hasattr(self, '_pred_stats') and self._pred_stats:
                pred_stats = {
                    'mean': sum(s['mean'] for s in self._pred_stats) / len(self._pred_stats),
                    'max': max(s['max'] for s in self._pred_stats),
                    'min': min(s['min'] for s in self._pred_stats),
                    'labels_mean': sum(s['labels_mean'] for s in self._pred_stats) / len(self._pred_stats)
                }

            # 记录历史
            self.history['train_loss'].append(train_metrics['loss'])
            for k, v in val_metrics.items():
                self.history[f'val_{k}'].append(v)

            # 打印结果
            self._print_epoch_results(epoch + 1, train_metrics, val_metrics, config, pred_stats)

            # 调整学习率
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics[config['optimize_for']])
                else:
                    scheduler.step()

            # 记录当前学习率
            self._current_lr = optimizer.param_groups[0]['lr']

            # 总是保存当前轮次模型
            self.save_checkpoint(epoch, optimizer, val_metrics, is_best=False)

            # 检查最佳模型
            current_score = val_metrics[config['optimize_for']]
            if current_score > best_score:
                best_score = current_score
                self.best_metrics = val_metrics.copy()
                self.best_epoch = epoch + 1
                self.save_checkpoint(epoch, optimizer, val_metrics, is_best=True)
                no_improve = 0
                print(f"  ✅ 新的最佳 {config['optimize_for']}: {current_score:.4f}")
            else:
                no_improve += 1

            # 保存特殊里程碑模型
            self._save_milestone_models(epoch, optimizer, val_metrics)

            # 智能停止检查
            should_stop, stop_reason = self._check_intelligent_stopping(
                epoch + 1, config['epochs'], val_metrics,
                pred_stats,
                config['optimize_for'], no_improve, self.patience
            )

            if should_stop:
                print(f"\n🛑 智能停止触发: {stop_reason}")
                break

            # 传统早停检查
            elif no_improve >= self.patience:
                print(f"\n⏹️  早停: {no_improve} 轮无改善")
                break

            # 激进策略的特殊处理
            if self.strategy_name == 'aggressive':
                self._aggressive_emergency_check(epoch, val_metrics, optimizer, config)

        print(f"\n🎉 训练完成!")
        print(f"  🏆 最佳轮次: {self.best_epoch}")
        print(f"  📈 最佳 {config['optimize_for']}: {best_score:.4f}")

        return self.best_metrics

    def _validate_epoch(self, val_loader, anomaly_ratio: float, epoch: int = 0) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        self.evaluator.reset()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                q, s, pid = self._get_batch_data(batch)

                # 使用与训练相同的异常生成方式
                if hasattr(self.strategy.generator, 'generate_anomalies_enhanced'):
                    # Enhanced/Aggressive 策略使用增强生成器
                    s_anomaly, labels = self.strategy.generator.generate_anomalies_enhanced(
                        q, s,
                        anomaly_ratio=anomaly_ratio,
                        min_anomaly_density=0.3 if self.strategy_name == 'enhanced' else 0.4,
                        progressive_difficulty=False,  # 验证时不使用渐进式
                        epoch=0
                    )
                else:
                    # Basic 策略使用原始生成器
                    s_anomaly, labels = self.strategy.generator.generate_anomalies(
                        q, s, anomaly_ratio=anomaly_ratio
                    )

                # 调试：检查异常密度
                if hasattr(self, '_debug_anomaly_density'):
                    valid_mask = (s >= 0)
                    if valid_mask.sum() > 0:
                        actual_density = labels[valid_mask].float().mean().item()
                        if not hasattr(self, '_density_samples'):
                            self._density_samples = []
                        self._density_samples.append(actual_density)

                # 预测
                logits = self.model(q, s_anomaly, pid)
                predictions = torch.sigmoid(logits)  # 转换为概率

                # 收集预测统计（每轮都收集）
                if predictions.dim() > 1:
                    pred_probs = predictions.squeeze(-1)
                else:
                    pred_probs = predictions

                # 统计预测分布
                pred_mean = pred_probs.mean().item()
                pred_max = pred_probs.max().item()
                pred_min = pred_probs.min().item()

                if not hasattr(self, '_pred_stats'):
                    self._pred_stats = []
                self._pred_stats.append({
                    'mean': pred_mean,
                    'max': pred_max,
                    'min': pred_min,
                    'labels_mean': labels.float().mean().item()
                })

                # 更新评估器
                self.evaluator.update(predictions, labels, q, s)

        return self.evaluator.compute_metrics()

    def _print_epoch_results(self, epoch: int, train_metrics: Dict, val_metrics: Dict, config: Dict, pred_stats: Dict = None):
        """打印epoch结果"""
        print(f"\n📊 Epoch {epoch} 结果:")

        # 训练指标
        train_info = [f"Loss: {train_metrics['loss']:.4f}"]
        if 'positive_ratio' in train_metrics:
            train_info.append(f"正样本比例: {train_metrics['positive_ratio']:.1%}")
        if 'anomaly_ratio' in train_metrics:
            train_info.append(f"异常比例: {train_metrics['anomaly_ratio']:.1%}")

        print(f"  📈 训练 - {', '.join(train_info)}")

        # 验证指标
        print(f"  📊 验证 - Recall: {val_metrics['recall']:.3f}, "
              f"Precision: {val_metrics['precision']:.3f}, "
              f"F1: {val_metrics['f1_score']:.3f}, "
              f"AUC: {val_metrics['auc_roc']:.3f}")

        # 显示实际异常密度
        if 'actual_anomaly_density' in val_metrics:
            print(f"  🔍 实际异常密度: {val_metrics['actual_anomaly_density']:.1%}")

        # 每轮显示预测分析
        if pred_stats is not None:
            # 分析预测行为和给出建议
            analysis, suggestion = self._analyze_predictions_with_suggestion(pred_stats, val_metrics, epoch, config)
            print(f"  📊 预测分析: {analysis}")
            print(f"    - 预测概率: 平均={pred_stats['mean']:.3f}, 最大={pred_stats['max']:.3f}, 最小={pred_stats['min']:.3f}")
            print(f"    - 标签比例: {pred_stats['labels_mean']:.1%}")
            if suggestion:
                print(f"  💡 训练建议: {suggestion}")

        # 当前最佳指标
        current_score = val_metrics[config['optimize_for']]
        print(f"  🎯 当前 {config['optimize_for']}: {current_score:.4f}")

        # 学习率（从优化器获取当前学习率）
        if hasattr(self, '_current_lr'):
            print(f"  📉 学习率: {self._current_lr:.2e}")

    def _analyze_predictions_with_suggestion(self, pred_stats: Dict, val_metrics: Dict,
                                           epoch: int, config: Dict) -> tuple[str, str]:
        """分析预测行为并根据轮次给出训练建议"""
        mean_pred = pred_stats['mean']
        max_pred = pred_stats['max']
        min_pred = pred_stats['min']
        label_ratio = pred_stats['labels_mean']

        recall = val_metrics['recall']
        precision = val_metrics['precision']
        auc = val_metrics['auc_roc']
        f1 = val_metrics['f1_score']

        total_epochs = config['epochs']
        optimize_target = config['optimize_for']

        # 基础性能分析
        if auc > 0.90:
            analysis = "🟢 模型性能优秀，区分能力强"
        elif auc > 0.80:
            analysis = "🟡 模型性能良好，继续优化"
        elif auc > 0.70:
            analysis = "🟠 模型学习中，有改进空间"
        elif auc > 0.60:
            analysis = "🔴 模型性能一般，需要关注"
        else:
            analysis = "🚨 模型性能较差，考虑调整"

        # 根据轮次和性能给出建议
        suggestion = self._get_training_suggestion(epoch, total_epochs, val_metrics, pred_stats, optimize_target)

        return analysis, suggestion

    def _get_training_suggestion(self, epoch: int, total_epochs: int, val_metrics: Dict,
                               pred_stats: Dict, optimize_target: str) -> str:
        """根据当前轮次和性能给出具体的训练建议"""
        recall = val_metrics['recall']
        precision = val_metrics['precision']
        auc = val_metrics['auc_roc']
        f1 = val_metrics['f1_score']
        mean_pred = pred_stats['mean']

        progress = epoch / total_epochs

        # 早期阶段 (前20%)
        if progress <= 0.2:
            if recall < 0.1:
                return "🚨 建议立即停止！召回率过低，考虑使用 aggressive 策略或增加异常比例"
            elif auc < 0.6:
                return "⚠️  性能较差，建议观察2-3轮，如无改善考虑调整策略"
            elif recall > 0.3 and auc > 0.75:
                return "✅ 开局良好，继续训练"
            else:
                return "📈 正常学习中，继续观察"

        # 中期阶段 (20%-60%)
        elif progress <= 0.6:
            if optimize_target == 'recall':
                if recall < 0.4:
                    return "🔄 召回率提升缓慢，考虑调整学习率或切换到 aggressive 策略"
                elif recall > 0.7:
                    return "🎯 召回率表现优秀，可考虑平衡精确率"
                else:
                    return "📊 召回率稳步提升，继续当前策略"

            elif optimize_target == 'precision':
                if precision < 0.4:
                    return "🎯 精确率较低，考虑降低异常比例或调整阈值"
                elif precision > 0.7:
                    return "✅ 精确率表现优秀，可尝试提升召回率"
                else:
                    return "📊 精确率稳步提升，继续当前策略"

            elif optimize_target == 'f1_score':
                if f1 < 0.4:
                    return "⚖️  F1分数较低，需要平衡召回率和精确率"
                elif f1 > 0.6:
                    return "🎯 F1分数表现良好，继续优化"
                else:
                    return "📈 F1分数稳步提升，保持当前策略"

            # 通用建议
            if auc > 0.85:
                return "🚀 模型性能优秀，可考虑提前结束或微调"
            elif mean_pred < 0.1:
                return "🔧 模型过于保守，考虑增加正样本权重"
            elif mean_pred > 0.8:
                return "🔧 模型过于激进，考虑降低异常比例"
            else:
                return "📊 继续训练，性能稳步提升"

        # 后期阶段 (60%-100%)
        else:
            if auc > 0.90:
                return "🏆 性能已达到优秀水平，建议提前结束训练"
            elif auc > 0.85:
                return "✅ 性能良好，可继续微调或准备结束"
            elif progress > 0.8 and auc < 0.75:
                return "⏰ 训练后期性能仍不理想，建议提前结束并调整策略"
            elif optimize_target == 'recall' and recall < 0.6:
                return "🎯 召回率仍需提升，考虑延长训练或调整策略"
            elif optimize_target == 'precision' and precision < 0.6:
                return "🎯 精确率仍需提升，考虑调整阈值或策略"
            else:
                return "🏁 接近训练结束，准备评估最终性能"

    def _check_intelligent_stopping(self, epoch: int, total_epochs: int, val_metrics: Dict,
                                   pred_stats: Dict, optimize_target: str, no_improve: int,
                                   patience: int) -> tuple[bool, str]:
        """智能停止检查"""
        recall = val_metrics['recall']
        precision = val_metrics['precision']
        auc = val_metrics['auc_roc']
        f1 = val_metrics['f1_score']

        progress = epoch / total_epochs

        # 1. 早期性能过差，立即停止
        if epoch <= 5:
            if recall < 0.05 and auc < 0.55:
                return True, "早期性能过差，建议调整策略重新训练"

        # 2. 性能已达到优秀水平，提前结束
        if auc > 0.92 and epoch >= 10:
            return True, f"性能已达到优秀水平 (AUC: {auc:.3f})，提前结束"

        # 3. 针对优化目标的特殊停止条件
        if optimize_target == 'recall':
            if recall > 0.85 and precision > 0.4:
                return True, f"召回率已达到优秀水平 ({recall:.3f})，建议结束"
        elif optimize_target == 'precision':
            if precision > 0.85 and recall > 0.4:
                return True, f"精确率已达到优秀水平 ({precision:.3f})，建议结束"
        elif optimize_target == 'f1_score':
            if f1 > 0.80:
                return True, f"F1分数已达到优秀水平 ({f1:.3f})，建议结束"

        # 4. 后期性能不佳，提前结束
        if progress > 0.7 and auc < 0.70:
            return True, "训练后期性能仍不理想，建议调整策略"

        # 5. 学习率过低，无法继续改善
        if hasattr(self, '_current_lr') and self._current_lr < 1e-6:
            return True, "学习率过低，模型无法继续改善"

        # 6. 预测行为异常
        if pred_stats:
            mean_pred = pred_stats['mean']
            if mean_pred < 0.01 or mean_pred > 0.99:
                return True, "预测行为异常，模型可能崩溃"

        # 7. 长期无改善且接近结束
        if no_improve >= patience // 2 and progress > 0.8:
            return True, f"长期无改善 ({no_improve}轮) 且接近训练结束"

        return False, ""

    def _save_milestone_models(self, epoch: int, optimizer, val_metrics: Dict):
        """保存里程碑模型"""
        from datetime import datetime

        auc = val_metrics.get('auc_roc', 0)
        recall = val_metrics.get('recall', 0)
        f1 = val_metrics.get('f1_score', 0)
        precision = val_metrics.get('precision', 0)

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': val_metrics,
            'timestamp': datetime.now().isoformat()
        }

        # 保存AUC > 0.93的模型
        if auc > 0.93:
            if not hasattr(self, '_saved_auc_milestones'):
                self._saved_auc_milestones = set()

            auc_key = f"{auc:.3f}"
            if auc_key not in self._saved_auc_milestones:
                milestone_path = os.path.join(self.save_dir, f'milestone_auc_{auc:.3f}_epoch_{epoch+1}.pt')
                try:
                    torch.save(checkpoint, milestone_path)
                    print(f"  🎯 里程碑模型: AUC {auc:.3f} (已保存)")
                    self._saved_auc_milestones.add(auc_key)
                except Exception as e:
                    print(f"  ❌ 保存里程碑模型失败: {e}")

        # 保存召回率 > 0.6的模型
        if recall > 0.6:
            if not hasattr(self, '_saved_recall_milestones'):
                self._saved_recall_milestones = set()

            recall_key = f"{recall:.3f}"
            if recall_key not in self._saved_recall_milestones:
                recall_path = os.path.join(self.save_dir, f'high_recall_{recall:.3f}_epoch_{epoch+1}.pt')
                try:
                    torch.save(checkpoint, recall_path)
                    print(f"  📈 高召回模型: Recall {recall:.3f} (已保存)")
                    self._saved_recall_milestones.add(recall_key)
                except Exception as e:
                    print(f"  ❌ 保存高召回模型失败: {e}")

        # 保存F1 > 0.55的模型
        if f1 > 0.55:
            if not hasattr(self, '_saved_f1_milestones'):
                self._saved_f1_milestones = set()

            f1_key = f"{f1:.3f}"
            if f1_key not in self._saved_f1_milestones:
                f1_path = os.path.join(self.save_dir, f'high_f1_{f1:.3f}_epoch_{epoch+1}.pt')
                try:
                    torch.save(checkpoint, f1_path)
                    print(f"  ⚖️  高F1模型: F1 {f1:.3f} (已保存)")
                    self._saved_f1_milestones.add(f1_key)
                except Exception as e:
                    print(f"  ❌ 保存高F1模型失败: {e}")

        print("-" * 60)

    def _aggressive_emergency_check(self, epoch: int, val_metrics: Dict, optimizer, config: Dict):
        """激进策略的紧急检查"""
        if epoch > 5 and val_metrics['recall'] < 0.3:
            print("\n🚨 执行紧急措施:")

            # 重新初始化分类头
            if hasattr(self.model, 'classifier'):
                for layer in self.model.classifier:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0.1)
                print("    - 重新初始化分类头")

            # 提升学习率
            new_lr = config['learning_rate'] * 3
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"    - 学习率提升到: {new_lr:.2e}")

            # 减少正则化
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = 1e-6
            print("    - 减少权重衰减")

    def _get_batch_data(self, batch):
        """获取批次数据"""
        if len(batch.data) == 2:
            q, s = batch.get("q", "s")
            pid = None
        else:
            q, s, pid = batch.get("q", "s", "pid")

        q = q.to(self.device)
        s = s.to(self.device)
        if pid is not None:
            pid = pid.to(self.device)

        return q, s, pid

    def get_strategy_info(self) -> Dict[str, Any]:
        """获取当前策略信息"""
        return {
            'name': self.strategy.get_strategy_name(),
            'default_params': self.strategy.get_default_params(),
            'description': self._get_strategy_description()
        }

    def _get_strategy_description(self) -> str:
        """获取策略描述"""
        descriptions = {
            'basic': '标准训练策略，适用于大多数场景',
            'enhanced': '增强训练策略，包含Focal Loss、渐进式训练等高级技术',
            'improved': '增强训练策略，包含Focal Loss、渐进式训练等高级技术',  # 向后兼容
            'aggressive': '激进训练策略，专门处理严重类别不平衡问题'
        }
        return descriptions.get(self.strategy_name, '未知策略')

    @staticmethod
    def list_available_strategies() -> Dict[str, str]:
        """列出所有可用策略"""
        return {
            'basic': '基础策略 - 标准训练，适合大多数场景',
            'enhanced': '增强策略 - 高级技术，更好的性能',
            'aggressive': '激进策略 - 处理严重不平衡，高召回率'
        }

    @staticmethod
    def get_strategy_recommendations(data_balance: str = 'balanced') -> str:
        """获取策略推荐"""
        recommendations = {
            'balanced': 'basic',
            'slightly_imbalanced': 'enhanced',
            'severely_imbalanced': 'aggressive'
        }
        return recommendations.get(data_balance, 'basic')

    def _print_optimization_advice(self, optimize_for: str, strategy: str):
        """打印优化目标建议"""
        advice = {
            'recall': {
                'description': '🎯 召回率优化 - 专注于找到所有异常',
                'suitable': '高风险场景：医疗诊断、安全监控、学术诚信',
                'trade_off': '可能产生较多误报，但不会漏掉重要异常',
                'best_strategy': 'aggressive'
            },
            'precision': {
                'description': '🎯 精确率优化 - 专注于避免误报',
                'suitable': '高成本场景：金融反欺诈、垃圾邮件、推荐系统',
                'trade_off': '减少误报，但可能漏掉一些异常',
                'best_strategy': 'enhanced'
            },
            'f1_score': {
                'description': '🎯 F1值优化 - 平衡查全和查准',
                'suitable': '通用场景：业务监控、内容审核、流量分析',
                'trade_off': '在召回率和精确率之间寻求平衡',
                'best_strategy': 'basic'
            },
            'auc_roc': {
                'description': '🎯 AUC优化 - 整体分类性能',
                'suitable': '模型比较：研究、基准测试、算法验证',
                'trade_off': '评估模型区分能力，适合模型选择',
                'best_strategy': 'enhanced'
            }
        }

        if optimize_for in advice:
            info = advice[optimize_for]
            print(f"\n💡 优化目标分析:")
            print(f"  {info['description']}")
            print(f"  📋 适用场景: {info['suitable']}")
            print(f"  ⚖️  权衡考虑: {info['trade_off']}")

            if strategy != info['best_strategy']:
                print(f"  💭 建议: 考虑使用 '{info['best_strategy']}' 策略以获得更好的 {optimize_for} 表现")
            else:
                print(f"  ✅ 策略匹配: '{strategy}' 策略很适合优化 {optimize_for}")

        print()
