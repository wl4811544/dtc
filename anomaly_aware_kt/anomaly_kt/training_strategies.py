"""
异常检测器训练策略模块

提供三种不同的训练策略：Basic, Enhanced, Aggressive
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
from abc import ABC, abstractmethod
from .trainer import BaseTrainer
from .generator import AnomalyGenerator
from .enhanced_generator import EnhancedAnomalyGenerator
from .evaluator import AnomalyEvaluator


class TrainingStrategy(ABC):
    """训练策略抽象基类"""

    def __init__(self, model: nn.Module, device: str, save_dir: str, patience: int):
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        self.patience = patience
        self.generator = AnomalyGenerator()
        self.evaluator = AnomalyEvaluator()

    @abstractmethod
    def get_strategy_name(self) -> str:
        """获取策略名称"""
        pass

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """获取默认参数"""
        pass

    @abstractmethod
    def create_optimizer(self, learning_rate: float) -> torch.optim.Optimizer:
        """创建优化器"""
        pass

    @abstractmethod
    def create_scheduler(self, optimizer: torch.optim.Optimizer, **kwargs) -> Optional[Any]:
        """创建学习率调度器"""
        pass

    @abstractmethod
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs) -> torch.Tensor:
        """计算损失"""
        pass

    @abstractmethod
    def train_epoch(self, train_loader, optimizer, scheduler, epoch: int, **kwargs) -> Dict[str, float]:
        """训练一个epoch"""
        pass


class BasicStrategy(TrainingStrategy):
    """基础训练策略"""

    def get_strategy_name(self) -> str:
        return "Basic"

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'learning_rate': 1e-3,
            'epochs': 30,
            'anomaly_ratio': 0.1,
            'optimize_for': 'f1_score',
            'patience': 10,
            'weight_decay': 1e-5,
            'gradient_clip': 1.0
        }

    def create_optimizer(self, learning_rate: float) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )

    def create_scheduler(self, optimizer: torch.optim.Optimizer, **kwargs) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs) -> torch.Tensor:
        # 使用模型内置损失函数
        return self.model.get_loss(kwargs['q'], kwargs['s_anomaly'], labels, kwargs.get('pid'))

    def train_epoch(self, train_loader, optimizer, scheduler, epoch: int, **kwargs) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        n_batches = 0
        anomaly_ratio = kwargs.get('anomaly_ratio', 0.1)

        # 添加进度条
        from tqdm import tqdm
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)

        for batch in progress_bar:
            q, s, pid = self._get_batch_data(batch)

            # 生成异常
            s_anomaly, labels = self.generator.generate_anomalies(q, s, anomaly_ratio=anomaly_ratio)
            labels = labels.to(self.device)

            # 计算损失
            loss = self.compute_loss(None, labels, q=q, s_anomaly=s_anomaly, pid=pid)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg_Loss': f'{total_loss/n_batches:.4f}'
            })

        return {'loss': total_loss / n_batches}

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


class EnhancedStrategy(TrainingStrategy):
    """增强训练策略"""

    def __init__(self, model: nn.Module, device: str, save_dir: str, patience: int):
        super().__init__(model, device, save_dir, patience)
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
        self.best_threshold = 0.5
        # 使用增强的异常生成器
        self.generator = EnhancedAnomalyGenerator()
        # 关闭调试模式
        self.generator._debug_mode = False

    def get_strategy_name(self) -> str:
        return "Enhanced"

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'learning_rate': 5e-4,
            'epochs': 50,
            'anomaly_ratio': 0.3,
            'optimize_for': 'recall',
            'patience': 15,
            'weight_decay': 1e-4,
            'gradient_clip': 1.0,
            'use_focal_loss': True,
            'use_class_weights': True,
            'use_progressive_training': True,
            'gradient_accumulation_steps': 2,
            'warmup_epochs': 5
        }

    def create_optimizer(self, learning_rate: float) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )

    def create_scheduler(self, optimizer: torch.optim.Optimizer, **kwargs) -> torch.optim.lr_scheduler.CosineAnnealingLR:
        total_epochs = kwargs.get('total_epochs', 50)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=1e-6
        )

    def focal_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Focal Loss实现"""
        ce_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs) -> torch.Tensor:
        use_focal_loss = kwargs.get('use_focal_loss', True)

        if logits is None:
            # 获取模型输出
            logits = self.model(kwargs['q'], kwargs['s_anomaly'], kwargs.get('pid'))

        if logits.dim() > 1:
            logits = logits.squeeze(-1)

        if use_focal_loss:
            return self.focal_loss(logits, labels)
        else:
            return F.binary_cross_entropy_with_logits(logits, labels.float())

    def get_dynamic_anomaly_ratio(self, epoch: int, total_epochs: int, base_ratio: float) -> float:
        """动态调整异常比例"""
        if epoch < 5:
            return base_ratio
        elif epoch < total_epochs // 2:
            return min(base_ratio * 1.5, 0.4)
        else:
            progress = (epoch - total_epochs // 2) / (total_epochs - total_epochs // 2)
            return base_ratio * 1.5 * (1 - progress) + base_ratio * progress

    def train_epoch(self, train_loader, optimizer, scheduler, epoch: int, **kwargs) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        n_batches = 0

        base_anomaly_ratio = kwargs.get('anomaly_ratio', 0.3)
        total_epochs = kwargs.get('total_epochs', 50)
        use_progressive = kwargs.get('use_progressive_training', True)

        # 动态异常比例
        if use_progressive:
            current_ratio = self.get_dynamic_anomaly_ratio(epoch, total_epochs, base_anomaly_ratio)
        else:
            current_ratio = base_anomaly_ratio

        # 添加进度条
        from tqdm import tqdm
        progress_bar = tqdm(train_loader, desc=f"Enhanced Training Epoch {epoch+1}", leave=False)

        for batch in progress_bar:
            q, s, pid = self._get_batch_data(batch)

            # 使用增强的异常生成
            s_anomaly, labels = self.generator.generate_anomalies_enhanced(
                q, s,
                anomaly_ratio=current_ratio,
                min_anomaly_density=0.3,
                progressive_difficulty=True,
                epoch=epoch
            )
            labels = labels.to(self.device)

            # 计算损失
            loss = self.compute_loss(
                None, labels,
                q=q, s_anomaly=s_anomaly, pid=pid,
                use_focal_loss=kwargs.get('use_focal_loss', True)
            )

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg_Loss': f'{total_loss/n_batches:.4f}',
                'Anomaly_Ratio': f'{current_ratio:.2%}'
            })

        return {
            'loss': total_loss / n_batches,
            'anomaly_ratio': current_ratio
        }

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


class AggressiveStrategy(TrainingStrategy):
    """激进训练策略"""

    def __init__(self, model: nn.Module, device: str, save_dir: str, patience: int):
        super().__init__(model, device, save_dir, patience)
        # 使用增强的异常生成器
        self.generator = EnhancedAnomalyGenerator()
        # 关闭调试模式
        self.generator._debug_mode = False

    def get_strategy_name(self) -> str:
        return "Aggressive"

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'learning_rate': 2e-3,  # 2x基础学习率
            'epochs': 30,
            'anomaly_ratio': 0.3,
            'min_anomaly_ratio': 0.2,
            'max_anomaly_ratio': 0.5,
            'optimize_for': 'recall',
            'patience': 8,
            'weight_decay': 1e-5,
            'gradient_clip': 0.5,
            'force_balance': True,
            'extreme_weights': True,
            'pos_weight': 10.0
        }

    def create_optimizer(self, learning_rate: float) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-5
        )

    def create_scheduler(self, optimizer: torch.optim.Optimizer, **kwargs) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, threshold=0.01
        )

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs) -> torch.Tensor:
        pos_weight = kwargs.get('pos_weight', 10.0)
        pos_weight_tensor = torch.tensor([pos_weight]).to(self.device)

        if logits is None:
            logits = self.model(kwargs['q'], kwargs['s_anomaly'], kwargs.get('pid'))

        if logits.dim() > 1:
            logits = logits.squeeze(-1)

        return F.binary_cross_entropy_with_logits(
            logits, labels.float(), pos_weight=pos_weight_tensor
        )

    def get_dynamic_anomaly_ratio(self, epoch: int, min_ratio: float, max_ratio: float) -> float:
        """激进的动态异常比例"""
        if epoch < 5:
            return max_ratio  # 50%
        elif epoch < 10:
            return 0.4
        elif epoch < 15:
            return 0.3
        else:
            # 线性降低到最小比例
            progress = (epoch - 15) / max(1, 30 - 15)  # 假设总共30轮
            return max(min_ratio, 0.3 * (1 - progress) + min_ratio * progress)

    def force_batch_balance(self, s_anomaly: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """强制批次平衡"""
        batch_size, seq_len = labels.shape
        positive_ratio = labels.float().mean()

        if positive_ratio < 0.3:
            # 随机选择位置强制设为异常
            mask = torch.rand_like(labels.float()) < 0.2
            mask = mask & (labels == 0)

            # 在这些位置创建异常
            for b in range(batch_size):
                for t in range(seq_len):
                    if mask[b, t]:
                        s_anomaly[b, t] = 1 - s_anomaly[b, t]
                        labels[b, t] = 1

        return s_anomaly, labels

    def train_epoch(self, train_loader, optimizer, scheduler, epoch: int, **kwargs) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        total_positive = 0
        total_samples = 0
        n_batches = 0

        min_ratio = kwargs.get('min_anomaly_ratio', 0.2)
        max_ratio = kwargs.get('max_anomaly_ratio', 0.5)
        force_balance = kwargs.get('force_balance', True)

        current_ratio = self.get_dynamic_anomaly_ratio(epoch, min_ratio, max_ratio)

        # 添加进度条
        from tqdm import tqdm
        progress_bar = tqdm(train_loader, desc=f"Aggressive Training Epoch {epoch+1}", leave=False)

        for batch in progress_bar:
            q, s, pid = self._get_batch_data(batch)

            # 使用增强的异常生成（激进模式）
            s_anomaly, labels = self.generator.generate_anomalies_enhanced(
                q, s,
                anomaly_ratio=current_ratio,
                min_anomaly_density=0.4,  # 更高的异常密度
                progressive_difficulty=True,
                epoch=epoch
            )
            labels = labels.to(self.device)

            # 强制批次平衡
            if force_balance:
                s_anomaly, labels = self.force_batch_balance(s_anomaly, labels)

            # 计算损失
            loss = self.compute_loss(
                None, labels,
                q=q, s_anomaly=s_anomaly, pid=pid,
                pos_weight=kwargs.get('pos_weight', 10.0)
            )

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            optimizer.step()

            # 统计
            total_loss += loss.item()
            total_positive += labels.sum().item()
            total_samples += labels.numel()
            n_batches += 1

            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg_Loss': f'{total_loss/n_batches:.4f}',
                'Pos_Ratio': f'{total_positive/total_samples:.2%}',
                'Anomaly_Ratio': f'{current_ratio:.2%}'
            })

        return {
            'loss': total_loss / n_batches,
            'positive_ratio': total_positive / total_samples,
            'anomaly_ratio': current_ratio
        }

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


class StrategyFactory:
    """训练策略工厂类"""

    _strategies = {
        'basic': BasicStrategy,
        'enhanced': EnhancedStrategy,
        'improved': EnhancedStrategy,  # 向后兼容
        'aggressive': AggressiveStrategy
    }

    @classmethod
    def create_strategy(cls, strategy_name: str, model: nn.Module, device: str,
                       save_dir: str, patience: int) -> TrainingStrategy:
        """创建训练策略实例"""
        strategy_name = strategy_name.lower()
        if strategy_name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(cls._strategies.keys())}")

        return cls._strategies[strategy_name](model, device, save_dir, patience)

    @classmethod
    def get_available_strategies(cls) -> list:
        """获取可用策略列表"""
        return list(cls._strategies.keys())

    @classmethod
    def get_strategy_info(cls, strategy_name: str) -> Dict[str, Any]:
        """获取策略信息"""
        strategy_name = strategy_name.lower()
        if strategy_name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        # 创建临时实例获取信息
        temp_model = nn.Linear(1, 1)  # 临时模型
        strategy = cls._strategies[strategy_name](temp_model, 'cpu', '/tmp', 10)

        return {
            'name': strategy.get_strategy_name(),
            'default_params': strategy.get_default_params()
        }
