"""
Stage 2: Curriculum Learning Anomaly Detection Training

基于课程学习的异常检测训练模块，实现我们设计的创新方法
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from DTransformer.model import DTransformer
from anomaly_kt.curriculum_learning import (
    CurriculumScheduler,
    CurriculumAnomalyGenerator,
    DifficultyEstimator,
    BaselineAnomalyGenerator
)
from anomaly_kt.detector import CausalAnomalyDetector
from anomaly_kt.trainer import AnomalyDetectorTrainer
from anomaly_kt.curriculum_learning.curriculum_trainer import CurriculumTrainer


def validate_stage2_parameters(args, dataset_config):
    """验证第二阶段所需的参数
    
    Args:
        args: 命令行参数对象
        dataset_config: 数据集配置字典
        
    Raises:
        ValueError: 如果缺少必需参数
    """
    # 检查基本参数
    required_basic_params = ['device', 'output_dir', 'baseline_model_path']
    for param in required_basic_params:
        if not hasattr(args, param) or getattr(args, param) is None:
            raise ValueError(f"Missing required parameter: {param}")
    
    # 检查基线模型文件是否存在
    if not os.path.exists(args.baseline_model_path):
        raise ValueError(f"Baseline model file not found: {args.baseline_model_path}")
    
    # 检查课程学习参数
    required_curriculum_params = [
        'curriculum_strategy', 'curriculum_epochs', 'anomaly_ratio'
    ]
    for param in required_curriculum_params:
        if not hasattr(args, param):
            raise ValueError(f"Missing required curriculum parameter: {param}")
    
    # 检查异常检测器参数
    required_detector_params = [
        'detector_hidden_dim', 'detector_num_layers', 'detector_num_heads', 'detector_dropout'
    ]
    for param in required_detector_params:
        if not hasattr(args, param):
            raise ValueError(f"Missing required detector parameter: {param}")
    
    # 检查训练参数
    required_training_params = ['learning_rate', 'patience']
    for param in required_training_params:
        if not hasattr(args, param):
            raise ValueError(f"Missing required training parameter: {param}")
    
    # 设置默认值
    if not hasattr(args, 'baseline_ratio'):
        args.baseline_ratio = 0.05
    if not hasattr(args, 'max_patience'):
        args.max_patience = 5
    if not hasattr(args, 'difficulty_estimation'):
        args.difficulty_estimation = True
    
    print("✓ 第二阶段参数验证通过")


def print_stage2_parameters(args, dataset_config):
    """打印第二阶段参数信息"""
    print("\n📋 第二阶段参数配置:")
    print("  基本参数:")
    print(f"    设备: {args.device}")
    print(f"    输出目录: {args.output_dir}")
    print(f"    基线模型: {args.baseline_model_path}")
    
    print("  课程学习参数:")
    print(f"    调度策略: {args.curriculum_strategy}")
    print(f"    训练轮数: {args.curriculum_epochs}")
    print(f"    异常比例: {args.anomaly_ratio}")
    print(f"    基线异常比例: {args.baseline_ratio}")
    print(f"    最大耐心值: {args.max_patience}")
    
    print("  异常检测器参数:")
    print(f"    隐藏层维度: {args.detector_hidden_dim}")
    print(f"    层数: {args.detector_num_layers}")
    print(f"    注意力头数: {args.detector_num_heads}")
    print(f"    Dropout: {args.detector_dropout}")
    
    print("  训练参数:")
    print(f"    学习率: {args.learning_rate}")
    print(f"    早停耐心: {args.patience}")
    print(f"    难度评估: {args.difficulty_estimation}")

    print("  数据集参数:")
    print(f"    问题总数: {dataset_config['n_questions']}")
    print(f"    问题ID总数: {dataset_config['n_pid']}")


def load_baseline_model(model_path: str, dataset_config: Dict, args) -> DTransformer:
    """加载基线模型

    Args:
        model_path: 基线模型文件路径
        dataset_config: 数据集配置
        args: 命令行参数

    Returns:
        加载的基线模型
    """
    print(f"\n📥 加载基线模型: {model_path}")

    # 从第一阶段的配置文件中读取正确的模型参数
    stage1_config_path = os.path.join(os.path.dirname(model_path), '..', 'config.yaml')
    stage1_config = {}

    if os.path.exists(stage1_config_path):
        import yaml
        with open(stage1_config_path, 'r') as f:
            stage1_config = yaml.safe_load(f)
        print(f"✓ 从第一阶段配置文件读取参数: {stage1_config_path}")
    else:
        print(f"⚠️ 未找到第一阶段配置文件: {stage1_config_path}")

    # 创建模型实例（使用第一阶段的实际参数）
    model = DTransformer(
        n_questions=dataset_config['n_questions'],
        n_pid=dataset_config['n_pid'] if stage1_config.get('with_pid', False) else 0,
        d_model=stage1_config.get('d_model', 256),
        n_heads=stage1_config.get('n_heads', 16),
        n_know=stage1_config.get('n_know', 16),
        n_layers=stage1_config.get('n_layers', 3),
        dropout=stage1_config.get('dropout', 0.2),
        lambda_cl=stage1_config.get('lambda_cl', 0.1),
        proj=stage1_config.get('proj', True),
        hard_neg=stage1_config.get('hard_neg', False),
        window=stage1_config.get('window', 1)
    )

    print(f"✓ 基线模型参数:")
    print(f"  d_model: {stage1_config.get('d_model', 256)}")
    print(f"  n_heads: {stage1_config.get('n_heads', 16)}")
    print(f"  n_layers: {stage1_config.get('n_layers', 3)}")
    print(f"  dropout: {stage1_config.get('dropout', 0.2)}")
    print(f"  with_pid: {stage1_config.get('with_pid', False)}")
    
    # 加载模型权重（兼容PyTorch 2.6+）
    checkpoint = torch.load(model_path, map_location=args.device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ 模型权重加载成功 (来自checkpoint)")
        if 'auc' in checkpoint:
            print(f"  基线模型AUC: {checkpoint['auc']:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print(f"✓ 模型权重加载成功 (直接加载)")
    
    model.to(args.device)
    model.eval()  # 基线模型用于特征提取，设为评估模式
    
    return model


def create_anomaly_detector(dataset_config: Dict, baseline_model: DTransformer, args) -> CausalAnomalyDetector:
    """创建异常检测器

    Args:
        dataset_config: 数据集配置
        baseline_model: 基线模型
        args: 命令行参数

    Returns:
        异常检测器实例
    """
    print("\n🔍 创建异常检测器...")

    # 从基线模型获取实际的模型参数
    d_model = baseline_model.q_embed.embedding_dim  # 从嵌入层获取d_model
    n_heads = baseline_model.n_heads  # 直接获取n_heads

    # 使用与基线模型一致的参数（确保兼容性）
    detector = CausalAnomalyDetector(
        n_questions=dataset_config['n_questions'],
        n_pid=dataset_config.get('n_pid', 0),
        d_model=d_model,  # 使用基线模型的实际维度
        n_heads=n_heads,  # 使用基线模型的实际头数
        n_layers=getattr(args, 'detector_num_layers', 3),
        dropout=getattr(args, 'detector_dropout', 0.2),
        window_size=getattr(args, 'detector_window_size', 10)
    )
    
    detector.to(args.device)
    
    print(f"✓ 异常检测器创建成功")
    print(f"  问题总数: {dataset_config['n_questions']}")
    print(f"  问题ID总数: {dataset_config.get('n_pid', 0)}")
    print(f"  隐藏维度: {d_model} (与基线模型一致)")
    print(f"  注意力头数: {n_heads} (与基线模型一致)")
    print(f"  层数: {getattr(args, 'detector_num_layers', 3)}")
    print(f"  Dropout: {getattr(args, 'detector_dropout', 0.2)}")
    
    return detector


def train_curriculum_anomaly_detector(args, dataset_config, train_data, val_data) -> str:
    """训练课程学习异常检测器
    
    这是第二阶段的核心函数，实现基于课程学习的异常检测训练
    
    Args:
        args: 命令行参数，包含课程学习和异常检测配置
        dataset_config: 数据集配置，包含 n_questions, n_pid 等
        train_data: 训练数据加载器
        val_data: 验证数据加载器
        
    Returns:
        str: 训练好的异常检测器文件路径
    """
    # 验证参数
    validate_stage2_parameters(args, dataset_config)
    
    print("\n" + "="*60)
    print("PHASE 2: Curriculum Learning Anomaly Detection Training")
    print("="*60)
    
    # 打印参数信息
    print_stage2_parameters(args, dataset_config)
    
    # 1. 加载基线模型
    baseline_model = load_baseline_model(args.baseline_model_path, dataset_config, args)
    
    # 2. 创建异常检测器（使用与基线模型一致的参数）
    anomaly_detector = create_anomaly_detector(dataset_config, baseline_model, args)
    
    # 3. 初始化课程学习组件
    print("\n🎓 初始化课程学习组件...")
    
    scheduler = CurriculumScheduler(
        strategy=args.curriculum_strategy,
        dataset_name=args.dataset,
        total_epochs=args.curriculum_epochs
    )
    
    curriculum_generator = CurriculumAnomalyGenerator(args.dataset)
    
    difficulty_estimator = DifficultyEstimator(args.dataset) if args.difficulty_estimation else None
    
    baseline_generator = BaselineAnomalyGenerator()
    
    print(f"✓ 课程调度器: {args.curriculum_strategy} 策略")
    print(f"✓ 异常生成器: {args.dataset} 数据集优化")
    print(f"✓ 难度评估器: {'启用' if args.difficulty_estimation else '禁用'}")
    
    # 4. 创建课程学习训练器
    print("\n🚀 创建课程学习训练器...")

    # 先创建基础训练器
    base_trainer = AnomalyDetectorTrainer(
        model=anomaly_detector,
        device=args.device,
        save_dir=os.path.join(args.output_dir, 'curriculum_anomaly'),
        patience=args.patience
    )

    # 创建课程学习训练器
    trainer = CurriculumTrainer(
        base_trainer=base_trainer,
        dataset_name=args.dataset,
        strategy=args.curriculum_strategy,
        total_epochs=args.curriculum_epochs,
        output_dir=os.path.join(args.output_dir, 'curriculum_learning')
    )
    
    print(f"✓ 训练器创建成功")
    print(f"  保存目录: {os.path.join(args.output_dir, 'curriculum_anomaly')}")

    # 5. 开始课程学习训练
    print("\n" + "="*60)
    print("开始课程学习训练...")
    print("="*60)

    # 创建优化器和损失函数
    import torch.optim as optim
    optimizer = optim.Adam(anomaly_detector.parameters(), lr=args.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()

    # 使用课程学习训练方法
    training_metrics = trainer.train_with_curriculum(
        train_loader=train_data,
        val_loader=val_data,
        model=anomaly_detector,
        optimizer=optimizer,
        criterion=criterion,
        epochs=args.curriculum_epochs
    )

    print(f"\nCurriculum learning training completed!")
    print(f"Best AUC: {training_metrics.get('auc', 0.0):.4f}")
    print(f"Best F1: {training_metrics.get('f1', 0.0):.4f}")
    
    # 7. 保存训练总结
    save_training_summary(args, training_metrics, scheduler)
    
    return os.path.join(args.output_dir, 'curriculum_anomaly', 'best_model.pt')


def save_training_summary(args, metrics: Dict, scheduler: CurriculumScheduler):
    """保存训练总结"""
    import json
    from datetime import datetime
    
    summary = {
        'experiment_info': {
            'dataset': args.dataset,
            'strategy': args.curriculum_strategy,
            'timestamp': datetime.now().isoformat(),
            'baseline_model': args.baseline_model_path
        },
        'curriculum_config': {
            'total_epochs': args.curriculum_epochs,
            'anomaly_ratio': args.anomaly_ratio,
            'baseline_ratio': args.baseline_ratio,
            'max_patience': args.max_patience
        },
        'final_metrics': metrics,
        'curriculum_summary': scheduler.get_schedule_summary(),
        'phase_transitions': scheduler.phase_history
    }
    
    summary_path = os.path.join(args.output_dir, 'curriculum_training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n📊 训练总结已保存: {summary_path}")


def test_curriculum_components(args, dataset_config):
    """测试课程学习组件（用于调试）"""
    print("\n🧪 测试课程学习组件...")
    
    # 创建模拟数据
    batch_size, seq_len = 4, 20
    q = torch.randint(0, dataset_config['n_questions'], (batch_size, seq_len))
    s = torch.randint(0, 2, (batch_size, seq_len))
    
    # 测试基线异常生成器
    baseline_gen = BaselineAnomalyGenerator()
    s_baseline, labels_baseline = baseline_gen.generate_baseline_anomalies(
        q, s, strategy='random_flip', anomaly_ratio=0.2
    )
    print(f"✓ 基线异常生成: {labels_baseline.sum().item()} 个异常")
    
    # 测试课程异常生成器
    curriculum_gen = CurriculumAnomalyGenerator(args.dataset)
    s_curr, labels_curr, diff_curr = curriculum_gen.generate_curriculum_anomalies(
        q, s,
        difficulty_levels=[1, 2],
        level_weights={1: 0.7, 2: 0.3},
        anomaly_ratio=0.1
    )
    print(f"✓ 课程异常生成: {labels_curr.sum().item()} 个异常")
    
    # 测试调度器
    scheduler = CurriculumScheduler(args.curriculum_strategy, args.dataset, 10)
    schedule_info = scheduler.update(0, {'auc': 0.7, 'f1': 0.65})
    print(f"✓ 课程调度器: Phase {schedule_info['current_phase']}")
    
    print("✅ 所有组件测试通过!")
    return True
