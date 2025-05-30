"""
Stage 1: Baseline DTransformer Model Training

从 full_pipeline.py 中提取的第一阶段代码，训练基线DTransformer模型
"""

import os
import sys
import torch

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from DTransformer.model import DTransformer
from anomaly_kt.trainer import KnowledgeTracingTrainer


def validate_stage1_parameters(args, dataset_config):
    """验证第一阶段所需的参数

    Args:
        args: 命令行参数对象
        dataset_config: 数据集配置字典

    Raises:
        ValueError: 如果缺少必需参数
    """
    # 检查基本参数
    required_basic_params = ['device', 'output_dir']
    for param in required_basic_params:
        if not hasattr(args, param) or getattr(args, param) is None:
            raise ValueError(f"Missing required parameter: {param}")

    # 检查模型参数
    required_model_params = [
        'd_model', 'n_heads', 'n_know', 'n_layers',
        'dropout', 'lambda_cl', 'window'
    ]
    for param in required_model_params:
        if not hasattr(args, param):
            raise ValueError(f"Missing required model parameter: {param}")

    # 检查训练参数
    required_training_params = ['kt_epochs', 'learning_rate', 'patience']
    for param in required_training_params:
        if not hasattr(args, param):
            raise ValueError(f"Missing required training parameter: {param}")

    # 检查数据集配置
    required_dataset_params = ['n_questions', 'n_pid']
    for param in required_dataset_params:
        if param not in dataset_config:
            raise ValueError(f"Missing required dataset parameter: {param}")

    # 检查可选参数，设置默认值
    if not hasattr(args, 'with_pid'):
        args.with_pid = False
    if not hasattr(args, 'proj'):
        args.proj = False
    if not hasattr(args, 'hard_neg'):
        args.hard_neg = False
    if not hasattr(args, 'use_cl'):
        args.use_cl = False

    print("✓ 第一阶段参数验证通过")


def print_stage1_parameters(args, dataset_config):
    """打印第一阶段参数信息"""
    print("\n📋 第一阶段参数配置:")
    print("  基本参数:")
    print(f"    设备: {args.device}")
    print(f"    输出目录: {args.output_dir}")
    print(f"    使用问题ID: {getattr(args, 'with_pid', False)}")

    print("  模型参数:")
    print(f"    模型维度: {args.d_model}")
    print(f"    注意力头数: {args.n_heads}")
    print(f"    知识概念数: {args.n_know}")
    print(f"    层数: {args.n_layers}")
    print(f"    Dropout: {args.dropout}")
    print(f"    对比学习权重: {args.lambda_cl}")
    print(f"    使用投影: {getattr(args, 'proj', False)}")
    print(f"    困难负样本: {getattr(args, 'hard_neg', False)}")
    print(f"    窗口大小: {args.window}")

    print("  训练参数:")
    print(f"    训练轮数: {args.kt_epochs}")
    print(f"    学习率: {args.learning_rate}")
    print(f"    早停耐心: {args.patience}")
    print(f"    使用对比学习: {getattr(args, 'use_cl', False)}")

    print("  数据集参数:")
    print(f"    问题总数: {dataset_config['n_questions']}")
    print(f"    问题ID总数: {dataset_config['n_pid']}")


def train_baseline_model(args, dataset_config, train_data, val_data):
    """训练基线DTransformer模型

    这是从 full_pipeline.py 中直接提取的第一阶段代码

    Args:
        args: 命令行参数，包含模型和训练配置
        dataset_config: 数据集配置，包含 n_questions, n_pid 等
        train_data: 训练数据加载器
        val_data: 验证数据加载器

    Returns:
        str: 训练好的模型文件路径
    """
    # 验证参数
    validate_stage1_parameters(args, dataset_config)

    print("\n" + "="*60)
    print("PHASE 1: Training Baseline DTransformer")
    print("="*60)

    # 打印参数信息
    print_stage1_parameters(args, dataset_config)

    # 创建模型
    model = DTransformer(
        n_questions=dataset_config['n_questions'],
        n_pid=dataset_config['n_pid'] if args.with_pid else 0,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_know=args.n_know,
        n_layers=args.n_layers,
        dropout=args.dropout,
        lambda_cl=args.lambda_cl,
        proj=args.proj,
        hard_neg=args.hard_neg,
        window=args.window
    )

    # 训练器
    trainer = KnowledgeTracingTrainer(
        model=model,
        device=args.device,
        save_dir=os.path.join(args.output_dir, 'baseline'),
        patience=args.patience
    )

    # 训练
    baseline_metrics = trainer.train(
        train_loader=train_data,
        val_loader=val_data,
        epochs=args.kt_epochs,
        learning_rate=args.learning_rate,
        use_cl=args.use_cl
    )

    print(f"\nBaseline training completed!")
    print(f"Best AUC: {baseline_metrics['auc']:.4f}")

    return os.path.join(args.output_dir, 'baseline', 'best_model.pt')
