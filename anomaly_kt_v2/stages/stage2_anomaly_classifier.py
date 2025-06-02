"""
第二阶段：异常分类器训练

基于课程学习的异常检测器训练。
完全基于DTransformer原始代码，不依赖任何anomaly_kt模块。
"""

import os
import sys
import torch

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from anomaly_kt_v2.anomaly_detection import (
    CausalAnomalyDetector,
    CurriculumTrainer,
    AnomalyDetectionEvaluator
)
from anomaly_kt_v2.core.common import print_stage_header, print_training_summary


def validate_stage2_parameters(args, dataset_config):
    """验证第二阶段参数"""
    print("🔍 验证第二阶段参数...")
    
    # 检查必需的基本参数
    required_basic = ['device', 'output_dir', 'baseline_model_path']
    for param in required_basic:
        if not hasattr(args, param) or getattr(args, param) is None:
            raise ValueError(f"缺少必需的基本参数: {param}")
    
    # 检查必需的模型参数
    required_model = ['d_model', 'n_heads', 'n_layers', 'dropout']
    for param in required_model:
        if not hasattr(args, param):
            raise ValueError(f"缺少必需的模型参数: {param}")
    
    # 检查必需的训练参数
    required_training = ['anomaly_epochs', 'learning_rate', 'patience']
    for param in required_training:
        if not hasattr(args, param):
            raise ValueError(f"缺少必需的训练参数: {param}")
    
    # 检查数据集配置
    required_dataset = ['n_questions', 'n_pid']
    for param in required_dataset:
        if param not in dataset_config:
            raise ValueError(f"数据集配置缺少必需参数: {param}")
    
    # 设置可选参数的默认值
    optional_params = {
        'with_pid': True,
        'window_size': 10,
        'curriculum_type': 'linear',
        'initial_difficulty': 0.1,
        'final_difficulty': 0.8,
        'warmup_epochs': 5
    }
    
    for param, default_value in optional_params.items():
        if not hasattr(args, param):
            setattr(args, param, default_value)
    
    print("✅ 第二阶段参数验证通过")


def print_stage2_parameters(args, dataset_config):
    """打印第二阶段参数配置"""
    print("\n📋 第二阶段参数配置:")
    
    print("  基本参数:")
    print(f"    设备: {args.device}")
    print(f"    输出目录: {args.output_dir}")
    print(f"    基线模型路径: {args.baseline_model_path}")
    print(f"    使用问题ID: {args.with_pid}")
    
    print("  异常检测器参数:")
    print(f"    模型维度: {args.d_model}")
    print(f"    注意力头数: {args.n_heads}")
    print(f"    层数: {args.n_layers}")
    print(f"    Dropout: {args.dropout}")
    print(f"    统计窗口大小: {args.window_size}")
    
    print("  课程学习参数:")
    print(f"    训练轮数: {args.anomaly_epochs}")
    print(f"    学习率: {args.learning_rate}")
    print(f"    早停耐心: {args.patience}")
    print(f"    课程类型: {args.curriculum_type}")
    print(f"    初始难度: {args.initial_difficulty}")
    print(f"    最终难度: {args.final_difficulty}")
    print(f"    预热轮数: {args.warmup_epochs}")
    
    print("  数据集参数:")
    print(f"    问题总数: {dataset_config['n_questions']}")
    print(f"    问题ID总数: {dataset_config['n_pid']}")


def train_anomaly_classifier(args, dataset_config, train_data, val_data):
    """
    训练异常分类器
    
    Args:
        args: 训练参数
        dataset_config: 数据集配置
        train_data: 训练数据加载器
        val_data: 验证数据加载器
        
    Returns:
        str: 训练好的模型路径
    """
    print_stage_header("异常分类器训练", 2)
    
    # 验证参数
    validate_stage2_parameters(args, dataset_config)
    
    # 打印参数配置
    print_stage2_parameters(args, dataset_config)
    
    # 创建异常检测器
    print("\n🔧 创建异常检测器...")
    detector = CausalAnomalyDetector(
        n_questions=dataset_config['n_questions'],
        n_pid=dataset_config['n_pid'] if args.with_pid else 0,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        window_size=args.window_size
    )
    
    # 计算参数数量
    total_params = sum(p.numel() for p in detector.parameters())
    trainable_params = sum(p.numel() for p in detector.parameters() if p.requires_grad)
    
    print(f"✅ 异常检测器创建成功")
    print(f"  参数总数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # 移动到设备
    detector.to(args.device)
    
    # 创建课程学习训练器
    print("\n🚀 创建课程学习训练器...")
    save_dir = os.path.join(args.output_dir, 'anomaly_classifier')
    trainer = CurriculumTrainer(
        model=detector,
        device=args.device,
        learning_rate=args.learning_rate,
        save_dir=save_dir,
        patience=args.patience,
        with_pid=args.with_pid
    )
    
    print(f"✅ 训练器创建成功")
    print(f"  保存目录: {save_dir}")
    
    # 课程学习配置
    curriculum_config = {
        'initial_difficulty': args.initial_difficulty,
        'final_difficulty': args.final_difficulty,
        'schedule_type': args.curriculum_type,
        'warmup_epochs': args.warmup_epochs
    }
    
    print(f"\n📚 课程学习配置:")
    for key, value in curriculum_config.items():
        print(f"  {key}: {value}")
    
    # 开始训练
    print("\n" + "="*60)
    print("开始异常分类器训练...")
    print("="*60)
    
    try:
        training_result = trainer.train(
            train_loader=train_data,
            val_loader=val_data,
            epochs=args.anomaly_epochs,
            curriculum_config=curriculum_config
        )
        
        # 打印训练总结
        print_training_summary("异常分类器", training_result, save_dir)
        
        # 模型路径
        model_path = os.path.join(save_dir, 'best_anomaly_detector.pt')
        print(f"\n💾 异常检测器已保存: {model_path}")
        
        # 评估模型
        print("\n📊 评估异常检测器性能...")
        evaluator = AnomalyDetectionEvaluator()
        
        # 加载最佳模型进行评估
        detector.load_state_dict(torch.load(model_path, weights_only=False)['model_state_dict'])
        
        eval_result = evaluator.evaluate_model(
            model=detector,
            data_loader=val_data,
            device=args.device,
            anomaly_strategies=['random_flip', 'uniform_random', 'systematic_bias'],
            with_pid=args.with_pid
        )
        
        # 生成评估报告
        report_path = os.path.join(save_dir, 'evaluation_report.txt')
        report = evaluator.generate_report(eval_result, report_path)
        print(f"\n📋 评估报告已保存: {report_path}")
        
        # 打印简要评估结果
        if 'overall' in eval_result:
            overall = eval_result['overall']
            print(f"\n📈 异常检测器性能:")
            print(f"  平均AUC: {overall.get('avg_auc', 0):.4f}")
            print(f"  性能等级: {overall.get('performance_grade', 'Unknown')}")
        
        return model_path
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def load_baseline_model_info(baseline_model_path: str) -> dict:
    """
    加载基线模型信息
    
    Args:
        baseline_model_path: 基线模型路径
        
    Returns:
        dict: 基线模型信息
    """
    if not os.path.exists(baseline_model_path):
        raise FileNotFoundError(f"基线模型文件不存在: {baseline_model_path}")
    
    # PyTorch 2.6+ 兼容性：禁用 weights_only 以支持旧模型文件
    checkpoint = torch.load(baseline_model_path, map_location='cpu', weights_only=False)
    
    # 提取模型配置信息
    if 'args' in checkpoint:
        baseline_args = checkpoint['args']
        model_info = {
            'd_model': getattr(baseline_args, 'd_model', 128),
            'n_heads': getattr(baseline_args, 'n_heads', 8),
            'n_layers': getattr(baseline_args, 'n_layers', 3),
            'dropout': getattr(baseline_args, 'dropout', 0.2),
            'with_pid': getattr(baseline_args, 'with_pid', True),
            'baseline_auc': checkpoint.get('auc', 0.0),
            'baseline_epoch': checkpoint.get('epoch', 0)
        }
    else:
        # 默认配置
        model_info = {
            'd_model': 128,
            'n_heads': 8,
            'n_layers': 3,
            'dropout': 0.2,
            'with_pid': True,
            'baseline_auc': checkpoint.get('auc', 0.0),
            'baseline_epoch': checkpoint.get('epoch', 0)
        }
    
    print(f"📄 基线模型信息:")
    print(f"  模型维度: {model_info['d_model']}")
    print(f"  注意力头数: {model_info['n_heads']}")
    print(f"  基线AUC: {model_info['baseline_auc']:.4f}")
    print(f"  训练轮数: {model_info['baseline_epoch']}")
    
    return model_info
