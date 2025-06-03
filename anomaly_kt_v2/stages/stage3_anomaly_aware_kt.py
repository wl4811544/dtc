"""
第三阶段：异常感知知识追踪

将第一阶段的基线模型和第二阶段的异常检测器融合，
实现异常感知的知识追踪，目标提升AUC 0.05-0.1。
"""

import os
import sys
import torch

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from anomaly_kt_v2.anomaly_aware import AnomalyAwareKT, AnomalyAwareTrainer
from anomaly_kt_v2.anomaly_detection import CausalAnomalyDetector
from anomaly_kt_v2.core.common import print_stage_header, print_training_summary
from DTransformer.model import DTransformer


def validate_stage3_parameters(args, dataset_config):
    """验证第三阶段参数"""
    print("🔍 验证第三阶段参数...")
    
    # 检查必需的基本参数
    required_basic = ['device', 'output_dir', 'baseline_model_path', 'anomaly_detector_path']
    for param in required_basic:
        if not hasattr(args, param) or getattr(args, param) is None:
            raise ValueError(f"缺少必需的基本参数: {param}")
    
    # 检查模型文件是否存在
    if not os.path.exists(args.baseline_model_path):
        raise FileNotFoundError(f"基线模型文件不存在: {args.baseline_model_path}")
    
    if not os.path.exists(args.anomaly_detector_path):
        raise FileNotFoundError(f"异常检测器文件不存在: {args.anomaly_detector_path}")
    
    # 检查必需的训练参数
    required_training = ['fusion_epochs', 'joint_epochs', 'finetune_epochs', 'learning_rate', 'patience']
    for param in required_training:
        if not hasattr(args, param):
            raise ValueError(f"缺少必需的训练参数: {param}")
    
    # 设置可选参数的默认值
    optional_params = {
        'with_pid': True,
        'fusion_type': 'attention',
        'enable_context_enhancement': True,
        'lambda_anomaly': 0.1,
        'freeze_pretrained': True
    }
    
    for param, default_value in optional_params.items():
        if not hasattr(args, param):
            setattr(args, param, default_value)
    
    print("✅ 第三阶段参数验证通过")


def print_stage3_parameters(args, dataset_config):
    """打印第三阶段参数配置"""
    print("\n📋 第三阶段参数配置:")
    
    print("  基本参数:")
    print(f"    设备: {args.device}")
    print(f"    输出目录: {args.output_dir}")
    print(f"    基线模型: {args.baseline_model_path}")
    print(f"    异常检测器: {args.anomaly_detector_path}")
    print(f"    使用问题ID: {args.with_pid}")
    
    print("  融合参数:")
    print(f"    融合类型: {args.fusion_type}")
    print(f"    上下文增强: {args.enable_context_enhancement}")
    print(f"    冻结预训练模型: {args.freeze_pretrained}")
    print(f"    异常损失权重: {args.lambda_anomaly}")
    
    print("  渐进式训练参数:")
    print(f"    融合层训练轮数: {args.fusion_epochs}")
    print(f"    联合训练轮数: {args.joint_epochs}")
    print(f"    端到端微调轮数: {args.finetune_epochs}")
    print(f"    学习率: {args.learning_rate}")
    print(f"    早停耐心: {args.patience}")
    
    print("  数据集参数:")
    print(f"    问题总数: {dataset_config['n_questions']}")
    print(f"    问题ID总数: {dataset_config['n_pid']}")


def load_pretrained_models(args, dataset_config):
    """加载预训练模型"""
    print("\n📄 加载预训练模型...")
    
    # 加载基线模型
    print("🔧 加载基线知识追踪模型...")
    baseline_checkpoint = torch.load(args.baseline_model_path, map_location='cpu', weights_only=False)
    
    # 从checkpoint中获取模型配置
    if 'args' in baseline_checkpoint:
        baseline_args = baseline_checkpoint['args']
        d_model = getattr(baseline_args, 'd_model', 128)
        n_heads = getattr(baseline_args, 'n_heads', 8)
        n_layers = getattr(baseline_args, 'n_layers', 3)
        dropout = getattr(baseline_args, 'dropout', 0.2)
    else:
        # 默认配置
        d_model = 128
        n_heads = 8
        n_layers = 3
        dropout = 0.2
    
    # 创建基线模型
    baseline_model = DTransformer(
        n_questions=dataset_config['n_questions'],
        n_pid=dataset_config['n_pid'] if args.with_pid else 0,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        n_know=dataset_config.get('n_know', 16)
    )
    
    # 加载权重
    baseline_model.load_state_dict(baseline_checkpoint['model_state_dict'])
    baseline_auc = baseline_checkpoint.get('auc', 0.0)
    
    print(f"✅ 基线模型加载成功")
    print(f"  模型维度: {d_model}")
    print(f"  注意力头数: {n_heads}")
    print(f"  基线AUC: {baseline_auc:.4f}")
    
    # 加载异常检测器
    print("🔧 加载异常检测器...")
    anomaly_checkpoint = torch.load(args.anomaly_detector_path, map_location='cpu', weights_only=False)
    
    # 创建异常检测器
    anomaly_detector = CausalAnomalyDetector(
        n_questions=dataset_config['n_questions'],
        n_pid=dataset_config['n_pid'] if args.with_pid else 0,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=2,  # 异常检测器通常层数较少
        dropout=0.1,
        window_size=10
    )
    
    # 加载权重
    anomaly_detector.load_state_dict(anomaly_checkpoint['model_state_dict'])
    anomaly_auc = anomaly_checkpoint.get('auc', 0.0)
    
    print(f"✅ 异常检测器加载成功")
    print(f"  检测器AUC: {anomaly_auc:.4f}")
    
    return baseline_model, anomaly_detector, d_model, baseline_auc


def train_anomaly_aware_kt(args, dataset_config, train_data, val_data):
    """
    训练异常感知知识追踪模型
    
    Args:
        args: 训练参数
        dataset_config: 数据集配置
        train_data: 训练数据加载器
        val_data: 验证数据加载器
        
    Returns:
        str: 训练好的模型路径
    """
    print_stage_header("异常感知知识追踪", 3)
    
    # 验证参数
    validate_stage3_parameters(args, dataset_config)
    
    # 打印参数配置
    print_stage3_parameters(args, dataset_config)
    
    # 加载预训练模型
    baseline_model, anomaly_detector, d_model, baseline_auc = load_pretrained_models(args, dataset_config)
    
    # 创建异常感知知识追踪模型
    print("\n🔧 创建异常感知知识追踪模型...")
    anomaly_aware_kt = AnomalyAwareKT(
        baseline_model=baseline_model,
        anomaly_detector=anomaly_detector,
        d_model=d_model,
        fusion_type=args.fusion_type,
        enable_context_enhancement=args.enable_context_enhancement,
        freeze_pretrained=args.freeze_pretrained,
        dropout=0.1
    )
    
    # 获取模型信息
    model_info = anomaly_aware_kt.get_model_info()
    
    print(f"✅ 异常感知模型创建成功")
    print(f"  总参数数: {model_info['total_parameters']:,}")
    print(f"  可训练参数: {model_info['trainable_parameters']:,}")
    print(f"  冻结参数: {model_info['frozen_parameters']:,}")
    print(f"  可训练比例: {model_info['trainable_ratio']:.1%}")
    print(f"  融合类型: {model_info['fusion_type']}")
    print(f"  上下文增强: {model_info['context_enhancement']}")
    
    # 移动到设备
    anomaly_aware_kt.to(args.device)
    
    # 创建训练器
    print("\n🚀 创建渐进式训练器...")
    save_dir = os.path.join(args.output_dir, 'anomaly_aware_kt')
    trainer = AnomalyAwareTrainer(
        model=anomaly_aware_kt,
        device=args.device,
        learning_rate=args.learning_rate,
        save_dir=save_dir,
        patience=args.patience
    )
    
    print(f"✅ 训练器创建成功")
    print(f"  保存目录: {save_dir}")
    
    # 开始渐进式训练
    print("\n" + "="*60)
    print("开始异常感知知识追踪训练...")
    print("="*60)
    
    try:
        training_result = trainer.progressive_train(
            train_loader=train_data,
            val_loader=val_data,
            stage1_epochs=args.fusion_epochs,
            stage2_epochs=args.joint_epochs,
            stage3_epochs=args.finetune_epochs,
            lambda_anomaly=args.lambda_anomaly
        )
        
        # 打印训练总结
        print_training_summary("异常感知知识追踪", training_result, save_dir)
        
        # 计算性能提升
        final_auc = training_result['best_auc']
        auc_improvement = final_auc - baseline_auc
        
        print(f"\n📊 性能对比:")
        print(f"  基线模型AUC: {baseline_auc:.4f}")
        print(f"  异常感知AUC: {final_auc:.4f}")
        print(f"  性能提升: {auc_improvement:+.4f}")
        
        if auc_improvement >= 0.05:
            print(f"🎉 达到目标！AUC提升 {auc_improvement:.4f} >= 0.05")
        elif auc_improvement >= 0.03:
            print(f"✅ 显著提升！AUC提升 {auc_improvement:.4f}")
        elif auc_improvement > 0:
            print(f"📈 有所提升，AUC提升 {auc_improvement:.4f}")
        else:
            print(f"⚠️ 性能未提升，需要调优")
        
        # 模型路径
        model_path = os.path.join(save_dir, 'best_anomaly_aware_kt.pt')
        print(f"\n💾 异常感知模型已保存: {model_path}")
        
        return model_path
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        raise
