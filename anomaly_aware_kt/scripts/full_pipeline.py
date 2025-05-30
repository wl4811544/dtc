#!/usr/bin/env python
"""
完整的异常感知知识追踪训练流程

包括：
1. 训练异常检测器
2. 训练异常感知的知识追踪模型
3. 评估性能提升
"""

import os
import sys
import argparse
import torch
import tomlkit
import yaml
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DTransformer.data import KTData
from DTransformer.model import DTransformer

from anomaly_kt.generator import AnomalyGenerator
from anomaly_kt.detector import CausalAnomalyDetector
from anomaly_kt.model import AnomalyAwareDTransformer
from anomaly_kt.trainer import KnowledgeTracingTrainer
from anomaly_kt.unified_trainer import UnifiedAnomalyTrainer
from anomaly_kt.evaluator import ComparisonEvaluator, plot_training_curves




def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        else:
            raise ValueError("Only YAML config files are supported")
    return config


def prepare_data(dataset_name: str, data_dir: str, batch_size: int, test_batch_size: int):
    """准备数据集"""
    # 加载数据集配置
    datasets = tomlkit.load(open(os.path.join(data_dir, 'datasets.toml')))
    dataset_config = datasets[dataset_name]

    # 创建数据加载器improved_anomaly_trainer
    train_data = KTData(
        os.path.join(data_dir, dataset_config['train']),
        dataset_config['inputs'],
        batch_size=batch_size,
        shuffle=True
    )

    val_data = KTData(
        os.path.join(data_dir, dataset_config.get('valid', dataset_config['test'])),
        dataset_config['inputs'],
        batch_size=test_batch_size
    )

    test_data = KTData(
        os.path.join(data_dir, dataset_config['test']),
        dataset_config['inputs'],
        batch_size=test_batch_size
    )

    return train_data, val_data, test_data, dataset_config


def train_baseline_model(args, dataset_config, train_data, val_data):
    """训练基线DTransformer模型"""
    print("\n" + "="*60)
    print("PHASE 1: Training Baseline DTransformer")
    print("="*60)

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


def train_anomaly_detector(args, dataset_config, train_data, val_data):
    """训练异常检测器"""
    print("\n" + "="*60)
    print("PHASE 2: Training Anomaly Detector")
    print("="*60)

    # 创建模型
    detector = CausalAnomalyDetector(
        n_questions=dataset_config['n_questions'],
        n_pid=dataset_config['n_pid'] if args.with_pid else 0,
        d_model=args.detector_d_model,
        n_heads=args.detector_n_heads,
        n_layers=args.detector_n_layers,
        dropout=args.detector_dropout,
        window_size=args.window_size
    )

    print(f"🧠 模型参数: {sum(p.numel() for p in detector.parameters()):,}")

    # 选择训练策略
    strategy = getattr(args, 'training_strategy', 'basic')

    # 创建统一训练器
    trainer = UnifiedAnomalyTrainer(
        model=detector,
        device=args.device,
        save_dir=os.path.join(args.output_dir, 'detector'),
        patience=args.detector_patience,
        strategy=strategy
    )

    # 显示策略信息
    strategy_info = trainer.get_strategy_info()
    print(f"\n📋 {strategy_info['description']}")

    # 训练
    detector_metrics = trainer.train(
        train_loader=train_data,
        val_loader=val_data,
        epochs=args.detector_epochs,
        learning_rate=args.detector_lr,
        anomaly_ratio=args.anomaly_ratio,
        optimize_for=args.optimize_for
    )

    print(f"\n🎉 检测器训练完成!")
    print(f"📊 最终结果:")
    print(f"  F1 Score: {detector_metrics['f1_score']:.4f}")
    print(f"  AUC-ROC: {detector_metrics['auc_roc']:.4f}")
    print(f"  Recall: {detector_metrics['recall']:.4f}")
    print(f"  Precision: {detector_metrics['precision']:.4f}")

    # 绘制训练曲线
    plot_training_curves(trainer.history, os.path.join(args.output_dir, 'detector'))

    return os.path.join(args.output_dir, 'detector', 'best_model.pt')


def train_anomaly_aware_model(args, dataset_config, train_data, val_data, detector_path):
    """训练异常感知的知识追踪模型"""
    print("\n" + "="*60)
    print("PHASE 3: Training Anomaly-Aware DTransformer")
    print("="*60)

    # 加载异常检测器
    detector = CausalAnomalyDetector(
        n_questions=dataset_config['n_questions'],
        n_pid=dataset_config['n_pid'] if args.with_pid else 0,
        d_model=args.detector_d_model,
        n_heads=args.detector_n_heads,
        n_layers=args.detector_n_layers,
        dropout=args.detector_dropout,
        window_size=args.window_size
    )

    # 正确加载checkpoint
    checkpoint = torch.load(detector_path, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        detector.load_state_dict(checkpoint['model_state_dict'])
    else:
        detector.load_state_dict(checkpoint)
    detector.eval()

    # 创建异常感知模型
    model = AnomalyAwareDTransformer(
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
        window=args.window,
        anomaly_detector=detector,
        anomaly_weight=args.anomaly_weight
    )

    # 训练器
    trainer = KnowledgeTracingTrainer(
        model=model,
        device=args.device,
        save_dir=os.path.join(args.output_dir, 'anomaly_aware'),
        patience=args.patience
    )

    # 训练
    anomaly_metrics = trainer.train(
        train_loader=train_data,
        val_loader=val_data,
        epochs=args.kt_epochs,
        learning_rate=args.learning_rate,
        use_cl=args.use_cl
    )

    print(f"\nAnomaly-aware training completed!")
    print(f"Best AUC: {anomaly_metrics['auc']:.4f}")

    return os.path.join(args.output_dir, 'anomaly_aware', 'best_model.pt')


def evaluate_models(args, dataset_config, test_data, baseline_path, anomaly_path, detector_path):
    """评估模型性能"""
    print("\n" + "="*60)
    print("PHASE 4: Model Evaluation")
    print("="*60)

    # 加载基线模型
    baseline_model = DTransformer(
        n_questions=dataset_config['n_questions'],
        n_pid=dataset_config['n_pid'] if args.with_pid else 0,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_know=args.n_know,
        n_layers=args.n_layers
    )
    baseline_model.load_state_dict(torch.load(baseline_path, map_location=args.device))
    baseline_model.to(args.device)

    # 加载异常检测器
    detector = CausalAnomalyDetector(
        n_questions=dataset_config['n_questions'],
        n_pid=dataset_config['n_pid'] if args.with_pid else 0,
        d_model=args.detector_d_model,
        n_heads=args.detector_n_heads,
        n_layers=args.detector_n_layers,
        dropout=args.detector_dropout,
        window_size=args.window_size
    )
    checkpoint = torch.load(detector_path, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        detector.load_state_dict(checkpoint['model_state_dict'])
    else:
        detector.load_state_dict(checkpoint)
    detector.to(args.device)

    # 加载异常感知模型
    anomaly_model = AnomalyAwareDTransformer(
        n_questions=dataset_config['n_questions'],
        n_pid=dataset_config['n_pid'] if args.with_pid else 0,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_know=args.n_know,
        n_layers=args.n_layers,
        anomaly_detector=detector,
        anomaly_weight=args.anomaly_weight
    )
    anomaly_model.load_state_dict(torch.load(anomaly_path, map_location=args.device))
    anomaly_model.to(args.device)

    # 评估
    evaluator = ComparisonEvaluator()
    results = evaluator.evaluate_models(test_data, baseline_model, anomaly_model, args.device)

    # 打印结果
    evaluator.print_comparison(results)

    # 保存结果
    import json
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Full Anomaly-Aware KT Pipeline')

    # 基本参数
    parser.add_argument('--dataset', required=True, choices=['assist09', 'assist17', 'algebra05', 'statics'])
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--config', help='Config file path')
    parser.add_argument('-p', '--with_pid', action='store_true')

    # 如果提供了配置文件，从配置文件加载参数
    args, _ = parser.parse_known_args()

    if args.config:
        config = load_config(args.config)
        parser.set_defaults(**config)

    # 数据参数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=64)

    # 基线模型参数
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_know', type=int, default=16)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lambda_cl', type=float, default=0.1)
    parser.add_argument('--proj', action='store_true')
    parser.add_argument('--hard_neg', action='store_true')
    parser.add_argument('--window', type=int, default=1)

    # 异常检测器参数
    parser.add_argument('--detector_d_model', type=int, default=128)
    parser.add_argument('--detector_n_heads', type=int, default=8)
    parser.add_argument('--detector_n_layers', type=int, default=2)
    parser.add_argument('--detector_dropout', type=float, default=0.1)
    parser.add_argument('--window_size', type=int, default=10)
    parser.add_argument('--anomaly_ratio', type=float, default=0.1)
    parser.add_argument('--optimize_for', default='f1_score',
                       choices=['f1_score', 'auc_roc', 'recall', 'precision', 'balanced_accuracy', 'mcc'],
                       help='Optimization target: f1_score (balanced), recall (find all anomalies), precision (avoid false alarms), auc_roc (overall performance)')

    # 训练参数
    parser.add_argument('--kt_epochs', type=int, default=100)
    parser.add_argument('--detector_epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--detector_lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--detector_patience', type=int, default=10)
    parser.add_argument('--use_cl', action='store_true')

    # 异常感知参数
    parser.add_argument('--anomaly_weight', type=float, default=0.5)

    # 控制参数
    parser.add_argument('--skip_baseline', action='store_true', help='Skip baseline training')
    parser.add_argument('--skip_detector', action='store_true', help='Skip detector training')
    parser.add_argument('--skip_anomaly_training', action='store_true', help='Skip anomaly-aware model training')
    parser.add_argument('--baseline_path', help='Path to existing baseline model')
    parser.add_argument('--detector_path', help='Path to existing detector model')
    parser.add_argument('--anomaly_path', help='Path to existing anomaly-aware model')

    # 训练策略参数
    parser.add_argument('--training_strategy', default='basic',
                       choices=['basic', 'enhanced', 'aggressive'],
                       help='Training strategy for anomaly detector: basic (default), enhanced, or aggressive')

    # 向后兼容的参数
    parser.add_argument('--use_aggressive_strategy', action='store_true',
                       help='Use aggressive training strategy (deprecated, use --training_strategy aggressive)')

    args = parser.parse_args()

    # 向后兼容性处理
    if args.use_aggressive_strategy:
        args.training_strategy = 'aggressive'
        print("⚠️  注意: --use_aggressive_strategy 已弃用，请使用 --training_strategy aggressive")

    # 检查基线模型文件是否存在
    if args.skip_baseline:
        if not args.baseline_path:
            print("ERROR: --skip_baseline requires --baseline_path to be specified")
            sys.exit(1)

        if not os.path.exists(args.baseline_path):
            print(f"ERROR: Baseline model file not found: {args.baseline_path}")
            print(f"Please check the path or remove --skip_baseline to train a new baseline model")
            sys.exit(1)
        else:
            print(f"✓ Baseline model found: {args.baseline_path}")

    # 检查异常检测器文件是否存在（如果指定了）
    if args.skip_detector:
        if not args.detector_path:
            print("ERROR: --skip_detector requires --detector_path to be specified")
            sys.exit(1)

        if not os.path.exists(args.detector_path):
            print(f"ERROR: Detector model file not found: {args.detector_path}")
            print(f"Please check the path or remove --skip_detector to train a new detector")
            sys.exit(1)
        else:
            print(f"✓ Detector model found: {args.detector_path}")

    # 检查异常感知模型文件是否存在（如果指定了）
    if args.skip_anomaly_training:
        if not args.anomaly_path:
            print("ERROR: --skip_anomaly_training requires --anomaly_path to be specified")
            sys.exit(1)

        if not os.path.exists(args.anomaly_path):
            print(f"ERROR: Anomaly-aware model file not found: {args.anomaly_path}")
            print(f"Please check the path or remove --skip_anomaly_training to train a new anomaly-aware model")
            sys.exit(1)
        else:
            print(f"✓ Anomaly-aware model found: {args.anomaly_path}")

    # 设置输出目录
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"output/{args.dataset}_{timestamp}"

    os.makedirs(args.output_dir, exist_ok=True)

    # 保存配置
    config_save_path = os.path.join(args.output_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)

    print("Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    # 准备数据
    train_data, val_data, test_data, dataset_config = prepare_data(
        args.dataset, args.data_dir, args.batch_size, args.test_batch_size
    )

    # 1. 训练基线模型
    if not args.skip_baseline:
        baseline_path = train_baseline_model(args, dataset_config, train_data, val_data)
    else:
        baseline_path = args.baseline_path
        print(f"Using existing baseline model: {baseline_path}")

    # 2. 训练异常检测器
    if not args.skip_detector:
        detector_path = train_anomaly_detector(args, dataset_config, train_data, val_data)
    else:
        detector_path = args.detector_path
        print(f"Using existing detector: {detector_path}")

    # 3. 训练异常感知模型
    if not args.skip_anomaly_training:
        anomaly_path = train_anomaly_aware_model(args, dataset_config, train_data, val_data, detector_path)
    else:
        anomaly_path = args.anomaly_path
        print(f"Using existing anomaly-aware model: {anomaly_path}")

    # 4. 评估结果
    results = evaluate_models(args, dataset_config, test_data, baseline_path, anomaly_path, detector_path)

    print("\n" + "="*60)
    print("PIPELINE COMPLETED!")
    print("="*60)

    # 检查是否达到目标
    improvement = results['improvements']['auc']
    if improvement >= 1.0:
        print(f"✓ SUCCESS: Target achieved! AUC improved by {improvement:.2f}%")
    else:
        print(f"✗ Target not met. AUC improved by {improvement:.2f}% (need ≥1%)")
        print("\nSuggestions:")
        print("- Try adjusting anomaly_weight (current: {})".format(args.anomaly_weight))
        print("- Increase detector training epochs")
        print("- Experiment with different anomaly_ratio values")


if __name__ == '__main__':
    main()