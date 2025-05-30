#!/usr/bin/env python
"""
修复的模型对比脚本
用于比较基线DTransformer和异常感知DTransformer的性能
"""

import os
import sys
import argparse
import torch
import tomlkit
import json
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DTransformer.data import KTData
from DTransformer.model import DTransformer
from DTransformer.eval import Evaluator

from anomaly_kt.detector import CausalAnomalyDetector
from anomaly_kt.model import AnomalyAwareDTransformer


def load_model_checkpoint(checkpoint_path, device='cpu'):
    """
    安全加载模型检查点，处理不同的存储格式
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 检查是否是完整的训练检查点格式
    if 'model_state_dict' in checkpoint:
        print(f"  - 检测到完整检查点格式，提取model_state_dict")
        return checkpoint['model_state_dict']
    else:
        print(f"  - 检测到直接模型参数格式")
        return checkpoint


def load_models(args, dataset_config):
    """加载基线和异常感知模型"""
    
    print("Loading baseline model...")
    # 1. 加载基线模型
    baseline_model = DTransformer(
        n_questions=dataset_config['n_questions'],
        n_pid=dataset_config['n_pid'] if args.with_pid else 0,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_know=args.n_know,
        n_layers=args.n_layers,
        proj=args.proj,
        lambda_cl=args.lambda_cl,
    )
    
    baseline_state_dict = load_model_checkpoint(args.baseline_model, args.device)
    baseline_model.load_state_dict(baseline_state_dict)
    baseline_model.to(args.device)
    baseline_model.eval()
    print("  ✓ Baseline model loaded successfully")
    
    # 2. 加载异常感知模型（如果指定）
    anomaly_model = None
    if args.anomaly_model:
        print("Loading anomaly-aware model...")
        
        # 首先加载异常检测器（如果需要）
        detector = None
        if args.detector_model:
            print("  - Loading anomaly detector...")
            detector = CausalAnomalyDetector(
                n_questions=dataset_config['n_questions'],
                n_pid=dataset_config['n_pid'] if args.with_pid else 0,
                d_model=args.detector_d_model,
                n_heads=args.detector_n_heads,
                n_layers=args.detector_n_layers,
                dropout=0.1,
                window_size=args.window_size
            )
            
            detector_state_dict = load_model_checkpoint(args.detector_model, args.device)
            detector.load_state_dict(detector_state_dict)
            detector.to(args.device)
            detector.eval()
            print("    ✓ Detector loaded successfully")
        
        # 创建异常感知模型
        anomaly_model = AnomalyAwareDTransformer(
            n_questions=dataset_config['n_questions'],
            n_pid=dataset_config['n_pid'] if args.with_pid else 0,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_know=args.n_know,
            n_layers=args.n_layers,
            proj=args.proj,
            lambda_cl=args.lambda_cl,
            anomaly_detector=detector,
            anomaly_weight=args.anomaly_weight
        )
        
        # 加载异常感知模型的参数
        anomaly_state_dict = load_model_checkpoint(args.anomaly_model, args.device)
        
        try:
            anomaly_model.load_state_dict(anomaly_state_dict)
            print("  ✓ Anomaly-aware model loaded successfully")
        except RuntimeError as e:
            print(f"  ✗ Error loading anomaly-aware model: {e}")
            print("  - 尝试部分加载...")
            
            # 尝试部分加载，忽略不匹配的键
            model_dict = anomaly_model.state_dict()
            filtered_dict = {k: v for k, v in anomaly_state_dict.items() 
                           if k in model_dict and v.shape == model_dict[k].shape}
            
            model_dict.update(filtered_dict)
            anomaly_model.load_state_dict(model_dict)
            
            print(f"  ✓ 成功加载 {len(filtered_dict)}/{len(anomaly_state_dict)} 个参数")
        
        anomaly_model.to(args.device)
        anomaly_model.eval()
    
    return baseline_model, anomaly_model


def evaluate_model(model, test_data, device, is_anomaly_aware=False):
    """评估单个模型"""
    evaluator = Evaluator()
    
    print(f"  - 正在评估{'异常感知' if is_anomaly_aware else '基线'}模型...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_data):
            # 获取数据
            if len(batch.data) == 2:
                q, s = batch.get("q", "s")
                pid = None
            else:
                q, s, pid = batch.get("q", "s", "pid")
            
            q, s = q.to(device), s.to(device)
            if pid is not None:
                pid = pid.to(device)
            
            try:
                # 预测
                if is_anomaly_aware and hasattr(model, 'predict_with_anomaly'):
                    y, *_ = model.predict_with_anomaly(q, s, pid)
                else:
                    y, *_ = model.predict(q, s, pid)
                
                # 评估
                evaluator.evaluate(s, torch.sigmoid(y))
                
            except Exception as e:
                print(f"    警告: 批次 {batch_idx} 处理失败: {e}")
                continue
            
            # 进度显示
            if batch_idx % 50 == 0:
                print(f"    处理进度: {batch_idx}/{len(test_data)} 批次")
    
    return evaluator.report()


def compare_models(baseline_model, anomaly_model, test_data, args):
    """比较两个模型的性能"""
    print("\n" + "="*60)
    print("开始模型评估...")
    print("="*60)
    
    # 评估基线模型
    print("\n1. 评估基线模型:")
    baseline_metrics = evaluate_model(baseline_model, test_data, args.device, is_anomaly_aware=False)
    print("  ✓ 基线模型评估完成")
    
    # 评估异常感知模型（如果有）
    anomaly_metrics = None
    if anomaly_model is not None:
        print("\n2. 评估异常感知模型:")
        anomaly_metrics = evaluate_model(anomaly_model, test_data, args.device, is_anomaly_aware=True)
        print("  ✓ 异常感知模型评估完成")
    
    # 打印结果
    print("\n" + "="*60)
    print("评估结果")
    print("="*60)
    
    print(f"\n基线模型性能:")
    for metric, value in baseline_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    if anomaly_metrics:
        print(f"\n异常感知模型性能:")
        for metric, value in anomaly_metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        # 计算提升
        print("\n" + "-"*40)
        print("性能提升分析:")
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
            symbol = "↑" if improvement > 0 else "↓"
            color = "✓" if improvement > 0 else "✗"
            print(f"  {metric.upper()}: {improvement:+.2f}% {symbol} {color}")
        
        # 检查是否达到目标
        print("\n" + "-"*40)
        print("目标达成情况:")
        auc_improvement = improvements['auc']
        if auc_improvement >= 1.0:
            print(f"✓ 成功: AUC提升 {auc_improvement:.2f}% (目标: ≥1%)")
        else:
            print(f"✗ 未达标: AUC提升 {auc_improvement:.2f}% (目标: ≥1%)")
            print("\n改进建议:")
            print("  - 调整异常权重 anomaly_weight")
            print("  - 增加异常检测器训练轮数")
            print("  - 尝试不同的异常生成比例")
    else:
        print("\n注意: 未提供异常感知模型，仅显示基线结果")
    
    # 保存结果
    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset': args.dataset,
        'baseline_model': args.baseline_model,
        'anomaly_model': args.anomaly_model,
        'baseline_metrics': baseline_metrics,
        'anomaly_metrics': anomaly_metrics,
        'improvements': improvements if anomaly_metrics else None
    }
    
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n结果已保存至: {args.output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Compare DTransformer models')
    
    # 数据参数
    parser.add_argument('-d', '--dataset', required=True, 
                        choices=['assist09', 'assist17', 'algebra05', 'statics'])
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('-p', '--with_pid', action='store_true')
    
    # 模型路径
    parser.add_argument('--baseline_model', required=True, 
                        help='Path to baseline DTransformer model')
    parser.add_argument('--anomaly_model', 
                        help='Path to anomaly-aware model (optional)')
    parser.add_argument('--detector_model', 
                        help='Path to anomaly detector model')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_know', type=int, default=32)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--proj', action='store_true', help='Use projection layer')
    parser.add_argument('--lambda_cl', type=float, default=0.1, help='CL loss weight')
    
    # 异常检测器参数
    parser.add_argument('--detector_d_model', type=int, default=128)
    parser.add_argument('--detector_n_heads', type=int, default=8)
    parser.add_argument('--detector_n_layers', type=int, default=2)
    parser.add_argument('--window_size', type=int, default=10)
    parser.add_argument('--anomaly_weight', type=float, default=0.5)
    
    # 其他参数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('-o', '--output_file', help='Save results to JSON file')
    
    args = parser.parse_args()
    
    print("配置信息:")
    print(f"  数据集: {args.dataset}")
    print(f"  设备: {args.device}")
    print(f"  基线模型: {args.baseline_model}")
    print(f"  异常感知模型: {args.anomaly_model}")
    print(f"  异常检测器: {args.detector_model}")
    
    # 加载数据集配置
    try:
        datasets = tomlkit.load(open(os.path.join(args.data_dir, 'datasets.toml')))
        dataset_config = datasets[args.dataset]
        print(f"  问题数: {dataset_config['n_questions']}")
        print(f"  知识点数: {dataset_config['n_pid']}")
    except Exception as e:
        print(f"错误: 无法加载数据集配置: {e}")
        return
    
    # 准备测试数据
    print(f"\n加载 {args.dataset} 测试数据...")
    try:
        test_data = KTData(
            os.path.join(args.data_dir, dataset_config['test']),
            dataset_config['inputs'],
            batch_size=args.batch_size
        )
        print(f"  ✓ 测试数据加载完成，共 {len(test_data)} 个批次")
    except Exception as e:
        print(f"错误: 无法加载测试数据: {e}")
        return
    
    # 加载模型
    try:
        baseline_model, anomaly_model = load_models(args, dataset_config)
    except Exception as e:
        print(f"错误: 模型加载失败: {e}")
        return
    
    # 比较模型
    try:
        results = compare_models(baseline_model, anomaly_model, test_data, args)
        print("\n✓ 模型比较完成!")
    except Exception as e:
        print(f"错误: 模型评估失败: {e}")
        return


if __name__ == '__main__':
    main()