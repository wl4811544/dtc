#!/usr/bin/env python3
"""
课程学习异常检测训练脚本

基于我们设计的课程学习框架进行异常检测训练。
集成了BaselineAnomalyGenerator、CurriculumAnomalyGenerator、
CurriculumScheduler等核心组件。

使用示例:
python run_curriculum_learning.py --dataset assist17 --strategy hybrid --epochs 100
"""

import os
import sys
import argparse
import torch
import yaml
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from anomaly_kt.curriculum_learning import (
    CurriculumScheduler,
    CurriculumAnomalyGenerator,
    DifficultyEstimator,
    BaselineAnomalyGenerator
)


def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description='课程学习异常检测训练')
    
    # 基本参数
    parser.add_argument('--dataset', required=True,
                       choices=['assist09', 'assist17', 'algebra05', 'statics'],
                       help='数据集名称')
    parser.add_argument('--strategy', default='hybrid',
                       choices=['performance_driven', 'time_driven', 'hybrid'],
                       help='课程调度策略')
    parser.add_argument('--epochs', type=int, default=100,
                       help='总训练轮数')
    parser.add_argument('--output_dir', default=None,
                       help='输出目录')
    
    # 课程学习参数
    parser.add_argument('--anomaly_ratio', type=float, default=0.1,
                       help='异常比例')
    parser.add_argument('--baseline_ratio', type=float, default=0.05,
                       help='基线异常比例')
    parser.add_argument('--max_patience', type=int, default=5,
                       help='课程推进的最大耐心值')
    
    # 实验参数
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备')
    parser.add_argument('--dry_run', action='store_true',
                       help='只测试组件，不进行实际训练')
    
    return parser


def test_curriculum_components(args):
    """测试课程学习组件"""
    print("🧪 测试课程学习组件")
    print("=" * 50)
    
    # 1. 测试BaselineAnomalyGenerator
    print("1. 测试BaselineAnomalyGenerator...")
    baseline_gen = BaselineAnomalyGenerator()
    
    # 创建模拟数据
    batch_size, seq_len = 4, 20
    q = torch.randint(0, 100, (batch_size, seq_len))
    s = torch.randint(0, 2, (batch_size, seq_len))
    
    s_anomaly, anomaly_labels = baseline_gen.generate_baseline_anomalies(
        q, s, strategy='random_flip', anomaly_ratio=0.2
    )
    
    print(f"  ✅ 生成异常数据: {s_anomaly.shape}")
    print(f"  ✅ 异常标签: {anomaly_labels.sum().item()} 个异常位置")
    
    # 2. 测试DifficultyEstimator
    print("\n2. 测试DifficultyEstimator...")
    difficulty_estimator = DifficultyEstimator(args.dataset)
    
    difficulty_score = difficulty_estimator.estimate_sample_difficulty(
        q[0], s[0], anomaly_labels[0], position=10
    )
    print(f"  ✅ 难度评估: {difficulty_score:.3f}")
    
    # 3. 测试CurriculumScheduler
    print("\n3. 测试CurriculumScheduler...")
    scheduler = CurriculumScheduler(args.strategy, args.dataset, args.epochs)
    
    # 模拟几轮更新
    for epoch in range(5):
        metrics = {
            'auc': 0.7 + epoch * 0.02,
            'f1': 0.65 + epoch * 0.02,
            'precision': 0.7 + epoch * 0.01,
            'recall': 0.6 + epoch * 0.03
        }
        schedule_info = scheduler.update(epoch, metrics)
        print(f"  Epoch {epoch}: Phase {schedule_info['current_phase']}, "
              f"Progress: {schedule_info['phase_progress']:.2f}")
    
    # 4. 测试CurriculumAnomalyGenerator
    print("\n4. 测试CurriculumAnomalyGenerator...")
    curriculum_gen = CurriculumAnomalyGenerator(args.dataset)
    
    curriculum_config = scheduler.get_current_curriculum_config()
    s_curr, labels_curr, diff_curr = curriculum_gen.generate_curriculum_anomalies(
        q, s,
        difficulty_levels=curriculum_config['difficulty_levels'],
        level_weights=curriculum_config['level_weights'],
        anomaly_ratio=args.anomaly_ratio,
        include_baseline=True,
        baseline_ratio=args.baseline_ratio
    )
    
    print(f"  ✅ 课程异常数据: {s_curr.shape}")
    print(f"  ✅ 异常标签: {labels_curr.sum().item()} 个异常位置")
    print(f"  ✅ 平均难度: {diff_curr[labels_curr > 0].mean().item():.3f}")
    
    print("\n✅ 所有组件测试通过!")
    return True


def load_config_if_exists(dataset: str) -> dict:
    """加载配置文件（如果存在）"""
    config_dir = Path(__file__).parent.parent / 'configs'
    config_file = config_dir / f'{dataset}_curriculum.yaml'
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print(f"📄 加载配置文件: {config_file}")
        return config
    else:
        print(f"⚠️  未找到配置文件: {config_file}")
        return {}


def create_output_directory(args) -> Path:
    """创建输出目录"""
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"output/curriculum_{args.dataset}_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 输出目录: {output_dir}")
    
    return output_dir


def save_experiment_config(args, output_dir: Path):
    """保存实验配置"""
    config = {
        'dataset': args.dataset,
        'strategy': args.strategy,
        'epochs': args.epochs,
        'anomaly_ratio': args.anomaly_ratio,
        'baseline_ratio': args.baseline_ratio,
        'max_patience': args.max_patience,
        'seed': args.seed,
        'device': args.device,
        'timestamp': datetime.now().isoformat()
    }
    
    config_file = output_dir / 'experiment_config.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config, f, indent=2)
    
    print(f"💾 实验配置已保存: {config_file}")


def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    print("🎓 课程学习异常检测训练")
    print("=" * 60)
    print(f"数据集: {args.dataset}")
    print(f"策略: {args.strategy}")
    print(f"轮数: {args.epochs}")
    print(f"设备: {args.device}")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 创建输出目录
    output_dir = create_output_directory(args)
    
    # 保存实验配置
    save_experiment_config(args, output_dir)
    
    # 加载配置文件
    config = load_config_if_exists(args.dataset)
    
    if args.dry_run:
        print("\n🧪 Dry Run 模式 - 仅测试组件")
        success = test_curriculum_components(args)
        if success:
            print("\n✅ 组件测试成功，可以开始正式训练")
            print("\n🚀 正式训练命令:")
            cmd = f"python {__file__} --dataset {args.dataset} --strategy {args.strategy} --epochs {args.epochs}"
            print(f"   {cmd}")
        return
    
    # TODO: 集成实际的训练流程
    print("\n⚠️  实际训练功能正在开发中...")
    print("当前已完成的组件:")
    print("  ✅ BaselineAnomalyGenerator")
    print("  ✅ CurriculumAnomalyGenerator")
    print("  ✅ DifficultyEstimator")
    print("  ✅ CurriculumScheduler")
    print("  🔄 CurriculumTrainer (需要与现有训练器集成)")
    
    print(f"\n📁 输出目录已准备: {output_dir}")
    print("💡 建议先运行 --dry_run 测试组件功能")


if __name__ == "__main__":
    main()
