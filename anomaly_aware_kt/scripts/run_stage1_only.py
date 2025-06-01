#!/usr/bin/env python
"""
单独运行第一阶段（基线模型训练）的示例脚本

这个脚本展示了如何使用提取的第一阶段代码来单独训练基线模型。
"""

import os
import sys
import argparse
import torch
import tomlkit
import yaml
from datetime import datetime
from typing import Dict, Any

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DTransformer.data import KTData
from anomaly_kt.stages.stage1_baseline import train_baseline_model


def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print(f"📄 已加载配置文件: {config_path}")
    return config


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> argparse.Namespace:
    """
    合并配置文件和命令行参数
    命令行参数优先级更高（如果用户显式提供了参数）
    """
    # 获取parser的默认值（不解析参数，避免required参数错误）
    parser = create_parser()

    # 获取所有参数的默认值
    defaults = {}
    for action in parser._actions:
        if action.dest != 'help':
            defaults[action.dest] = action.default

    # 创建新的args对象，从原args开始
    merged_args = argparse.Namespace(**vars(args))

    # 用配置文件中的值覆盖默认值（但不覆盖用户显式提供的命令行参数）
    for key, value in config.items():
        if hasattr(merged_args, key):  # 只设置脚本支持的参数
            # 检查用户是否显式提供了这个参数
            current_value = getattr(merged_args, key)
            default_value = defaults.get(key)

            # 如果当前值等于默认值，说明用户没有显式提供，使用配置文件的值
            if current_value == default_value:
                setattr(merged_args, key, value)
            # 否则保持用户提供的命令行参数值

    return merged_args


def auto_detect_config(dataset: str) -> str:
    """根据数据集自动检测配置文件"""
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs')
    config_file = f"{dataset}_baseline.yaml"
    config_path = os.path.join(config_dir, config_file)

    if os.path.exists(config_path):
        return config_path
    else:
        print(f"⚠️  未找到数据集 {dataset} 的默认配置文件: {config_path}")
        return None


def prepare_data(dataset_name: str, data_dir: str, batch_size: int, test_batch_size: int):
    """准备数据集"""
    # 加载数据集配置
    datasets = tomlkit.load(open(os.path.join(data_dir, 'datasets.toml')))
    dataset_config = datasets[dataset_name]

    # 创建数据加载器
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


def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description='Stage 1: Baseline Model Training')

    # 配置文件参数
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径 (YAML格式)')
    parser.add_argument('--auto_config', action='store_true',
                       help='根据数据集自动选择配置文件')

    # 基本参数
    parser.add_argument('--dataset', required=True,
                       choices=['assist09', 'assist17', 'algebra05', 'statics'],
                       help='Dataset name')
    parser.add_argument('--data_dir', default='data', help='Data directory')
    parser.add_argument('--output_dir', default=None, help='Output directory')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('-p', '--with_pid', action='store_true',
                       help='Use problem ID')

    # 数据参数
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=64, help='Test batch size')

    # 基线模型参数
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n_know', type=int, default=16, help='Number of knowledge concepts')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--lambda_cl', type=float, default=0.1, help='Contrastive learning weight')
    parser.add_argument('--proj', action='store_true', help='Use projection')
    parser.add_argument('--hard_neg', action='store_true', help='Use hard negatives')
    parser.add_argument('--window', type=int, default=1, help='Window size')

    # 训练参数
    parser.add_argument('--kt_epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--use_cl', action='store_true', help='Use contrastive learning')

    return parser


def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()

    print("="*60)
    print("第一阶段：基线模型训练")
    print("="*60)

    # 处理配置文件
    config = {}
    if args.config:
        # 用户指定了配置文件
        config = load_config(args.config)
    elif args.auto_config:
        # 自动检测配置文件
        config_path = auto_detect_config(args.dataset)
        if config_path:
            config = load_config(config_path)
        else:
            print("🔧 使用默认参数（未找到配置文件）")
    else:
        print("🔧 使用命令行参数和默认值")

    # 合并配置文件和命令行参数
    if config:
        print("🔄 合并配置文件和命令行参数...")
        args = merge_config_with_args(config, args)
        print("✅ 参数合并完成，命令行参数优先级更高")

    # 设置输出目录
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"output/stage1_{args.dataset}_{timestamp}"

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 输出目录: {args.output_dir}")

    # 保存配置
    config_save_path = os.path.join(args.output_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    print(f"📄 配置已保存到: {config_save_path}")

    # 准备数据
    print("\n📊 准备数据...")
    train_data, val_data, test_data, dataset_config = prepare_data(
        args.dataset, args.data_dir, args.batch_size, args.test_batch_size
    )

    print(f"  数据集: {args.dataset}")
    print(f"  问题数量: {dataset_config['n_questions']}")

    # 打印配置
    print("\n🔧 配置信息:")
    print(f"  模型维度: {args.d_model}")
    print(f"  注意力头数: {args.n_heads}")
    print(f"  知识概念数: {args.n_know}")
    print(f"  层数: {args.n_layers}")
    print(f"  训练轮数: {args.kt_epochs}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  设备: {args.device}")

    try:
        # 运行第一阶段训练
        model_path = train_baseline_model(args, dataset_config, train_data, val_data)

        print("\n" + "="*60)
        print("✅ 第一阶段完成!")
        print("="*60)
        print(f"📁 模型保存路径: {model_path}")
        print(f"📁 输出目录: {args.output_dir}")

        # 提供下一步建议
        print("\n💡 下一步:")
        print("1. 检查训练日志和模型性能")
        print("2. 运行第二阶段训练异常检测器")
        print("3. 或者使用此模型进行推理")

    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
