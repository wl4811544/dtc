#!/usr/bin/env python
"""
Stage 1: 基线模型训练脚本

训练标准的DTransformer知识追踪模型作为后续异常感知训练的基线
"""

import os
import sys
import argparse
import torch
from datetime import datetime

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(project_root))  # 添加上级目录以访问DTransformer
sys.path.append(project_root)  # 添加项目根目录

from anomaly_kt_v2.core.common import prepare_data, setup_output_directory, save_config, print_stage_header
from anomaly_kt_v2.configs import load_auto_config, merge_config_with_args
from anomaly_kt_v2.stages.stage1_baseline import train_baseline_model


def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description='Stage 1: 基线模型训练')

    # 配置文件参数
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径 (YAML格式)')
    parser.add_argument('--auto_config', action='store_true',
                       help='根据数据集自动选择配置文件')

    # 基本参数
    parser.add_argument('--dataset', required=True,
                       choices=['assist09', 'assist17', 'algebra05', 'statics'],
                       help='数据集名称')
    parser.add_argument('--model_type', default='basic',
                       choices=['basic', 'extended'],
                       help='模型类型: basic(基础模型) 或 extended(扩展模型)')
    parser.add_argument('--data_dir', default='data', help='数据目录')
    parser.add_argument('--output_dir', default=None, help='输出目录')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='训练设备')
    parser.add_argument('-p', '--with_pid', action='store_true', default=True,
                       help='使用问题ID')

    # 数据参数
    parser.add_argument('--batch_size', type=int, default=16, help='训练批次大小')
    parser.add_argument('--test_batch_size', type=int, default=32, help='测试批次大小')

    # 基线模型参数
    parser.add_argument('--d_model', type=int, default=128, help='模型维度')
    parser.add_argument('--n_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--n_know', type=int, default=16, help='知识概念数')
    parser.add_argument('--n_layers', type=int, default=3, help='网络层数')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout率')
    parser.add_argument('--lambda_cl', type=float, default=0.1, help='对比学习权重')
    parser.add_argument('--proj', action='store_true', default=True, help='使用投影层')
    parser.add_argument('--hard_neg', action='store_true', default=False, help='使用困难负样本')
    parser.add_argument('--window', type=int, default=1, help='注意力窗口大小')

    # 训练参数
    parser.add_argument('--kt_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='学习率')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--use_cl', action='store_true', default=True, help='使用对比学习')

    return parser


def apply_model_type_presets(args):
    """根据模型类型应用预设配置"""
    if args.model_type == 'basic':
        # 基础模型预设 (与您的基础模型配置一致)
        args.d_model = 128
        args.n_heads = 8
        args.experiment_suffix = "basic"
        print("🔧 应用基础模型预设配置")
        print("  - d_model: 128, n_heads: 8")
        print("  - 与您的基础模型训练配置一致")
        print("  - 预期AUC: 0.7407")

    elif args.model_type == 'extended':
        # 扩展模型预设 (与您的扩展模型配置一致)
        args.d_model = 256
        args.n_heads = 16
        args.experiment_suffix = "extended"
        print("🚀 应用扩展模型预设配置")
        print("  - d_model: 256, n_heads: 16")
        print("  - 与您的扩展模型训练配置一致")
        print("  - 预期AUC: 0.7404")


def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()

    # 打印阶段标题
    print_stage_header("基线模型训练", 1)

    # 应用模型类型预设
    apply_model_type_presets(args)

    # 处理配置文件
    config = {}
    if args.config:
        # 用户指定了配置文件
        from anomaly_kt_v2.configs import load_config
        config = load_config(args.config)
        print(f"📄 已加载配置文件: {args.config}")
    elif args.auto_config:
        # 自动检测配置文件
        config = load_auto_config(args.dataset, 'baseline')
        if config:
            print(f"📄 已自动加载配置文件: {args.dataset}_baseline.yaml")
        else:
            print("🔧 使用默认参数（未找到配置文件）")
    else:
        print("🔧 使用命令行参数和默认值")

    # 合并配置文件和命令行参数
    if config:
        print("🔄 合并配置文件和命令行参数...")
        merge_config_with_args(config, args)
        print("✅ 参数合并完成，命令行参数优先级更高")

    # 设置输出目录 (包含模型类型)
    stage_name = f"stage1_{args.model_type}"
    args.output_dir = setup_output_directory(args.output_dir, args.dataset, stage_name)
    print(f"📁 输出目录: {args.output_dir}")

    # 保存配置
    config_save_path = save_config(vars(args), args.output_dir)
    print(f"📄 配置已保存到: {config_save_path}")

    # 准备数据
    print("\n📊 准备数据...")
    train_data, val_data, test_data, dataset_config = prepare_data(
        args.dataset, args.data_dir, args.batch_size, args.test_batch_size
    )

    print(f"✅ 数据准备完成")
    print(f"  数据集: {args.dataset}")
    print(f"  问题数量: {dataset_config['n_questions']}")
    print(f"  问题ID数量: {dataset_config.get('n_pid', 0)}")

    try:
        # 运行第一阶段训练
        model_path = train_baseline_model(args, dataset_config, train_data, val_data)

        print("\n" + "="*60)
        print("🎉 第一阶段训练完成!")
        print("="*60)
        print(f"💾 模型保存路径: {model_path}")
        print(f"📁 输出目录: {args.output_dir}")

        # 提供下一步建议
        print("\n💡 下一步建议:")
        print("1. 检查训练日志和模型性能")
        print("2. 运行第二阶段训练异常分类器")
        print("3. 或者使用此模型进行推理")

        print(f"\n📋 第二阶段命令示例:")
        print(f"python scripts/run_stage2_anomaly_classifier.py \\")
        print(f"    --dataset {args.dataset} \\")
        print(f"    --model_type {args.model_type} \\")
        print(f"    --baseline_model_path {model_path} \\")
        print(f"    --device {args.device}")

    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
