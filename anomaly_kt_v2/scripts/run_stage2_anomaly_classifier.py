#!/usr/bin/env python
"""
第二阶段执行脚本：异常分类器训练

基于课程学习的异常检测器训练脚本。
完全基于DTransformer原始代码，不依赖任何anomaly_kt模块。
"""

import os
import sys
import argparse
import torch

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(project_root))  # 添加上级目录以访问DTransformer
sys.path.append(project_root)  # 添加项目根目录

from DTransformer.data import KTData
from anomaly_kt_v2.core.common import prepare_data, setup_output_directory, save_config, print_stage_header
from anomaly_kt_v2.configs import load_auto_config, merge_config_with_args
from anomaly_kt_v2.stages.stage2_anomaly_classifier import train_anomaly_classifier, load_baseline_model_info


def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description='第二阶段：异常分类器训练')
    
    # 基本参数
    parser.add_argument('--dataset', required=True,
                       choices=['assist09', 'assist17', 'algebra05', 'statics'],
                       help='数据集名称')
    parser.add_argument('--model_type', default='basic',
                       choices=['basic', 'extended'],
                       help='模型类型: basic(基础模型) 或 extended(扩展模型)')
    parser.add_argument('--baseline_model_path', required=True,
                       help='第一阶段基线模型路径')
    parser.add_argument('--data_dir', default='data', help='数据目录')
    parser.add_argument('--output_dir', default=None, help='输出目录')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='训练设备')
    parser.add_argument('-p', '--with_pid', action='store_true', default=True,
                       help='使用问题ID')
    
    # 配置文件参数
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--auto_config', action='store_true',
                       help='自动加载配置文件')
    
    # 数据参数
    parser.add_argument('--batch_size', type=int, default=16, help='训练批次大小')
    parser.add_argument('--test_batch_size', type=int, default=32, help='测试批次大小')
    
    # 异常检测器参数
    parser.add_argument('--d_model', type=int, default=128, help='模型隐藏维度')
    parser.add_argument('--n_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--n_layers', type=int, default=2, help='异常检测器层数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    parser.add_argument('--window_size', type=int, default=10, help='统计特征窗口大小')
    
    # 课程学习参数
    parser.add_argument('--anomaly_epochs', type=int, default=50, help='异常检测器训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--curriculum_type', default='linear',
                       choices=['linear', 'exponential', 'cosine', 'step'],
                       help='课程调度类型')
    parser.add_argument('--initial_difficulty', type=float, default=0.1, help='初始难度')
    parser.add_argument('--final_difficulty', type=float, default=0.8, help='最终难度')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='预热轮数')
    
    return parser


def apply_model_type_presets(args):
    """根据模型类型应用预设配置"""
    if args.model_type == 'basic':
        # 基础模型预设
        args.d_model = 128
        args.n_heads = 8
        args.experiment_suffix = "basic"
        print("🔧 应用基础模型预设配置")
        print("  - 异常检测器: d_model=128, n_heads=8")
        
    elif args.model_type == 'extended':
        # 扩展模型预设
        args.d_model = 256
        args.n_heads = 16
        args.experiment_suffix = "extended"
        print("🚀 应用扩展模型预设配置")
        print("  - 异常检测器: d_model=256, n_heads=16")


def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()

    # 打印阶段标题
    print_stage_header("异常分类器训练", 2)
    
    # 应用模型类型预设
    apply_model_type_presets(args)
    
    # 验证基线模型路径
    if not os.path.exists(args.baseline_model_path):
        print(f"❌ 基线模型文件不存在: {args.baseline_model_path}")
        return
    
    # 加载基线模型信息
    print("\n📄 加载基线模型信息...")
    baseline_info = load_baseline_model_info(args.baseline_model_path)
    
    # 从基线模型继承配置
    args.d_model = baseline_info['d_model']
    args.n_heads = baseline_info['n_heads']
    args.with_pid = baseline_info['with_pid']
    
    print(f"✅ 已从基线模型继承配置")
    print(f"  模型维度: {args.d_model}")
    print(f"  注意力头数: {args.n_heads}")
    print(f"  使用问题ID: {args.with_pid}")
    
    # 配置文件处理
    if args.auto_config:
        # 自动加载配置文件
        print(f"\n📄 自动加载配置文件...")
        config_name = f"{args.dataset}_stage2"
        config = load_auto_config(args.dataset, 'stage2')
        if config:
            print(f"✅ 已自动加载配置文件: {config_name}.yaml")
            merge_config_with_args(config, args)
            print("🔄 合并配置文件和命令行参数...")
            print("✅ 参数合并完成，命令行参数优先级更高")
        else:
            print(f"⚠️ 未找到自动配置文件，使用默认参数")
    
    elif args.config:
        # 用户指定了配置文件
        from anomaly_kt_v2.configs import load_config
        config = load_config(args.config)
        merge_config_with_args(config, args)
        print(f"📄 已加载配置文件: {args.config}")
        print("🔄 合并配置文件和命令行参数...")
        print("✅ 参数合并完成，命令行参数优先级更高")
    
    # 设置输出目录 (包含模型类型)
    stage_name = f"stage2_{args.model_type}"
    args.output_dir = setup_output_directory(args.output_dir, args.dataset, stage_name)
    print(f"📁 输出目录: {args.output_dir}")
    
    # 保存配置
    config_path = save_config(vars(args), args.output_dir)
    print(f"📄 配置已保存到: {config_path}")
    
    try:
        # 准备数据
        print("\n📊 准备数据...")
        train_data, val_data, test_data, dataset_config = prepare_data(
            args.dataset, args.data_dir, args.batch_size, args.test_batch_size
        )
        print("✅ 数据准备完成")
        print(f"  数据集: {args.dataset}")
        print(f"  问题数量: {dataset_config['n_questions']}")
        print(f"  问题ID数量: {dataset_config['n_pid']}")
        
        # 开始训练
        model_path = train_anomaly_classifier(args, dataset_config, train_data, val_data)
        
        # 训练成功
        print(f"\n🎉 第二阶段训练完成！")
        print(f"📁 输出目录: {args.output_dir}")
        print(f"💾 异常检测器: {model_path}")
        
        # 第三阶段命令示例
        print(f"\n📋 第三阶段命令示例:")
        print(f"python scripts/run_stage3_anomaly_aware_kt.py \\")
        print(f"    --dataset {args.dataset} \\")
        print(f"    --model_type {args.model_type} \\")
        print(f"    --baseline_model_path {args.baseline_model_path} \\")
        print(f"    --anomaly_detector_path {model_path} \\")
        print(f"    --device {args.device}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
