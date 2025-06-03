#!/usr/bin/env python
"""
第三阶段执行脚本：异常感知知识追踪

将基线模型和异常检测器融合，实现异常感知的知识追踪，
目标提升AUC 0.05-0.1。
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
from anomaly_kt_v2.stages.stage3_anomaly_aware_kt import train_anomaly_aware_kt


def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description='第三阶段：异常感知知识追踪')
    
    # 基本参数
    parser.add_argument('--dataset', required=True,
                       choices=['assist09', 'assist17', 'algebra05', 'statics'],
                       help='数据集名称')
    parser.add_argument('--model_type', default='basic',
                       choices=['basic', 'extended'],
                       help='模型类型: basic(基础模型) 或 extended(扩展模型)')
    parser.add_argument('--baseline_model_path', required=True,
                       help='第一阶段基线模型路径')
    parser.add_argument('--anomaly_detector_path', required=True,
                       help='第二阶段异常检测器路径')
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
    
    # 融合参数
    parser.add_argument('--fusion_type', default='attention',
                       choices=['attention', 'gating', 'weighted'],
                       help='融合类型')
    parser.add_argument('--enable_context_enhancement', action='store_true', default=True,
                       help='启用上下文增强')
    parser.add_argument('--freeze_pretrained', action='store_true', default=True,
                       help='冻结预训练模型')
    parser.add_argument('--lambda_anomaly', type=float, default=0.1,
                       help='异常一致性损失权重')
    
    # 渐进式训练参数
    parser.add_argument('--fusion_epochs', type=int, default=10,
                       help='融合层训练轮数')
    parser.add_argument('--joint_epochs', type=int, default=20,
                       help='联合训练轮数')
    parser.add_argument('--finetune_epochs', type=int, default=10,
                       help='端到端微调轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--patience', type=int, default=10,
                       help='早停耐心值')
    
    return parser


def apply_model_type_presets(args):
    """根据模型类型应用预设配置"""
    if args.model_type == 'basic':
        # 基础模型预设
        args.experiment_suffix = "basic"
        print("🔧 应用基础模型预设配置")
        print("  - 异常感知融合: 标准配置")
        
    elif args.model_type == 'extended':
        # 扩展模型预设
        args.experiment_suffix = "extended"
        print("🚀 应用扩展模型预设配置")
        print("  - 异常感知融合: 增强配置")


def validate_model_paths(args):
    """验证模型路径"""
    print("\n🔍 验证模型路径...")
    
    # 验证基线模型
    if not os.path.exists(args.baseline_model_path):
        print(f"❌ 基线模型文件不存在: {args.baseline_model_path}")
        return False
    
    # 验证异常检测器
    if not os.path.exists(args.anomaly_detector_path):
        print(f"❌ 异常检测器文件不存在: {args.anomaly_detector_path}")
        return False
    
    print("✅ 模型路径验证通过")
    return True


def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()

    # 打印阶段标题
    print_stage_header("异常感知知识追踪", 3)
    
    # 应用模型类型预设
    apply_model_type_presets(args)
    
    # 验证模型路径
    if not validate_model_paths(args):
        return
    
    # 配置文件处理
    if args.auto_config:
        # 自动加载配置文件
        print(f"\n📄 自动加载配置文件...")
        config_name = f"{args.dataset}_stage3"
        config = load_auto_config(args.dataset, 'stage3')
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
    stage_name = f"stage3_{args.model_type}"
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
        model_path = train_anomaly_aware_kt(args, dataset_config, train_data, val_data)
        
        # 训练成功
        print(f"\n🎉 第三阶段训练完成！")
        print(f"📁 输出目录: {args.output_dir}")
        print(f"💾 异常感知模型: {model_path}")
        
        # 性能评估建议
        print(f"\n📋 后续评估建议:")
        print(f"1. 在测试集上评估最终性能")
        print(f"2. 与基线模型进行详细对比")
        print(f"3. 分析异常检测的贡献")
        print(f"4. 进行消融研究验证各组件效果")
        
        # 评估命令示例
        print(f"\n📋 评估命令示例:")
        print(f"python scripts/evaluate_final_performance.py \\")
        print(f"    --dataset {args.dataset} \\")
        print(f"    --baseline_model {args.baseline_model_path} \\")
        print(f"    --anomaly_aware_model {model_path} \\")
        print(f"    --device {args.device}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
