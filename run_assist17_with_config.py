#!/usr/bin/env python3
"""
ASSIST17 配置文件训练快速启动脚本
"""

import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='ASSIST17 配置文件训练')
    parser.add_argument('--dry_run', action='store_true', help='只显示命令，不执行')
    parser.add_argument('--custom_config', type=str, help='使用自定义配置文件路径')
    parser.add_argument('--d_model', type=int, help='覆盖模型维度')
    parser.add_argument('--learning_rate', type=float, help='覆盖学习率')
    parser.add_argument('--kt_epochs', type=int, help='覆盖训练轮数')
    parser.add_argument('--batch_size', type=int, help='覆盖批次大小')
    
    args = parser.parse_args()
    
    # 基础命令
    cmd_parts = [
        "python anomaly_aware_kt/scripts/run_stage1_only.py",
        "--dataset assist17"
    ]
    
    # 配置文件选择
    if args.custom_config:
        cmd_parts.append(f"--config {args.custom_config}")
    else:
        cmd_parts.append("--auto_config")
    
    # 参数覆盖
    if args.d_model:
        cmd_parts.append(f"--d_model {args.d_model}")
    if args.learning_rate:
        cmd_parts.append(f"--learning_rate {args.learning_rate}")
    if args.kt_epochs:
        cmd_parts.append(f"--kt_epochs {args.kt_epochs}")
    if args.batch_size:
        cmd_parts.append(f"--batch_size {args.batch_size}")
    
    cmd = " ".join(cmd_parts)
    
    print("🚀 ASSIST17 基线训练")
    print("=" * 60)
    print(f"执行命令: {cmd}")
    print("=" * 60)
    
    if args.dry_run:
        print("🔍 Dry run 模式 - 不会执行实际训练")
        return
    
    # 执行命令
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print("✅ 训练完成!")
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  用户中断训练")
        sys.exit(1)

if __name__ == "__main__":
    print("📋 使用示例:")
    print("  基础训练:     python run_assist17_with_config.py")
    print("  大模型训练:   python run_assist17_with_config.py --d_model 256")
    print("  快速训练:     python run_assist17_with_config.py --kt_epochs 50")
    print("  查看命令:     python run_assist17_with_config.py --dry_run")
    print()
    
    main()
