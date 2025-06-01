#!/usr/bin/env python3
"""
assist17数据集基线训练启动脚本
使用保守的起点参数设置
"""

import os
import sys
import argparse
from assist17_baseline_config import Assist17BaselineConfig

def main():
    # 创建配置
    config = Assist17BaselineConfig()
    
    # 打印配置信息
    config.print_config()
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 构建训练命令
    cmd_parts = [
        "python run_stage1_only.py",
        f"--dataset {config.dataset}",
        f"--device {config.device}",
        f"--data_dir {config.data_dir}",
        f"--output_dir {config.output_dir}",
        f"--batch_size {config.batch_size}",
        f"--test_batch_size {config.test_batch_size}",
        f"--d_model {config.d_model}",
        f"--n_heads {config.n_heads}",
        f"--n_know {config.n_know}",
        f"--n_layers {config.n_layers}",
        f"--dropout {config.dropout}",
        f"--lambda_cl {config.lambda_cl}",
        f"--window {config.window}",
        f"--kt_epochs {config.kt_epochs}",
        f"--learning_rate {config.learning_rate}",
        f"--patience {config.patience}",
    ]
    
    # 添加布尔参数
    if config.with_pid:
        cmd_parts.append("--with_pid")
    if config.proj:
        cmd_parts.append("--proj")
    if config.hard_neg:
        cmd_parts.append("--hard_neg")
    if config.use_cl:
        cmd_parts.append("--use_cl")
    
    # 组合命令
    cmd = " ".join(cmd_parts)
    
    print("\n" + "=" * 60)
    print("执行训练命令:")
    print("=" * 60)
    print(cmd)
    print("=" * 60)
    
    # 执行训练
    print("\n开始训练...")
    exit_code = os.system(cmd)
    
    if exit_code == 0:
        print("\n✅ 训练完成!")
        print(f"结果保存在: {config.output_dir}")
    else:
        print("\n❌ 训练失败!")
        print(f"退出码: {exit_code}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ASSIST17基线训练')
    parser.add_argument('--dry_run', action='store_true', 
                       help='只打印命令，不执行训练')
    
    args = parser.parse_args()
    
    if args.dry_run:
        config = Assist17BaselineConfig()
        config.print_config()
        print("\n这是dry run模式，不会执行实际训练。")
    else:
        main()
