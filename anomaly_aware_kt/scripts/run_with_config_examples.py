#!/usr/bin/env python3
"""
配置文件使用示例脚本
展示如何使用不同的配置文件运行训练
"""

import os
import subprocess
import sys

def run_command(cmd, description):
    """运行命令并打印描述"""
    print("=" * 80)
    print(f"🚀 {description}")
    print("=" * 80)
    print(f"命令: {cmd}")
    print("-" * 80)
    
    # 询问用户是否执行
    response = input("是否执行此命令? (y/n/q): ").strip().lower()
    if response == 'q':
        print("退出脚本")
        sys.exit(0)
    elif response == 'y':
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"❌ 命令执行失败，退出码: {result.returncode}")
        else:
            print("✅ 命令执行成功")
    else:
        print("⏭️  跳过此命令")
    print()

def main():
    """主函数"""
    print("📚 配置文件使用示例")
    print("这个脚本展示了如何使用不同方式运行训练")
    print()

    # 获取脚本路径
    script_path = "anomaly_aware_kt/scripts/run_stage1_only.py"
    
    examples = [
        {
            "cmd": f"python {script_path} --dataset assist17 --auto_config",
            "desc": "示例1: 使用自动配置文件 (assist17_baseline.yaml)"
        },
        {
            "cmd": f"python {script_path} --dataset assist17 --config anomaly_aware_kt/configs/assist17_baseline.yaml",
            "desc": "示例2: 手动指定配置文件"
        },
        {
            "cmd": f"python {script_path} --dataset assist17 --auto_config --d_model 256 --n_heads 16",
            "desc": "示例3: 配置文件 + 命令行参数覆盖 (d_model和n_heads会覆盖配置文件)"
        },
        {
            "cmd": f"python {script_path} --dataset assist17 --auto_config --kt_epochs 50 --learning_rate 0.0005",
            "desc": "示例4: 配置文件 + 训练参数覆盖"
        },
        {
            "cmd": f"python {script_path} --dataset statics --auto_config",
            "desc": "示例5: 小数据集配置 (statics_baseline.yaml)"
        },
        {
            "cmd": f"python {script_path} --dataset assist09 --auto_config --batch_size 32",
            "desc": "示例6: 中等数据集 + 批次大小覆盖"
        },
        {
            "cmd": f"python {script_path} --dataset algebra05 --auto_config --use_cl",
            "desc": "示例7: 小数据集 + 强制启用对比学习"
        }
    ]

    print("可用的示例命令:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['desc']}")
    print()

    # 让用户选择
    while True:
        try:
            choice = input("请选择要运行的示例 (1-7, 'a'=全部, 'q'=退出): ").strip().lower()
            
            if choice == 'q':
                print("退出脚本")
                break
            elif choice == 'a':
                for example in examples:
                    run_command(example['cmd'], example['desc'])
                break
            else:
                idx = int(choice) - 1
                if 0 <= idx < len(examples):
                    example = examples[idx]
                    run_command(example['cmd'], example['desc'])
                    break
                else:
                    print("无效选择，请重试")
        except ValueError:
            print("无效输入，请重试")
        except KeyboardInterrupt:
            print("\n用户中断，退出")
            break

if __name__ == "__main__":
    main()
