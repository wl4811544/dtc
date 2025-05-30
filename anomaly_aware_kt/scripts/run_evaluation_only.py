#!/usr/bin/env python
"""
直接执行第4阶段评估的示例脚本

这个脚本展示了如何使用已训练好的模型直接执行第4阶段（评估阶段），
跳过前面的训练阶段。

使用方法:
python run_evaluation_only.py --dataset assist09 \
    --baseline_path path/to/baseline_model.pt \
    --detector_path path/to/detector_model.pt \
    --anomaly_path path/to/anomaly_aware_model.pt \
    --skip_baseline --skip_detector --skip_anomaly_training
"""

import os
import sys
import subprocess

def main():
    # 示例参数 - 请根据实际情况修改这些路径
    dataset = "assist09"
    
    # 模型路径 - 请替换为您的实际模型路径
    baseline_model_path = "output/assist09_20241201_120000/baseline/best_model.pt"
    detector_model_path = "output/assist09_20241201_120000/detector/best_model.pt"
    anomaly_model_path = "output/assist09_20241201_120000/anomaly_aware/best_model.pt"
    
    # 输出目录
    output_dir = "output/evaluation_only"
    
    # 构建命令
    cmd = [
        "python", "full_pipeline.py",
        "--dataset", dataset,
        "--skip_baseline",
        "--skip_detector", 
        "--skip_anomaly_training",
        "--baseline_path", baseline_model_path,
        "--detector_path", detector_model_path,
        "--anomaly_path", anomaly_model_path,
        "--output_dir", output_dir,
        # 其他必要参数
        "--d_model", "128",
        "--n_heads", "8",
        "--n_know", "16",
        "--n_layers", "3",
        "--detector_d_model", "128",
        "--detector_n_heads", "8",
        "--detector_n_layers", "2",
        "--window_size", "10",
        "--anomaly_weight", "0.5"
    ]
    
    print("执行第4阶段评估...")
    print("命令:", " ".join(cmd))
    print()
    
    # 检查模型文件是否存在
    model_files = [
        ("基线模型", baseline_model_path),
        ("异常检测器", detector_model_path),
        ("异常感知模型", anomaly_model_path)
    ]
    
    missing_files = []
    for name, path in model_files:
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
        else:
            print(f"✓ 找到{name}: {path}")
    
    if missing_files:
        print("\n❌ 以下模型文件不存在:")
        for missing in missing_files:
            print(f"  - {missing}")
        print("\n请确保所有模型文件存在，或修改脚本中的路径。")
        return
    
    print("\n开始执行评估...")
    
    # 执行命令
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ 评估完成!")
        print("\n输出:")
        print(result.stdout)
        if result.stderr:
            print("\n错误信息:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"❌ 执行失败: {e}")
        print(f"返回码: {e.returncode}")
        if e.stdout:
            print(f"标准输出: {e.stdout}")
        if e.stderr:
            print(f"错误输出: {e.stderr}")
    except FileNotFoundError:
        print("❌ 找不到 full_pipeline.py 文件")
        print("请确保在正确的目录中运行此脚本")

if __name__ == "__main__":
    print("="*60)
    print("直接执行第4阶段评估")
    print("="*60)
    print()
    print("注意: 请在运行前修改脚本中的模型路径!")
    print()
    
    # 询问用户是否继续
    response = input("是否继续执行? (y/N): ").strip().lower()
    if response in ['y', 'yes']:
        main()
    else:
        print("已取消执行")
