# check_checkpoint.py
import torch

# 检查异常感知模型的checkpoint
checkpoint_path = "output/assist17_20250525_000228/anomaly_aware/best_model.pt"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("Checkpoint contents:")
for key in checkpoint.keys():
    if key == 'metrics':
        print(f"  {key}: {checkpoint[key]}")
    elif key == 'epoch':
        print(f"  {key}: {checkpoint[key]}")
    elif key == 'model_state_dict':
        print(f"  {key}: {len(checkpoint[key])} parameters")
    else:
        print(f"  {key}: ...")

# 查看训练的最佳指标
if 'metrics' in checkpoint:
    print("\nBest metrics from training:")
    for metric, value in checkpoint['metrics'].items():
        print(f"  {metric}: {value}")