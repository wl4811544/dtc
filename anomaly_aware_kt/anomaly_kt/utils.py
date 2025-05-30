# anomaly_kt/utils.py
"""
工具函数模块
"""

import torch
import numpy as np
import random
import os


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state, filename):
    """保存检查点"""
    torch.save(state, filename)


def load_checkpoint(filename, device='cpu'):
    """加载检查点"""
    return torch.load(filename, map_location=device)