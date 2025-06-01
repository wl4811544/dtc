#!/usr/bin/env python3
"""
assist17数据集基线训练参数配置
针对大规模数据集的保守起点设置
"""

class Assist17BaselineConfig:
    def __init__(self):
        # ==================== 基本参数 ====================
        self.device = 'cuda'
        self.dataset = 'assist17'
        self.data_dir = 'data'
        self.output_dir = 'output/assist17_baseline'
        self.with_pid = True  # assist17有问题ID
        
        # ==================== 数据加载参数 ====================
        self.batch_size = 16          # 保守起点，避免内存问题
        self.test_batch_size = 32     # 测试时可以用更大batch
        
        # ==================== 模型架构参数 ====================
        # 保守的起点设置 - 可以稳定训练并观察效果
        self.d_model = 128            # 中等维度，平衡性能和效率
        self.n_heads = 8              # d_model的因子
        self.n_know = 16              # 知识概念数，可以后续调整
        self.n_layers = 3             # 3层Transformer
        self.dropout = 0.2            # 适中的dropout
        self.lambda_cl = 0.1          # 对比学习权重
        self.proj = True              # 使用投影层
        self.hard_neg = False         # 起点不用困难负样本
        self.window = 1               # 注意力窗口
        
        # ==================== 训练参数 ====================
        self.kt_epochs = 100          # 最大训练轮数
        self.learning_rate = 1e-3     # 标准学习率
        self.patience = 10            # 早停耐心值
        self.use_cl = True            # 大数据集可以使用对比学习
        
        # ==================== 数据集特定配置 ====================
        self.n_questions = 102        # assist17问题数
        self.n_pid = 3162            # assist17知识点数
        
    def to_dict(self):
        """转换为字典格式，方便传递给训练函数"""
        return {
            'device': self.device,
            'dataset': self.dataset,
            'data_dir': self.data_dir,
            'output_dir': self.output_dir,
            'with_pid': self.with_pid,
            'batch_size': self.batch_size,
            'test_batch_size': self.test_batch_size,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_know': self.n_know,
            'n_layers': self.n_layers,
            'dropout': self.dropout,
            'lambda_cl': self.lambda_cl,
            'proj': self.proj,
            'hard_neg': self.hard_neg,
            'window': self.window,
            'kt_epochs': self.kt_epochs,
            'learning_rate': self.learning_rate,
            'patience': self.patience,
            'use_cl': self.use_cl,
            'n_questions': self.n_questions,
            'n_pid': self.n_pid
        }
    
    def print_config(self):
        """打印配置信息"""
        print("=" * 60)
        print("ASSIST17 基线训练配置")
        print("=" * 60)
        print(f"数据集: {self.dataset}")
        print(f"设备: {self.device}")
        print(f"输出目录: {self.output_dir}")
        print(f"使用问题ID: {self.with_pid}")
        print()
        print("数据加载:")
        print(f"  训练批次大小: {self.batch_size}")
        print(f"  测试批次大小: {self.test_batch_size}")
        print()
        print("模型架构:")
        print(f"  模型维度: {self.d_model}")
        print(f"  注意力头数: {self.n_heads}")
        print(f"  知识概念数: {self.n_know}")
        print(f"  Transformer层数: {self.n_layers}")
        print(f"  Dropout率: {self.dropout}")
        print(f"  对比学习权重: {self.lambda_cl}")
        print(f"  投影层: {self.proj}")
        print(f"  困难负样本: {self.hard_neg}")
        print(f"  注意力窗口: {self.window}")
        print()
        print("训练参数:")
        print(f"  最大轮数: {self.kt_epochs}")
        print(f"  学习率: {self.learning_rate}")
        print(f"  早停耐心值: {self.patience}")
        print(f"  使用对比学习: {self.use_cl}")
        print()
        print("数据集信息:")
        print(f"  问题总数: {self.n_questions}")
        print(f"  知识点总数: {self.n_pid}")
        print("=" * 60)


# 使用示例
if __name__ == "__main__":
    config = Assist17BaselineConfig()
    config.print_config()
    
    # 获取配置字典
    config_dict = config.to_dict()
    print("\n配置字典格式:")
    for key, value in config_dict.items():
        print(f"  {key}: {value}")
