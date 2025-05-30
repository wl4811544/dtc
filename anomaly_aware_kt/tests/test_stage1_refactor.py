#!/usr/bin/env python
"""
测试第一阶段重构的脚本

这个脚本验证重构后的第一阶段代码是否正确工作，
包括配置创建、模型初始化等功能。
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from anomaly_kt.stages.common import StageConfig, BaseStage, prepare_data, setup_output_directory
    from anomaly_kt.stages.stage1_baseline import BaselineConfig, BaselineTrainer
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在正确的目录中运行此脚本")
    sys.exit(1)


class MockArgs:
    """模拟命令行参数"""
    def __init__(self):
        # 基本参数
        self.dataset = 'assist09'
        self.data_dir = 'data'
        self.output_dir = 'test_output'
        self.device = 'cpu'
        self.with_pid = False
        
        # 数据参数
        self.batch_size = 32
        self.test_batch_size = 64
        
        # 基线模型参数
        self.d_model = 128
        self.n_heads = 8
        self.n_know = 16
        self.n_layers = 3
        self.dropout = 0.2
        self.lambda_cl = 0.1
        self.proj = False
        self.hard_neg = False
        self.window = 1
        
        # 训练参数
        self.kt_epochs = 10  # 测试时使用较少的轮数
        self.learning_rate = 1e-3
        self.patience = 5
        self.use_cl = False


class MockDatasetConfig:
    """模拟数据集配置"""
    def __init__(self):
        self.config = {
            'n_questions': 100,
            'n_pid': 50
        }
    
    def __getitem__(self, key):
        return self.config[key]


class TestStage1Refactor(unittest.TestCase):
    """测试第一阶段重构"""
    
    def setUp(self):
        """设置测试环境"""
        self.args = MockArgs()
        self.dataset_config = MockDatasetConfig()
        
    def test_baseline_config_creation(self):
        """测试基线配置创建"""
        config = BaselineConfig(self.args, self.dataset_config)
        
        # 验证基本属性
        self.assertEqual(config.device, 'cpu')
        self.assertEqual(config.d_model, 128)
        self.assertEqual(config.n_heads, 8)
        self.assertEqual(config.epochs, 10)
        
        # 验证模型参数
        model_params = config.get_model_params()
        self.assertEqual(model_params['n_questions'], 100)
        self.assertEqual(model_params['d_model'], 128)
        self.assertEqual(model_params['n_heads'], 8)
        
        # 验证训练参数
        training_params = config.get_training_params()
        self.assertEqual(training_params['epochs'], 10)
        self.assertEqual(training_params['learning_rate'], 1e-3)
        
        print("✓ 基线配置创建测试通过")
        
    def test_baseline_trainer_creation(self):
        """测试基线训练器创建"""
        config = BaselineConfig(self.args, self.dataset_config)
        trainer = BaselineTrainer(config)
        
        # 验证基本属性
        self.assertEqual(trainer.device, 'cpu')
        self.assertEqual(trainer.output_dir, 'test_output')
        self.assertIsInstance(trainer.config, BaselineConfig)
        
        print("✓ 基线训练器创建测试通过")
        
    @patch('anomaly_kt.stages.stage1_baseline.DTransformer')
    def test_model_creation(self, mock_dtransformer):
        """测试模型创建"""
        # 设置mock
        mock_model = Mock()
        mock_model.parameters.return_value = [Mock(numel=lambda: 1000)]
        mock_dtransformer.return_value = mock_model
        
        config = BaselineConfig(self.args, self.dataset_config)
        trainer = BaselineTrainer(config)
        
        # 创建模型
        model = trainer.create_model()
        
        # 验证模型创建
        self.assertIsNotNone(model)
        mock_dtransformer.assert_called_once()
        
        # 验证调用参数
        call_args = mock_dtransformer.call_args[1]
        self.assertEqual(call_args['n_questions'], 100)
        self.assertEqual(call_args['d_model'], 128)
        self.assertEqual(call_args['n_heads'], 8)
        
        print("✓ 模型创建测试通过")
        
    @patch('anomaly_kt.stages.stage1_baseline.KnowledgeTracingTrainer')
    @patch('anomaly_kt.stages.stage1_baseline.DTransformer')
    def test_trainer_creation(self, mock_dtransformer, mock_kt_trainer):
        """测试训练器创建"""
        # 设置mock
        mock_model = Mock()
        mock_model.parameters.return_value = [Mock(numel=lambda: 1000)]
        mock_dtransformer.return_value = mock_model
        
        mock_trainer_instance = Mock()
        mock_kt_trainer.return_value = mock_trainer_instance
        
        config = BaselineConfig(self.args, self.dataset_config)
        trainer = BaselineTrainer(config)
        
        # 创建模型和训练器
        model = trainer.create_model()
        kt_trainer = trainer.create_trainer(model)
        
        # 验证训练器创建
        self.assertIsNotNone(kt_trainer)
        mock_kt_trainer.assert_called_once()
        
        # 验证调用参数
        call_args = mock_kt_trainer.call_args[1]
        self.assertEqual(call_args['device'], 'cpu')
        self.assertEqual(call_args['patience'], 5)
        self.assertTrue(call_args['save_dir'].endswith('baseline'))
        
        print("✓ 训练器创建测试通过")
        
    def test_stage_config_inheritance(self):
        """测试阶段配置继承"""
        config = BaselineConfig(self.args, self.dataset_config)
        
        # 验证继承的方法
        self.assertTrue(hasattr(config, 'get_model_save_path'))
        
        # 测试模型保存路径
        save_path = config.get_model_save_path('baseline')
        self.assertTrue(save_path.endswith('baseline/best_model.pt'))
        
        print("✓ 阶段配置继承测试通过")
        
    def test_common_utilities(self):
        """测试通用工具函数"""
        # 测试输出目录设置
        output_dir = setup_output_directory(None, 'test_dataset')
        self.assertTrue('test_dataset' in output_dir)
        
        # 测试指定输出目录
        specified_dir = setup_output_directory('custom_output', 'test_dataset')
        self.assertEqual(specified_dir, 'custom_output')
        
        print("✓ 通用工具函数测试通过")


def run_integration_test():
    """运行集成测试"""
    print("\n" + "="*50)
    print("集成测试：完整流程验证")
    print("="*50)
    
    try:
        # 创建mock参数
        args = MockArgs()
        dataset_config = MockDatasetConfig()
        
        # 创建配置
        config = BaselineConfig(args, dataset_config)
        print("✓ 配置创建成功")
        
        # 创建训练器
        trainer = BaselineTrainer(config)
        print("✓ 训练器创建成功")
        
        # 验证配置参数
        model_params = config.get_model_params()
        training_params = config.get_training_params()
        
        print(f"✓ 模型参数: {len(model_params)} 个")
        print(f"✓ 训练参数: {len(training_params)} 个")
        
        print("\n🎉 集成测试通过！重构的第一阶段代码结构正确。")
        
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    print("="*60)
    print("第一阶段重构测试")
    print("="*60)
    
    # 运行单元测试
    print("\n运行单元测试...")
    unittest.main(argv=[''], exit=False, verbosity=0)
    
    # 运行集成测试
    run_integration_test()
    
    print("\n" + "="*60)
    print("所有测试完成")
    print("="*60)
