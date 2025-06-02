#!/usr/bin/env python
"""
Stage 1 测试脚本

验证第一阶段基线训练的功能正确性
"""

import os
import sys
import tempfile
import shutil
import unittest
from unittest.mock import patch, MagicMock

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import load_config, load_auto_config, merge_config_with_args
from core.common import setup_output_directory, save_config, print_stage_header
from stages.stage1_baseline import validate_stage1_parameters, print_stage1_parameters


class TestStage1Configuration(unittest.TestCase):
    """测试第一阶段配置功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, 'test_config.yaml')
        
        # 创建测试配置文件
        test_config = """
dataset: assist17
device: cuda
d_model: 128
n_heads: 8
n_layers: 3
dropout: 0.2
kt_epochs: 10
learning_rate: 0.001
patience: 5
"""
        with open(self.config_file, 'w') as f:
            f.write(test_config)
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.test_dir)
    
    def test_load_config(self):
        """测试配置文件加载"""
        config = load_config(self.config_file)
        
        self.assertEqual(config['dataset'], 'assist17')
        self.assertEqual(config['d_model'], 128)
        self.assertEqual(config['n_heads'], 8)
    
    def test_load_auto_config(self):
        """测试自动配置加载"""
        # 测试存在的配置
        config = load_auto_config('assist17', 'baseline')
        if config:  # 如果配置文件存在
            self.assertIn('dataset', config)
            self.assertEqual(config['dataset'], 'assist17')
        
        # 测试不存在的配置
        config = load_auto_config('nonexistent', 'baseline')
        self.assertIsNone(config)
    
    def test_merge_config_with_args(self):
        """测试配置合并"""
        config = {'d_model': 256, 'n_heads': 16, 'learning_rate': 0.001}
        
        # 模拟命令行参数
        class MockArgs:
            def __init__(self):
                self.d_model = None  # 未设置，应该使用配置文件值
                self.n_heads = 8     # 已设置，应该保持命令行值
                self.learning_rate = None
        
        args = MockArgs()
        merge_config_with_args(config, args)
        
        self.assertEqual(args.d_model, 256)  # 来自配置文件
        self.assertEqual(args.n_heads, 8)    # 来自命令行
        self.assertEqual(args.learning_rate, 0.001)  # 来自配置文件


class TestStage1Common(unittest.TestCase):
    """测试第一阶段通用功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.test_dir)
    
    def test_setup_output_directory(self):
        """测试输出目录设置"""
        # 测试自动生成目录
        output_dir = setup_output_directory(None, 'assist17', 'stage1')
        self.assertTrue(os.path.exists(output_dir))
        self.assertIn('stage1_assist17', output_dir)
        
        # 测试指定目录
        custom_dir = os.path.join(self.test_dir, 'custom_output')
        output_dir = setup_output_directory(custom_dir, 'assist17', 'stage1')
        self.assertEqual(output_dir, custom_dir)
        self.assertTrue(os.path.exists(output_dir))
    
    def test_save_config(self):
        """测试配置保存"""
        config = {
            'dataset': 'assist17',
            'd_model': 128,
            'learning_rate': 0.001
        }
        
        config_path = save_config(config, self.test_dir)
        self.assertTrue(os.path.exists(config_path))
        
        # 验证保存的配置
        import yaml
        with open(config_path, 'r') as f:
            saved_config = yaml.safe_load(f)
        
        self.assertEqual(saved_config['dataset'], 'assist17')
        self.assertEqual(saved_config['d_model'], 128)
    
    def test_print_stage_header(self):
        """测试阶段标题打印"""
        # 这个测试主要确保函数不会抛出异常
        try:
            print_stage_header("测试阶段", 1)
            print_stage_header("测试阶段")
        except Exception as e:
            self.fail(f"print_stage_header raised {e} unexpectedly!")


class TestStage1Validation(unittest.TestCase):
    """测试第一阶段参数验证"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建模拟参数对象
        class MockArgs:
            def __init__(self):
                self.device = 'cuda'
                self.output_dir = '/tmp/test'
                self.d_model = 128
                self.n_heads = 8
                self.n_know = 16
                self.n_layers = 3
                self.dropout = 0.2
                self.lambda_cl = 0.1
                self.window = 1
                self.kt_epochs = 100
                self.learning_rate = 0.001
                self.patience = 10
                self.with_pid = True
                self.proj = False
                self.hard_neg = False
                self.use_cl = False
        
        self.valid_args = MockArgs()
        self.dataset_config = {
            'n_questions': 1000,
            'n_pid': 500
        }
    
    def test_valid_parameters(self):
        """测试有效参数验证"""
        try:
            validate_stage1_parameters(self.valid_args, self.dataset_config)
        except Exception as e:
            self.fail(f"validate_stage1_parameters raised {e} unexpectedly!")
    
    def test_missing_basic_parameters(self):
        """测试缺少基本参数"""
        # 删除必需的基本参数
        delattr(self.valid_args, 'device')
        
        with self.assertRaises(ValueError) as context:
            validate_stage1_parameters(self.valid_args, self.dataset_config)
        
        self.assertIn('device', str(context.exception))
    
    def test_missing_model_parameters(self):
        """测试缺少模型参数"""
        # 删除必需的模型参数
        delattr(self.valid_args, 'd_model')
        
        with self.assertRaises(ValueError) as context:
            validate_stage1_parameters(self.valid_args, self.dataset_config)
        
        self.assertIn('d_model', str(context.exception))
    
    def test_missing_dataset_config(self):
        """测试缺少数据集配置"""
        # 删除必需的数据集参数
        invalid_dataset_config = {'n_questions': 1000}  # 缺少n_pid
        
        with self.assertRaises(ValueError) as context:
            validate_stage1_parameters(self.valid_args, invalid_dataset_config)
        
        self.assertIn('n_pid', str(context.exception))
    
    def test_optional_parameters_defaults(self):
        """测试可选参数的默认值设置"""
        # 删除可选参数
        delattr(self.valid_args, 'with_pid')
        delattr(self.valid_args, 'proj')
        
        validate_stage1_parameters(self.valid_args, self.dataset_config)
        
        # 验证默认值被设置
        self.assertEqual(self.valid_args.with_pid, False)
        self.assertEqual(self.valid_args.proj, False)
    
    def test_print_parameters(self):
        """测试参数打印功能"""
        try:
            print_stage1_parameters(self.valid_args, self.dataset_config)
        except Exception as e:
            self.fail(f"print_stage1_parameters raised {e} unexpectedly!")


class TestStage1Integration(unittest.TestCase):
    """测试第一阶段集成功能"""
    
    @patch('DTransformer.eval.Evaluator')
    @patch('DTransformer.model.DTransformer')
    def test_train_baseline_model_mock(self, mock_dtransformer, mock_evaluator):
        """测试基线模型训练（使用模拟对象）"""
        from stages.stage1_baseline import train_baseline_model
        
        # 设置模拟对象
        mock_model = MagicMock()
        mock_model.get_loss.return_value = torch.tensor(0.5)
        mock_model.get_cl_loss.return_value = (torch.tensor(0.5), torch.tensor(0.4), torch.tensor(0.1))
        mock_model.parameters.return_value = [torch.tensor([1.0])]
        mock_dtransformer.return_value = mock_model

        mock_evaluator_instance = MagicMock()
        mock_evaluator_instance.evaluate.return_value = {'auc': 0.75, 'acc': 0.70}
        mock_evaluator.return_value = mock_evaluator_instance
        
        # 创建测试参数
        class MockArgs:
            def __init__(self):
                self.device = 'cpu'
                self.output_dir = tempfile.mkdtemp()
                self.d_model = 64
                self.n_heads = 4
                self.n_know = 8
                self.n_layers = 2
                self.dropout = 0.2
                self.lambda_cl = 0.1
                self.window = 1
                self.kt_epochs = 5
                self.learning_rate = 0.001
                self.patience = 3
                self.with_pid = True
                self.proj = False
                self.hard_neg = False
                self.use_cl = False
        
        args = MockArgs()
        dataset_config = {'n_questions': 100, 'n_pid': 50}
        
        # 模拟数据加载器
        train_data = MagicMock()
        val_data = MagicMock()
        
        try:
            model_path = train_baseline_model(args, dataset_config, train_data, val_data)
            
            # 验证模型创建
            mock_dtransformer.assert_called_once()

            # 验证评估器创建
            mock_evaluator.assert_called_once()

            # 验证返回的模型路径
            self.assertIn('best_model.pt', model_path)
            
        finally:
            # 清理临时目录
            shutil.rmtree(args.output_dir)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestStage1Configuration,
        TestStage1Common,
        TestStage1Validation,
        TestStage1Integration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("="*60)
    print("Stage 1 功能测试")
    print("="*60)
    
    success = run_tests()
    
    if success:
        print("\n✅ 所有测试通过！")
        sys.exit(0)
    else:
        print("\n❌ 部分测试失败！")
        sys.exit(1)
