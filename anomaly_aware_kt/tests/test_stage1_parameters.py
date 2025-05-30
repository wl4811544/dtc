#!/usr/bin/env python
"""
测试第一阶段参数验证功能

这个脚本测试参数验证和打印功能是否正常工作。
"""

import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly_kt.stages.stage1_baseline import validate_stage1_parameters, print_stage1_parameters


class MockArgs:
    """模拟完整的参数对象"""
    def __init__(self, missing_params=None):
        # 基本参数
        self.device = 'cpu'
        self.output_dir = 'test_output'
        self.with_pid = True
        
        # 模型参数
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
        self.kt_epochs = 50
        self.learning_rate = 1e-3
        self.patience = 10
        self.use_cl = True
        
        # 删除指定的参数（用于测试缺失参数的情况）
        if missing_params:
            for param in missing_params:
                if hasattr(self, param):
                    delattr(self, param)


class MockDatasetConfig:
    """模拟数据集配置"""
    def __init__(self, missing_params=None):
        self.config = {
            'n_questions': 100,
            'n_pid': 50
        }
        
        # 删除指定的参数（用于测试缺失参数的情况）
        if missing_params:
            for param in missing_params:
                if param in self.config:
                    del self.config[param]
    
    def __getitem__(self, key):
        return self.config[key]
    
    def __contains__(self, key):
        return key in self.config


def test_complete_parameters():
    """测试完整参数的情况"""
    print("="*50)
    print("测试1: 完整参数验证")
    print("="*50)
    
    try:
        args = MockArgs()
        dataset_config = MockDatasetConfig()
        
        # 验证参数
        validate_stage1_parameters(args, dataset_config)
        
        # 打印参数
        print_stage1_parameters(args, dataset_config)
        
        print("\n✅ 完整参数测试通过")
        return True
        
    except Exception as e:
        print(f"\n❌ 完整参数测试失败: {e}")
        return False


def test_missing_basic_parameters():
    """测试缺失基本参数的情况"""
    print("\n" + "="*50)
    print("测试2: 缺失基本参数")
    print("="*50)
    
    missing_params = ['device', 'output_dir']
    
    for param in missing_params:
        try:
            args = MockArgs(missing_params=[param])
            dataset_config = MockDatasetConfig()
            
            validate_stage1_parameters(args, dataset_config)
            print(f"❌ 应该检测到缺失参数 {param}")
            return False
            
        except ValueError as e:
            print(f"✅ 正确检测到缺失参数 {param}: {e}")
        except Exception as e:
            print(f"❌ 意外错误: {e}")
            return False
    
    return True


def test_missing_model_parameters():
    """测试缺失模型参数的情况"""
    print("\n" + "="*50)
    print("测试3: 缺失模型参数")
    print("="*50)
    
    missing_params = ['d_model', 'n_heads', 'n_layers']
    
    for param in missing_params:
        try:
            args = MockArgs(missing_params=[param])
            dataset_config = MockDatasetConfig()
            
            validate_stage1_parameters(args, dataset_config)
            print(f"❌ 应该检测到缺失模型参数 {param}")
            return False
            
        except ValueError as e:
            print(f"✅ 正确检测到缺失模型参数 {param}: {e}")
        except Exception as e:
            print(f"❌ 意外错误: {e}")
            return False
    
    return True


def test_missing_dataset_parameters():
    """测试缺失数据集参数的情况"""
    print("\n" + "="*50)
    print("测试4: 缺失数据集参数")
    print("="*50)
    
    missing_params = ['n_questions', 'n_pid']
    
    for param in missing_params:
        try:
            args = MockArgs()
            dataset_config = MockDatasetConfig(missing_params=[param])
            
            validate_stage1_parameters(args, dataset_config)
            print(f"❌ 应该检测到缺失数据集参数 {param}")
            return False
            
        except ValueError as e:
            print(f"✅ 正确检测到缺失数据集参数 {param}: {e}")
        except Exception as e:
            print(f"❌ 意外错误: {e}")
            return False
    
    return True


def test_optional_parameters():
    """测试可选参数的默认值设置"""
    print("\n" + "="*50)
    print("测试5: 可选参数默认值")
    print("="*50)
    
    try:
        # 创建缺少可选参数的args对象
        args = MockArgs(missing_params=['with_pid', 'proj', 'hard_neg', 'use_cl'])
        dataset_config = MockDatasetConfig()
        
        # 验证参数（应该自动设置默认值）
        validate_stage1_parameters(args, dataset_config)
        
        # 检查默认值是否正确设置
        assert args.with_pid == False, "with_pid 默认值应该是 False"
        assert args.proj == False, "proj 默认值应该是 False"
        assert args.hard_neg == False, "hard_neg 默认值应该是 False"
        assert args.use_cl == False, "use_cl 默认值应该是 False"
        
        print("✅ 可选参数默认值设置正确")
        print(f"  with_pid: {args.with_pid}")
        print(f"  proj: {args.proj}")
        print(f"  hard_neg: {args.hard_neg}")
        print(f"  use_cl: {args.use_cl}")
        
        return True
        
    except Exception as e:
        print(f"❌ 可选参数测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("第一阶段参数验证测试")
    print("="*60)
    
    tests = [
        test_complete_parameters,
        test_missing_basic_parameters,
        test_missing_model_parameters,
        test_missing_dataset_parameters,
        test_optional_parameters
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "="*60)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！第一阶段参数验证功能正常。")
    else:
        print("❌ 部分测试失败，请检查代码。")
    
    print("="*60)


if __name__ == '__main__':
    main()
