#!/usr/bin/env python3
"""
测试第二阶段组件的快速脚本

用于验证课程学习组件是否正常工作
"""

import os
import sys
import torch

# 添加项目路径
sys.path.append('anomaly_aware_kt')

def test_imports():
    """测试导入"""
    print("🧪 测试组件导入...")
    
    try:
        from anomaly_kt.curriculum_learning import (
            BaselineAnomalyGenerator,
            CurriculumAnomalyGenerator,
            DifficultyEstimator,
            CurriculumScheduler
        )
        print("✅ 课程学习组件导入成功")
        
        from anomaly_kt.stages import (
            train_curriculum_anomaly_detector,
            test_curriculum_components
        )
        print("✅ 第二阶段模块导入成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False


def test_basic_functionality():
    """测试基本功能"""
    print("\n🔧 测试基本功能...")
    
    try:
        from anomaly_kt.curriculum_learning import BaselineAnomalyGenerator
        
        # 创建模拟数据
        batch_size, seq_len = 2, 10
        q = torch.randint(0, 100, (batch_size, seq_len))
        s = torch.randint(0, 2, (batch_size, seq_len))
        
        # 测试基线异常生成器
        generator = BaselineAnomalyGenerator()
        s_anomaly, anomaly_labels = generator.generate_baseline_anomalies(
            q, s, strategy='random_flip', anomaly_ratio=0.2
        )
        
        print(f"✅ 基线异常生成: 输入 {s.shape}, 输出 {s_anomaly.shape}")
        print(f"   异常数量: {anomaly_labels.sum().item()}")
        
        return True
        
    except Exception as e:
        print(f"❌ 功能测试失败: {e}")
        return False


def test_stage2_script():
    """测试第二阶段脚本"""
    print("\n📜 测试第二阶段脚本...")
    
    script_path = "anomaly_aware_kt/scripts/run_stage2_curriculum.py"
    
    if not os.path.exists(script_path):
        print(f"❌ 脚本文件不存在: {script_path}")
        return False
    
    print(f"✅ 脚本文件存在: {script_path}")
    
    # 测试帮助信息
    import subprocess
    try:
        result = subprocess.run([
            sys.executable, script_path, "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ 脚本帮助信息正常")
            return True
        else:
            print(f"❌ 脚本执行失败: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 脚本执行超时")
        return False
    except Exception as e:
        print(f"❌ 脚本测试失败: {e}")
        return False


def test_config_file():
    """测试配置文件"""
    print("\n📄 测试配置文件...")
    
    config_path = "anomaly_aware_kt/configs/assist17_curriculum.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 检查关键配置项
        required_keys = [
            'curriculum_strategy', 'curriculum_epochs', 
            'anomaly_ratio', 'baseline_ratio',
            'detector_hidden_dim', 'learning_rate'
        ]
        
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            print(f"❌ 配置文件缺少关键项: {missing_keys}")
            return False
        
        print("✅ 配置文件格式正确，包含所有必需参数")
        print(f"   策略: {config['curriculum_strategy']}")
        print(f"   轮数: {config['curriculum_epochs']}")
        print(f"   异常比例: {config['anomaly_ratio']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置文件测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🎓 第二阶段组件测试")
    print("=" * 50)
    
    tests = [
        ("导入测试", test_imports),
        ("功能测试", test_basic_functionality),
        ("脚本测试", test_stage2_script),
        ("配置测试", test_config_file)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        success = test_func()
        results.append((test_name, success))
    
    # 总结
    print("\n" + "=" * 50)
    print("📊 测试总结:")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！可以开始使用第二阶段训练")
        print("\n💡 使用示例:")
        print("python anomaly_aware_kt/scripts/run_stage2_curriculum.py \\")
        print("  --dataset assist17 \\")
        print("  --baseline_model_path output/stage1_xxx/baseline/best_model.pt \\")
        print("  --auto_config \\")
        print("  --dry_run")
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查环境配置")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
