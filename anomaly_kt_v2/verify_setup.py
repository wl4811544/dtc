#!/usr/bin/env python
"""
验证安装和设置脚本

检查所有依赖是否正确安装，以及DTransformer项目是否可用
"""

import sys
import os
import importlib

def check_python_version():
    """检查Python版本"""
    print("🐍 检查Python版本...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ❌ Python版本过低: {version.major}.{version.minor}.{version.micro}")
        print("  需要Python 3.8+")
        return False

def check_package(package_name, import_name=None):
    """检查包是否安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        if hasattr(module, '__version__'):
            version = module.__version__
        else:
            version = "unknown"
        print(f"  ✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"  ❌ {package_name}: 未安装")
        return False

def check_dependencies():
    """检查依赖包"""
    print("\n📦 检查依赖包...")
    
    packages = [
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("PyYAML", "yaml"),
        ("tomlkit", "tomlkit"),
        ("tqdm", "tqdm")
    ]
    
    all_ok = True
    for package_name, import_name in packages:
        if not check_package(package_name, import_name):
            all_ok = False
    
    return all_ok

def check_dtransformer():
    """检查DTransformer项目"""
    print("\n🔧 检查DTransformer项目...")
    
    # 检查DTransformer目录
    dtransformer_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'DTransformer')
    if not os.path.exists(dtransformer_path):
        print(f"  ❌ DTransformer目录不存在: {dtransformer_path}")
        return False
    
    print(f"  ✅ DTransformer目录: {dtransformer_path}")
    
    # 检查关键文件
    key_files = ['model.py', 'data.py', 'eval.py']
    for file_name in key_files:
        file_path = os.path.join(dtransformer_path, file_name)
        if os.path.exists(file_path):
            print(f"  ✅ {file_name}")
        else:
            print(f"  ❌ {file_name}: 文件不存在")
            return False
    
    # 尝试导入DTransformer模块
    try:
        sys.path.append(os.path.dirname(dtransformer_path))
        from DTransformer.model import DTransformer
        from DTransformer.data import KTData
        from DTransformer.eval import Evaluator
        print("  ✅ DTransformer模块导入成功")
        return True
    except ImportError as e:
        print(f"  ❌ DTransformer模块导入失败: {e}")
        return False

def check_data_directory():
    """检查数据目录"""
    print("\n📊 检查数据目录...")
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    if not os.path.exists(data_dir):
        print(f"  ⚠️ 数据目录不存在: {data_dir}")
        print("  请确保数据目录存在并包含数据集文件")
        return False
    
    print(f"  ✅ 数据目录: {data_dir}")
    
    # 检查datasets.toml
    datasets_file = os.path.join(data_dir, 'datasets.toml')
    if os.path.exists(datasets_file):
        print("  ✅ datasets.toml")
        return True
    else:
        print("  ⚠️ datasets.toml: 文件不存在")
        return False

def check_cuda():
    """检查CUDA可用性"""
    print("\n🚀 检查CUDA...")
    
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            print(f"  ✅ CUDA可用")
            print(f"  📱 设备数量: {device_count}")
            print(f"  🎯 当前设备: {current_device} ({device_name})")
            return True
        else:
            print("  ⚠️ CUDA不可用，将使用CPU")
            return False
    except Exception as e:
        print(f"  ❌ CUDA检查失败: {e}")
        return False

def check_project_structure():
    """检查项目结构"""
    print("\n🏗️ 检查项目结构...")
    
    project_root = os.path.dirname(__file__)
    required_dirs = ['configs', 'core', 'stages', 'scripts', 'tests']
    required_files = ['__init__.py']
    
    all_ok = True
    
    # 检查目录
    for dir_name in required_dirs:
        dir_path = os.path.join(project_root, dir_name)
        if os.path.exists(dir_path):
            print(f"  ✅ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/: 目录不存在")
            all_ok = False
    
    # 检查文件
    for file_name in required_files:
        file_path = os.path.join(project_root, file_name)
        if os.path.exists(file_path):
            print(f"  ✅ {file_name}")
        else:
            print(f"  ❌ {file_name}: 文件不存在")
            all_ok = False
    
    return all_ok

def main():
    """主函数"""
    print("="*60)
    print("🔍 Anomaly-Aware Knowledge Tracing v2 环境验证")
    print("="*60)
    
    checks = [
        ("Python版本", check_python_version),
        ("依赖包", check_dependencies),
        ("DTransformer项目", check_dtransformer),
        ("数据目录", check_data_directory),
        ("CUDA支持", check_cuda),
        ("项目结构", check_project_structure)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"  ❌ {check_name}检查失败: {e}")
            results.append((check_name, False))
    
    # 总结
    print("\n" + "="*60)
    print("📋 验证总结")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for check_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {check_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📊 总体结果: {passed}/{total} 项检查通过")
    
    if passed == total:
        print("\n🎉 所有检查通过！环境配置正确。")
        print("\n💡 下一步:")
        print("  python scripts/run_stage1_baseline.py --dataset assist17 --auto_config --device cuda")
        return True
    else:
        print("\n⚠️ 部分检查失败，请根据上述信息修复问题。")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
