#!/usr/bin/env python3
"""
参数一致性检查工具

确保第一阶段、第二阶段、第三阶段的关键参数保持一致，
以保证最终对比评估的公平性。
"""

import yaml
import os
from typing import Dict, List, Tuple


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    if not os.path.exists(config_path):
        return {}
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_critical_parameters(config: Dict, stage: str) -> Dict:
    """提取关键参数"""
    critical_params = {}
    
    if stage == "stage1":
        # 第一阶段关键参数
        critical_params = {
            'batch_size': config.get('batch_size', None),
            'learning_rate': config.get('learning_rate', None),
            'd_model': config.get('d_model', None),
            'n_heads': config.get('n_heads', None),
            'n_layers': config.get('n_layers', None),
            'dropout': config.get('dropout', None),
            'lambda_cl': config.get('lambda_cl', None),
            'with_pid': config.get('with_pid', None),
            'n_know': config.get('n_know', None),
        }
    
    elif stage == "stage2":
        # 第二阶段关键参数（需要与第一阶段一致的部分）
        critical_params = {
            'batch_size': config.get('batch_size', config.get('training', {}).get('batch_size', None)),
            'learning_rate': config.get('learning_rate', config.get('training', {}).get('learning_rate', None)),
            # 第二阶段通过命令行参数传递第一阶段的模型参数
            # 这些参数在配置文件中可能不存在，但在脚本中有默认值
        }
    
    # 移除None值
    return {k: v for k, v in critical_params.items() if v is not None}


def check_consistency(stage1_config: Dict, stage2_config: Dict) -> Tuple[bool, List[str]]:
    """检查参数一致性"""
    stage1_params = extract_critical_parameters(stage1_config, "stage1")
    stage2_params = extract_critical_parameters(stage2_config, "stage2")
    
    inconsistencies = []
    
    # 检查共同参数
    common_params = set(stage1_params.keys()) & set(stage2_params.keys())
    
    for param in common_params:
        val1 = stage1_params[param]
        val2 = stage2_params[param]
        
        if val1 != val2:
            inconsistencies.append(f"{param}: Stage1={val1}, Stage2={val2}")
    
    return len(inconsistencies) == 0, inconsistencies


def print_parameter_summary(config: Dict, stage: str):
    """打印参数摘要"""
    params = extract_critical_parameters(config, stage)
    
    print(f"\n📋 {stage.upper()} 关键参数:")
    for param, value in params.items():
        print(f"  {param}: {value}")


def main():
    """主函数"""
    print("🔍 参数一致性检查工具")
    print("=" * 50)
    
    # 配置文件路径
    stage1_config_path = "anomaly_aware_kt/configs/assist17_baseline.yaml"
    stage2_config_path = "anomaly_aware_kt/configs/assist17_curriculum.yaml"
    
    # 加载配置
    stage1_config = load_config(stage1_config_path)
    stage2_config = load_config(stage2_config_path)
    
    if not stage1_config:
        print(f"❌ 无法加载第一阶段配置: {stage1_config_path}")
        return
    
    if not stage2_config:
        print(f"❌ 无法加载第二阶段配置: {stage2_config_path}")
        return
    
    print(f"✅ 已加载配置文件:")
    print(f"  Stage1: {stage1_config_path}")
    print(f"  Stage2: {stage2_config_path}")
    
    # 打印参数摘要
    print_parameter_summary(stage1_config, "stage1")
    print_parameter_summary(stage2_config, "stage2")
    
    # 检查一致性
    print(f"\n🔍 一致性检查:")
    is_consistent, inconsistencies = check_consistency(stage1_config, stage2_config)
    
    if is_consistent:
        print("✅ 参数一致性检查通过!")
        print("   所有关键参数在两个阶段间保持一致")
    else:
        print("❌ 发现参数不一致:")
        for inconsistency in inconsistencies:
            print(f"   • {inconsistency}")
        
        print(f"\n💡 建议:")
        print("1. 修复配置文件中的不一致参数")
        print("2. 确保第二阶段脚本正确传递第一阶段的模型参数")
        print("3. 重新运行此检查工具验证修复")
    
    # 额外检查：第二阶段特有参数
    print(f"\n📊 第二阶段特有参数:")
    stage2_specific = {
        'curriculum_strategy': stage2_config.get('curriculum_strategy'),
        'curriculum_epochs': stage2_config.get('curriculum_epochs'),
        'anomaly_ratio': stage2_config.get('anomaly_ratio'),
        'detector_hidden_dim': stage2_config.get('detector_hidden_dim'),
    }
    
    for param, value in stage2_specific.items():
        if value is not None:
            print(f"  {param}: {value}")
    
    # 重要提醒
    print(f"\n⚠️  重要提醒:")
    print("1. 第二阶段必须使用与第一阶段相同的模型架构参数")
    print("2. 学习率、批次大小等训练参数也应保持一致")
    print("3. 异常检测器的参数可以独立优化")
    print("4. 课程学习的参数不影响最终对比的公平性")
    
    # 使用建议
    print(f"\n🚀 使用建议:")
    print("运行第二阶段时，确保传递正确的第一阶段参数:")
    print("python anomaly_aware_kt/scripts/run_stage2_curriculum.py \\")
    print("  --dataset assist17 \\")
    print("  --baseline_model_path output/stage1_xxx/baseline/best_model.pt \\")
    print("  --auto_config \\")
    print(f"  --d_model {stage1_config.get('d_model', 128)} \\")
    print(f"  --n_heads {stage1_config.get('n_heads', 8)} \\")
    print(f"  --n_layers {stage1_config.get('n_layers', 3)} \\")
    print(f"  --dropout {stage1_config.get('dropout', 0.2)}")
    if stage1_config.get('with_pid', False):
        print("  --with_pid")


if __name__ == "__main__":
    main()
