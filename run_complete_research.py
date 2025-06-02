#!/usr/bin/env python3
"""
完美研究计划执行脚本

自动化执行所有实验配置的完整流程
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path


class ResearchPlanExecutor:
    """研究计划执行器"""
    
    def __init__(self, base_dir="output"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # 实验配置
        self.experiments = {
            'assist17_base': {
                'dataset': 'assist17',
                'config': 'assist17_base.yaml',
                'curriculum_config': 'assist17_base_curriculum.yaml',
                'model_params': '--d_model 128 --n_heads 8 --n_layers 3'
            },
            'assist17_expanded': {
                'dataset': 'assist17',
                'config': 'assist17_expanded.yaml', 
                'curriculum_config': 'assist17_expanded_curriculum.yaml',
                'model_params': '--d_model 256 --n_heads 16 --n_layers 3'
            },
            'assist17_deep': {
                'dataset': 'assist17',
                'config': 'assist17_deep.yaml',
                'curriculum_config': 'assist17_deep_curriculum.yaml', 
                'model_params': '--d_model 128 --n_heads 8 --n_layers 4'
            },
            'assist09_base': {
                'dataset': 'assist09',
                'config': 'assist09_base.yaml',
                'curriculum_config': 'assist09_base_curriculum.yaml',
                'model_params': '--d_model 128 --n_heads 8 --n_layers 3'
            },
            'algebra05_base': {
                'dataset': 'algebra05', 
                'config': 'algebra05_base.yaml',
                'curriculum_config': 'algebra05_base_curriculum.yaml',
                'model_params': '--d_model 128 --n_heads 8 --n_layers 3'
            }
        }
        
        self.results = {}
        
    def run_stage1(self, exp_name, exp_config):
        """运行第一阶段"""
        print(f"\n🚀 运行第一阶段: {exp_name}")
        
        output_dir = self.base_dir / f"{exp_name}_stage1"
        
        cmd = [
            'python', 'anomaly_aware_kt/scripts/run_stage1_only.py',
            '--dataset', exp_config['dataset'],
            '--config', f"anomaly_aware_kt/configs/{exp_config['config']}",
            '--output_dir', str(output_dir)
        ]
        
        # 添加模型参数
        cmd.extend(exp_config['model_params'].split())
        
        print(f"执行命令: {' '.join(cmd)}")
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()
        
        success = result.returncode == 0
        
        stage_result = {
            'success': success,
            'duration': end_time - start_time,
            'output_dir': str(output_dir),
            'command': ' '.join(cmd)
        }
        
        if success:
            print(f"✅ 第一阶段完成: {exp_name} ({stage_result['duration']:.1f}s)")
            # 查找最佳模型路径
            best_model_path = output_dir / 'baseline' / 'best_model.pt'
            stage_result['best_model_path'] = str(best_model_path)
        else:
            print(f"❌ 第一阶段失败: {exp_name}")
            print(f"错误: {result.stderr}")
            
        return stage_result
    
    def run_stage2(self, exp_name, exp_config, stage1_result):
        """运行第二阶段"""
        if not stage1_result['success']:
            print(f"⏭️  跳过第二阶段: {exp_name} (第一阶段失败)")
            return {'success': False, 'reason': 'stage1_failed'}
        
        print(f"\n🎓 运行第二阶段: {exp_name}")
        
        output_dir = self.base_dir / f"{exp_name}_stage2"
        
        cmd = [
            'python', 'anomaly_aware_kt/scripts/run_stage2_curriculum.py',
            '--dataset', exp_config['dataset'],
            '--baseline_model_path', stage1_result['best_model_path'],
            '--config', f"anomaly_aware_kt/configs/{exp_config['curriculum_config']}",
            '--output_dir', str(output_dir)
        ]
        
        # 添加模型参数（确保一致性）
        cmd.extend(exp_config['model_params'].split())
        
        print(f"执行命令: {' '.join(cmd)}")
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()
        
        success = result.returncode == 0
        
        stage_result = {
            'success': success,
            'duration': end_time - start_time,
            'output_dir': str(output_dir),
            'command': ' '.join(cmd)
        }
        
        if success:
            print(f"✅ 第二阶段完成: {exp_name} ({stage_result['duration']:.1f}s)")
            # 查找异常检测器路径
            detector_path = output_dir / 'curriculum_anomaly' / 'best_model.pt'
            stage_result['detector_path'] = str(detector_path)
        else:
            print(f"❌ 第二阶段失败: {exp_name}")
            print(f"错误: {result.stderr}")
            
        return stage_result
    
    def run_experiment(self, exp_name):
        """运行单个实验的完整流程"""
        print(f"\n{'='*60}")
        print(f"开始实验: {exp_name}")
        print(f"{'='*60}")
        
        exp_config = self.experiments[exp_name]
        exp_result = {
            'experiment': exp_name,
            'config': exp_config,
            'start_time': datetime.now().isoformat(),
            'stages': {}
        }
        
        # 第一阶段
        stage1_result = self.run_stage1(exp_name, exp_config)
        exp_result['stages']['stage1'] = stage1_result
        
        # 第二阶段
        stage2_result = self.run_stage2(exp_name, exp_config, stage1_result)
        exp_result['stages']['stage2'] = stage2_result
        
        # TODO: 第三阶段（待实现）
        # stage3_result = self.run_stage3(exp_name, exp_config, stage1_result, stage2_result)
        # exp_result['stages']['stage3'] = stage3_result
        
        exp_result['end_time'] = datetime.now().isoformat()
        exp_result['total_duration'] = sum(
            stage.get('duration', 0) for stage in exp_result['stages'].values()
        )
        
        self.results[exp_name] = exp_result
        
        # 保存中间结果
        self.save_results()
        
        return exp_result
    
    def run_all_experiments(self):
        """运行所有实验"""
        print("🎯 开始完美研究计划执行")
        print(f"总实验数量: {len(self.experiments)}")
        
        for exp_name in self.experiments:
            try:
                self.run_experiment(exp_name)
            except KeyboardInterrupt:
                print(f"\n⚠️  用户中断，保存当前结果...")
                self.save_results()
                break
            except Exception as e:
                print(f"❌ 实验 {exp_name} 发生错误: {e}")
                continue
        
        print(f"\n🎉 研究计划执行完成!")
        self.print_summary()
    
    def save_results(self):
        """保存结果"""
        results_file = self.base_dir / 'research_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"📊 结果已保存: {results_file}")
    
    def print_summary(self):
        """打印总结"""
        print(f"\n📊 实验总结:")
        print(f"{'实验名称':<20} {'第一阶段':<10} {'第二阶段':<10} {'总时长':<10}")
        print("-" * 60)
        
        for exp_name, result in self.results.items():
            stage1_status = "✅" if result['stages']['stage1']['success'] else "❌"
            stage2_status = "✅" if result['stages']['stage2']['success'] else "❌"
            duration = f"{result.get('total_duration', 0):.1f}s"
            
            print(f"{exp_name:<20} {stage1_status:<10} {stage2_status:<10} {duration:<10}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='完美研究计划执行器')
    parser.add_argument('--experiment', type=str, help='运行特定实验')
    parser.add_argument('--stage', type=str, choices=['1', '2', '3'], help='运行特定阶段')
    parser.add_argument('--dry_run', action='store_true', help='仅显示将要执行的命令')
    
    args = parser.parse_args()
    
    executor = ResearchPlanExecutor()
    
    if args.experiment:
        if args.experiment in executor.experiments:
            executor.run_experiment(args.experiment)
        else:
            print(f"❌ 未知实验: {args.experiment}")
            print(f"可用实验: {list(executor.experiments.keys())}")
    else:
        executor.run_all_experiments()


if __name__ == "__main__":
    main()
