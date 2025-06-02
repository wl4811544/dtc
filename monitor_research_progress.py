#!/usr/bin/env python3
"""
研究进度监控面板

实时监控所有实验的进度和状态
"""

import os
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path


class ResearchMonitor:
    """研究进度监控器"""
    
    def __init__(self, output_dir="output"):
        self.output_dir = Path(output_dir)
        self.experiments = [
            'assist17_base', 'assist17_expanded', 'assist17_deep',
            'assist09_base', 'algebra05_base'
        ]
    
    def check_experiment_status(self, exp_name):
        """检查单个实验状态"""
        status = {
            'experiment': exp_name,
            'stage1': {'status': 'not_started', 'progress': 0},
            'stage2': {'status': 'not_started', 'progress': 0},
            'stage3': {'status': 'not_started', 'progress': 0}
        }
        
        # 检查第一阶段
        stage1_dir = self.output_dir / f"{exp_name}_stage1"
        if stage1_dir.exists():
            best_model = stage1_dir / 'baseline' / 'best_model.pt'
            if best_model.exists():
                status['stage1'] = {'status': 'completed', 'progress': 100}
            else:
                # 检查训练日志估算进度
                log_file = stage1_dir / 'training.log'
                if log_file.exists():
                    progress = self.estimate_progress_from_log(log_file)
                    status['stage1'] = {'status': 'running', 'progress': progress}
                else:
                    status['stage1'] = {'status': 'started', 'progress': 0}
        
        # 检查第二阶段
        stage2_dir = self.output_dir / f"{exp_name}_stage2"
        if stage2_dir.exists():
            best_detector = stage2_dir / 'curriculum_anomaly' / 'best_model.pt'
            if best_detector.exists():
                status['stage2'] = {'status': 'completed', 'progress': 100}
            else:
                log_file = stage2_dir / 'curriculum_training.log'
                if log_file.exists():
                    progress = self.estimate_progress_from_log(log_file)
                    status['stage2'] = {'status': 'running', 'progress': progress}
                else:
                    status['stage2'] = {'status': 'started', 'progress': 0}
        
        # 检查第三阶段（待实现）
        stage3_dir = self.output_dir / f"{exp_name}_stage3"
        if stage3_dir.exists():
            status['stage3'] = {'status': 'started', 'progress': 0}
        
        return status
    
    def estimate_progress_from_log(self, log_file):
        """从日志文件估算训练进度"""
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # 查找最新的epoch信息
            latest_epoch = 0
            total_epochs = 100  # 默认值
            
            for line in reversed(lines[-50:]):  # 检查最后50行
                if 'Epoch' in line and '/' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'Epoch' in part and i + 1 < len(parts):
                            epoch_info = parts[i + 1]
                            if '/' in epoch_info:
                                current, total = epoch_info.split('/')
                                latest_epoch = int(current)
                                total_epochs = int(total)
                                break
                    break
            
            progress = min(100, (latest_epoch / total_epochs) * 100)
            return int(progress)
            
        except Exception:
            return 0
    
    def get_gpu_status(self):
        """获取GPU状态"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total', 
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                gpu_info = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split(', ')
                        if len(parts) >= 5:
                            gpu_info.append({
                                'index': parts[0],
                                'name': parts[1],
                                'utilization': f"{parts[2]}%",
                                'memory': f"{parts[3]}MB/{parts[4]}MB"
                            })
                return gpu_info
            else:
                return []
        except Exception:
            return []
    
    def get_running_processes(self):
        """获取运行中的训练进程"""
        try:
            result = subprocess.run(
                ['ps', 'aux'], capture_output=True, text=True
            )
            
            processes = []
            for line in result.stdout.split('\n'):
                if 'run_stage' in line and 'python' in line:
                    parts = line.split()
                    if len(parts) >= 11:
                        processes.append({
                            'pid': parts[1],
                            'cpu': parts[2],
                            'memory': parts[3],
                            'command': ' '.join(parts[10:])
                        })
            
            return processes
        except Exception:
            return []
    
    def display_status(self):
        """显示状态面板"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("🎯 完美研究计划 - 实时监控面板")
        print("=" * 80)
        print(f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 实验状态
        print(f"\n📊 实验进度:")
        print(f"{'实验名称':<20} {'第一阶段':<15} {'第二阶段':<15} {'第三阶段':<15}")
        print("-" * 80)
        
        for exp_name in self.experiments:
            status = self.check_experiment_status(exp_name)
            
            stage1_display = self.format_stage_status(status['stage1'])
            stage2_display = self.format_stage_status(status['stage2'])
            stage3_display = self.format_stage_status(status['stage3'])
            
            print(f"{exp_name:<20} {stage1_display:<15} {stage2_display:<15} {stage3_display:<15}")
        
        # GPU状态
        gpu_info = self.get_gpu_status()
        if gpu_info:
            print(f"\n🖥️  GPU状态:")
            for gpu in gpu_info:
                print(f"  GPU{gpu['index']}: {gpu['utilization']} 使用率, 内存: {gpu['memory']}")
        
        # 运行中的进程
        processes = self.get_running_processes()
        if processes:
            print(f"\n🔄 运行中的训练进程:")
            for proc in processes:
                print(f"  PID {proc['pid']}: CPU {proc['cpu']}%, 内存 {proc['memory']}%")
                print(f"    {proc['command'][:60]}...")
        
        # 磁盘使用情况
        try:
            result = subprocess.run(['df', '-h', str(self.output_dir)], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    disk_info = lines[1].split()
                    print(f"\n💾 磁盘使用: {disk_info[2]} / {disk_info[1]} ({disk_info[4]})")
        except Exception:
            pass
        
        print(f"\n💡 提示:")
        print("  - 按 Ctrl+C 退出监控")
        print("  - 实验结果保存在 output/ 目录")
        print("  - 详细日志查看: tail -f output/实验名_stage阶段/training.log")
    
    def format_stage_status(self, stage_info):
        """格式化阶段状态显示"""
        status = stage_info['status']
        progress = stage_info['progress']
        
        if status == 'not_started':
            return "⚪ 未开始"
        elif status == 'started':
            return "🟡 已开始"
        elif status == 'running':
            return f"🔵 进行中 {progress}%"
        elif status == 'completed':
            return "🟢 已完成"
        else:
            return "❓ 未知"
    
    def run_monitor(self, refresh_interval=30):
        """运行监控循环"""
        try:
            while True:
                self.display_status()
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            print(f"\n\n👋 监控已停止")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='研究进度监控面板')
    parser.add_argument('--interval', type=int, default=30, 
                       help='刷新间隔（秒）')
    parser.add_argument('--output_dir', default='output',
                       help='输出目录')
    
    args = parser.parse_args()
    
    monitor = ResearchMonitor(args.output_dir)
    monitor.run_monitor(args.interval)


if __name__ == "__main__":
    main()
