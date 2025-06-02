#!/bin/bash

# ASSIST17 实验2: 模型容量扩展
# 目标: 验证更大模型 (d_model=256, n_heads=16) 是否能提升性能

echo "🚀 开始实验2: 模型容量扩展"
echo "配置: d_model=256, n_heads=16"
echo "基线对比: AUC=0.7407 (实验1)"
echo "预期目标: AUC>0.75"
echo "=================================="

# 记录开始时间
echo "开始时间: $(date)" > experiment2_log.txt

# 执行训练
echo "启动训练..."
nohup python anomaly_aware_kt/scripts/run_stage1_only.py \
    --dataset assist17 \
    --auto_config \
    --d_model 256 \
    --n_heads 16 \
    > nohup_exp2.out 2>&1 &

# 获取进程ID
PID=$!
echo "训练进程ID: $PID"
echo "进程ID: $PID" >> experiment2_log.txt

echo ""
echo "✅ 实验2已启动!"
echo "📊 监控命令:"
echo "  查看进度: tail -f nohup_exp2.out"
echo "  查看GPU:  nvidia-smi"
echo "  查看进程: ps aux | grep $PID"
echo ""
echo "📁 输出文件:"
echo "  训练日志: nohup_exp2.out"
echo "  实验记录: experiment2_log.txt"
echo ""
echo "⏱️  预计训练时间: 90-120分钟"
echo "🎯 目标: 验证AUC是否能超过0.75"
