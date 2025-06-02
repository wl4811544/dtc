#!/bin/bash

# 实验监控脚本
# 用于监控当前运行的训练实验

echo "🔍 实验监控面板"
echo "=================="

# 检查是否有Python训练进程在运行
PYTHON_PROCS=$(ps aux | grep "run_stage1_only.py" | grep -v grep)

if [ -z "$PYTHON_PROCS" ]; then
    echo "❌ 没有发现运行中的训练进程"
    echo ""
    echo "💡 启动实验2命令:"
    echo "   bash run_experiment2.sh"
    exit 1
else
    echo "✅ 发现运行中的训练进程:"
    echo "$PYTHON_PROCS"
    echo ""
fi

# 显示GPU使用情况
echo "🖥️  GPU状态:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
while IFS=, read -r index name util mem_used mem_total; do
    echo "  GPU$index: $util% 使用率, 内存: ${mem_used}MB/${mem_total}MB"
done
echo ""

# 显示最新的训练日志
if [ -f "nohup_exp2.out" ]; then
    echo "📊 实验2最新进度 (最后10行):"
    echo "----------------------------------------"
    tail -10 nohup_exp2.out
    echo "----------------------------------------"
    echo ""
    
    # 提取最新的性能指标
    LATEST_AUC=$(grep "Val - ACC:" nohup_exp2.out | tail -1 | grep -o "AUC: [0-9.]*" | cut -d' ' -f2)
    LATEST_EPOCH=$(grep "Epoch [0-9]*/100" nohup_exp2.out | tail -1 | grep -o "Epoch [0-9]*" | cut -d' ' -f2)
    
    if [ ! -z "$LATEST_AUC" ] && [ ! -z "$LATEST_EPOCH" ]; then
        echo "📈 当前性能:"
        echo "  轮次: $LATEST_EPOCH/100"
        echo "  最新AUC: $LATEST_AUC"
        echo "  基线对比: 0.7407 (实验1)"
        
        # 计算改善程度
        IMPROVEMENT=$(echo "$LATEST_AUC - 0.7407" | bc -l 2>/dev/null || echo "计算中...")
        if [ "$IMPROVEMENT" != "计算中..." ]; then
            echo "  改善程度: +$IMPROVEMENT"
        fi
    fi
elif [ -f "nohup.out" ]; then
    echo "📊 基线实验日志 (最后5行):"
    echo "----------------------------------------"
    tail -5 nohup.out
    echo "----------------------------------------"
else
    echo "❓ 未找到训练日志文件"
fi

echo ""
echo "🔄 实时监控命令:"
echo "  tail -f nohup_exp2.out    # 实时查看实验2日志"
echo "  watch nvidia-smi          # 实时查看GPU状态"
echo "  bash monitor_experiment.sh # 刷新此监控面板"
