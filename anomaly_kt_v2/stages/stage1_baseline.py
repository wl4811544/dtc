"""
Stage 1: 基线DTransformer模型训练

训练标准的DTransformer知识追踪模型作为后续异常感知训练的基线
"""

import os
import sys
import torch

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from DTransformer.model import DTransformer
from DTransformer.eval import Evaluator
from anomaly_kt_v2.core.common import print_stage_header, print_training_summary


def validate_stage1_parameters(args, dataset_config):
    """验证第一阶段所需的参数

    Args:
        args: 命令行参数对象
        dataset_config: 数据集配置字典

    Raises:
        ValueError: 如果缺少必需参数
    """
    # 检查基本参数
    required_basic_params = ['device', 'output_dir']
    for param in required_basic_params:
        if not hasattr(args, param) or getattr(args, param) is None:
            raise ValueError(f"Missing required parameter: {param}")

    # 检查模型参数
    required_model_params = [
        'd_model', 'n_heads', 'n_know', 'n_layers',
        'dropout', 'lambda_cl', 'window'
    ]
    for param in required_model_params:
        if not hasattr(args, param):
            raise ValueError(f"Missing required model parameter: {param}")

    # 检查训练参数
    required_training_params = ['kt_epochs', 'learning_rate', 'patience']
    for param in required_training_params:
        if not hasattr(args, param):
            raise ValueError(f"Missing required training parameter: {param}")

    # 检查数据集配置
    required_dataset_params = ['n_questions', 'n_pid']
    for param in required_dataset_params:
        if param not in dataset_config:
            raise ValueError(f"Missing required dataset parameter: {param}")

    # 检查可选参数，设置默认值
    if not hasattr(args, 'with_pid'):
        args.with_pid = False
    if not hasattr(args, 'proj'):
        args.proj = False
    if not hasattr(args, 'hard_neg'):
        args.hard_neg = False
    if not hasattr(args, 'use_cl'):
        args.use_cl = False

    print("✅ 第一阶段参数验证通过")


def print_stage1_parameters(args, dataset_config):
    """打印第一阶段参数信息"""
    print("\n📋 第一阶段参数配置:")
    print("  基本参数:")
    print(f"    设备: {args.device}")
    print(f"    输出目录: {args.output_dir}")
    print(f"    使用问题ID: {getattr(args, 'with_pid', False)}")

    print("  模型参数:")
    print(f"    模型维度: {args.d_model}")
    print(f"    注意力头数: {args.n_heads}")
    print(f"    知识概念数: {args.n_know}")
    print(f"    层数: {args.n_layers}")
    print(f"    Dropout: {args.dropout}")
    print(f"    对比学习权重: {args.lambda_cl}")
    print(f"    使用投影: {getattr(args, 'proj', False)}")
    print(f"    困难负样本: {getattr(args, 'hard_neg', False)}")
    print(f"    窗口大小: {args.window}")

    print("  训练参数:")
    print(f"    训练轮数: {args.kt_epochs}")
    print(f"    学习率: {args.learning_rate}")
    print(f"    早停耐心: {args.patience}")
    print(f"    使用对比学习: {getattr(args, 'use_cl', False)}")

    print("  数据集参数:")
    print(f"    问题总数: {dataset_config['n_questions']}")
    print(f"    问题ID总数: {dataset_config['n_pid']}")


def train_baseline_model(args, dataset_config, train_data, val_data):
    """训练基线DTransformer模型

    Args:
        args: 命令行参数，包含模型和训练配置
        dataset_config: 数据集配置，包含 n_questions, n_pid 等
        train_data: 训练数据加载器
        val_data: 验证数据加载器

    Returns:
        str: 训练好的模型文件路径
    """
    # 验证参数
    validate_stage1_parameters(args, dataset_config)

    # 打印阶段标题
    print_stage_header("基线DTransformer模型训练", 1)

    # 打印参数信息
    print_stage1_parameters(args, dataset_config)

    # 创建模型
    print("\n🔧 创建DTransformer模型...")
    model = DTransformer(
        n_questions=dataset_config['n_questions'],
        n_pid=dataset_config['n_pid'] if args.with_pid else 0,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_know=args.n_know,
        n_layers=args.n_layers,
        dropout=args.dropout,
        lambda_cl=args.lambda_cl,
        proj=args.proj,
        hard_neg=args.hard_neg,
        window=args.window
    )

    print(f"✅ 模型创建成功")
    print(f"  参数总数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 设置设备
    model.to(args.device)

    # 创建优化器
    print("\n🚀 创建优化器...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # 创建评估器
    evaluator = Evaluator()

    # 创建保存目录
    save_dir = os.path.join(args.output_dir, 'baseline')
    os.makedirs(save_dir, exist_ok=True)

    print(f"✅ 训练准备完成")
    print(f"  保存目录: {save_dir}")
    print(f"  优化器: Adam")
    print(f"  学习率: {args.learning_rate}")

    # 开始训练
    print("\n" + "="*60)
    print("开始基线模型训练...")
    print("="*60)

    # 训练循环
    best_auc = 0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.kt_epochs + 1):
        print(f"\nEpoch {epoch}/{args.kt_epochs}")

        # 训练阶段
        model.train()
        total_loss = 0.0
        total_pred_loss = 0.0
        total_cl_loss = 0.0
        total_cnt = 0

        from tqdm import tqdm
        for batch in tqdm(train_data, desc="Training"):
            # 获取批次数据
            if args.with_pid:
                q, s, pid = batch.get("q", "s", "pid")
            else:
                q, s = batch.get("q", "s")
                pid = None

            q = q.to(args.device)
            s = s.to(args.device)
            if pid is not None:
                pid = pid.to(args.device)

            # 前向传播
            if args.use_cl:
                loss, pred_loss, cl_loss = model.get_cl_loss(q, s, pid)
                total_pred_loss += pred_loss.item()
                total_cl_loss += cl_loss.item()
            else:
                loss = model.get_loss(q, s, pid)
                pred_loss = loss
                cl_loss = torch.tensor(0.0)
                total_pred_loss += pred_loss.item()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_cnt += 1

        # 计算平均损失
        avg_loss = total_loss / total_cnt
        avg_pred_loss = total_pred_loss / total_cnt
        avg_cl_loss = total_cl_loss / total_cnt if args.use_cl else 0.0

        # 验证阶段
        model.eval()
        with torch.no_grad():
            # 收集验证数据的预测结果
            for batch in val_data:
                if args.with_pid:
                    q, s, pid = batch.get("q", "s", "pid")
                else:
                    q, s = batch.get("q", "s")
                    pid = None

                q = q.to(args.device)
                s = s.to(args.device)
                if pid is not None:
                    pid = pid.to(args.device)

                # 获取预测结果 (只取logits，忽略其他返回值)
                pred, *_ = model.predict(q, s, pid)

                # 准备真实标签和预测值
                # 使用sigmoid将logits转换为概率
                pred_probs = torch.sigmoid(pred)

                # 只评估有效位置 (s >= 0)
                valid_mask = s >= 0
                target_flat = s[valid_mask].float()
                pred_flat = pred_probs[valid_mask]

                evaluator.evaluate(target_flat, pred_flat)

            # 获取评估结果
            val_metrics = evaluator.report()

            # 重置评估器为下一轮准备
            evaluator = Evaluator()

        # 打印训练信息
        print(f"  Train Loss: {avg_loss:.4f} (Pred: {avg_pred_loss:.4f}, CL: {avg_cl_loss:.4f})")
        print(f"  Val AUC: {val_metrics['auc']:.4f}, Val ACC: {val_metrics['acc']:.4f}")

        # 保存最佳模型
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_epoch = epoch
            patience_counter = 0

            # 保存模型
            model_path = os.path.join(save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'auc': val_metrics['auc'],
                'acc': val_metrics['acc'],
                'args': args
            }, model_path)

            print(f"  ✅ 新的最佳模型已保存 (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1
            print(f"  ⏳ 无改进 ({patience_counter}/{args.patience})")

        # 早停检查
        if patience_counter >= args.patience:
            print(f"\n🛑 早停触发！最佳AUC: {best_auc:.4f} (Epoch {best_epoch})")
            break

    # 训练完成
    final_metrics = {
        'auc': best_auc,
        'acc': val_metrics['acc'],
        'best_epoch': best_epoch,
        'total_epochs': epoch
    }

    # 打印训练总结
    print_training_summary("基线模型", final_metrics, save_dir)

    model_path = os.path.join(save_dir, 'best_model.pt')
    print(f"\n💾 模型已保存: {model_path}")

    return model_path
