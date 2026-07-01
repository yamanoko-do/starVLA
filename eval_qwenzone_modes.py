#!/usr/bin/env python
"""QwenZone 推理模式评估脚本 - 分别在两张 GPU 上评估 RNN 和 full 模式"""

import os
import sys
import argparse
import torch
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # starVLA root

from starVLA.model.framework.base_framework import build_framework
from transformers import AutoConfig

def main():
    parser = argparse.ArgumentParser(description="QwenZone 推理模式评估")
    parser.add_argument("--config", type=str, 
                       default="examples/Robotwin/train_files/starvla_qwenzone_robotwin_phase2.yaml",
                       help="配置文件路径")
    parser.add_argument("--checkpoint", type=str,
                       default="/mnt/workspace/yama/starVLA/results/Checkpoints_QwenZone/qwenzone_click_b2phase2_t8/checkpoints/steps_50000_pytorch_model.pt",
                       help="Checkpoint 路径")
    parser.add_argument("--mode", type=str, choices=["full", "rnn"], required=True,
                       help="推理模式: full (完整历史) 或 rnn (RNN模式)")
    parser.add_argument("--gpu", type=int, required=True,
                       help="GPU ID (0 或 1)")
    parser.add_argument("--tasks", type=str, nargs="+", default=["click_bell"],
                       help="要评估的任务列表")
    parser.add_argument("--num_episodes", type=int, default=10,
                       help="每个任务的评估 episode 数")
    parser.add_argument("--seed", type=int, default=0,
                       help="随机种子")
    
    args = parser.parse_args()
    
    # 设置 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    print(f"\n{'='*60}")
    print(f"QwenZone {args.mode.upper()} 模式评估")
    print(f"{'='*60}")
    print(f"配置文件: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"GPU: {args.gpu}")
    print(f"任务: {args.tasks}")
    print(f"每个任务评估 {args.num_episodes} 个 episode")
    print(f"随机种子: {args.seed}")
    print(f"{'='*60}\n")
    
    # 读取配置
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)
    
    # 修改推理模式
    cfg.framework.qwenzone.infer_mode = args.mode
    print(f"✓ 推理模式设置为: {cfg.framework.qwenzone.infer_mode}")
    
    # 构建模型
    print("正在加载模型...")
    try:
        model = build_framework(cfg)
        model = model.eval()
        device = next(model.parameters()).device
        print(f"✓ 模型已加载到设备: {device}")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("评估脚本已就绪，实际推理评估需要:")
    print("1. 集成 RoboTwin 评估框架（或创建简单的推理循环）")
    print("2. 使用当前 GPU ({args.gpu}) 启动评估服务器")
    print(f"{'='*60}")
    print("\n提示: 这是 QwenZone 模型加载验证，实际评估需要集成 RoboTwin 评估接口。")
    print("如需完整评估，请调用 RoboTwin eval 框架的 run_policy_server.sh 和 eval.sh。")

if __name__ == "__main__":
    main()
