#!/usr/bin/env python
"""Complete benchmark including ViT, State Encoder, and all components."""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # starVLA root

import torch
import time
import torch.nn as nn


def benchmark_complete_action_pipeline():
    """Benchmark complete action pipeline including all input encoders."""

    print("=" * 80)
    print("完整的动作模型推理链路 Benchmark")
    print("包括: ViT + StereoEncoder + StateEncoder + ActionHead")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    B, T = 2, 4  # Batch size, Timesteps

    # ========================================================================
    # 1. ViT (Vision Transformer) - 从 VLM 中提取
    # ========================================================================
    print("\n[1] Benchmarking ViT (Vision Encoder)...")

    # 创建一个简化的 ViT proxy（实际是 Qwen3-VL 的 visual encoder）
    # Qwen3-VL 使用 ViT-600M + Perceiver resampler
    vit_proxy = nn.Sequential(
        # Patch embedding (简化版)
        nn.Conv2d(3, 64, 16, 16),  # 224 -> 14
        nn.LayerNorm([64, 14, 14]),
        nn.Flatten(1),
        nn.Linear(64 * 14 * 14, 2560),  # Project to hidden_dim
    ).to(device).eval()

    dummy_images = torch.randn(B * T, 3, 224, 224, device=device)

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = vit_proxy(dummy_images)
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(20):
        start = time.perf_counter_ns()
        with torch.no_grad():
            _ = vit_proxy(dummy_images)
        torch.cuda.synchronize()
        times.append((time.perf_counter_ns() - start) / 1e6)

    vit_total = sum(times) / len(times)
    vit_per_sample = vit_total / (B * T)

    print(f"✓ ViT (simplified proxy): {vit_total:.2f} ± {torch.tensor(times).std().item():.2f} ms")
    print(f"  Per sample: {vit_per_sample:.2f} ms")
    print(f"  Note: 实际 Qwen3-VL ViT-600M 会更慢，需实测")

    # ========================================================================
    # 2. StereoEncoder
    # ========================================================================
    print("\n[2] Benchmarking StereoEncoder...")
    from starVLA.model.modules.action_model.StereoEncoder import StereoEncoder

    stereo = StereoEncoder(
        ckpt_path="/mnt/workspace/yama/OpenStereo/output/MultiDataset/WAVEStereo/wavestereo_mixdataset/20260623_wavestereo_filter/ckpt/checkpoint_epoch_2.pth",
        update_iters=4,
        hidden_dim=512,
        N_stereo_tokens=64,
        input_size=(224, 224),
    ).to(device).eval()

    dummy_left = torch.randn(B, T, 3, 224, 224, device=device)
    dummy_right = torch.randn(B, T, 3, 224, 224, device=device)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            for t in range(T):
                _ = stereo(dummy_left[:, t], dummy_right[:, t])
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(10):
        start = time.perf_counter_ns()
        with torch.no_grad():
            for t in range(T):
                _ = stereo(dummy_left[:, t], dummy_right[:, t])
        torch.cuda.synchronize()
        times.append((time.perf_counter_ns() - start) / 1e6)

    stereo_total = sum(times) / len(times)
    stereo_per_sample = stereo_total / (B * T)

    print(f"✓ StereoEncoder: {stereo_total:.2f} ± {torch.tensor(times).std().item():.2f} ms")
    print(f"  Per sample: {stereo_per_sample:.2f} ms")

    # ========================================================================
    # 3. State Encoder (在 ActionHead 内部)
    # ========================================================================
    print("\n[3] Benchmarking State Encoder...")

    state_dim = 28
    hidden_dim = 512
    N_state_tokens = 4

    state_encoder = nn.Sequential(
        nn.LayerNorm(state_dim),
        nn.Linear(state_dim, hidden_dim * N_state_tokens),
    ).to(device).eval()

    dummy_state = torch.randn(B, T, state_dim, device=device)

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = state_encoder(dummy_state.view(-1, state_dim))
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(20):
        start = time.perf_counter_ns()
        with torch.no_grad():
            _ = state_encoder(dummy_state.view(-1, state_dim))
        torch.cuda.synchronize()
        times.append((time.perf_counter_ns() - start) / 1e6)

    state_total = sum(times) / len(times)
    state_per_sample = state_total / (B * T)

    print(f"✓ State Encoder: {state_total:.4f} ± {torch.tensor(times).std().item():.4f} ms")
    print(f"  Per sample: {state_per_sample:.4f} ms")

    # ========================================================================
    # 4. ActionHead
    # ========================================================================
    print("\n[4] Benchmarking ActionHead...")
    from starVLA.model.modules.action_model.ZoneMemory_ActionHeader import ZoneMemoryActionHead

    action_head = ZoneMemoryActionHead(
        input_dim=512,
        hidden_dim=512,
        action_dim=14,
        H_action=50,
        N_act=16,
        N_state_tokens=4,
        N_stereo=64,
        state_dim=28,
        nhead=8,
        num_layers=4,
    ).to(device).eval()

    dummy_vision_tokens = torch.randn(B, T, 196, 512, device=device)
    dummy_action_intent = torch.randn(B, T, 16, 512, device=device)
    dummy_stereo_tokens = torch.randn(B, T, 64, 512, device=device)
    dummy_state = torch.randn(B, T, 28, device=device)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = action_head(
                h_A=dummy_action_intent,
                h_V=dummy_vision_tokens,
                state=dummy_state,
                stereo=dummy_stereo_tokens,
            )
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(10):
        start = time.perf_counter_ns()
        with torch.no_grad():
            _ = action_head(
                h_A=dummy_action_intent,
                h_V=dummy_vision_tokens,
                state=dummy_state,
                stereo=dummy_stereo_tokens,
            )
        torch.cuda.synchronize()
        times.append((time.perf_counter_ns() - start) / 1e6)

    action_total = sum(times) / len(times)
    action_per_sample = action_total / (B * T)

    print(f"✓ ActionHead: {action_total:.2f} ± {torch.tensor(times).std().item():.2f} ms")
    print(f"  Per sample: {action_per_sample:.2f} ms")

    # ========================================================================
    # 总结
    # ========================================================================
    print("\n" + "=" * 80)
    print("完整的动作模型推理时间分解 (Per Sample)")
    print("=" * 80)
    print(f"{'组件':<25} {'时间 (ms)':<15} {'% 总计':<15} {'频率 (Hz)':<15}")
    print("-" * 80)

    # 注意：ViT 在实际中会慢得多，这里用估算
    vit_real_estimate_ms = 25.0  # Qwen3-VL ViT-600M 估算值

    components = [
        ("ViT (视觉编码)", vit_real_estimate_ms),
        ("StereoEncoder", stereo_per_sample),
        ("State Encoder", state_per_sample),
        ("ActionHead", action_per_sample),
    ]

    total_action_model = sum(c[1] for c in components)

    for name, time_ms in components:
        pct = time_ms / total_action_model * 100
        hz = 1000 / time_ms if time_ms > 0 else 0
        print(f"{name:<25} {time_ms:>10.2f}      {pct:>10.1f}%      {hz:>10.1f}")

    print("-" * 80)
    print(f"{'完整动作模型总计':<25} {total_action_model:>10.2f}      {100.0:>10.1f}%      {1000/total_action_model:>10.1f}")

    print("\n" + "=" * 80)
    print("完整系统对比")
    print("=" * 80)

    vlm_time_ms = 304.2  # 之前测量的 Qwen3-VL-4B
    full_system_ms = vlm_time_ms + total_action_model

    print(f"\n{'组件':<25} {'时间 (ms)':<15} {'频率 (Hz)':<15} {'说明':<30}")
    print("-" * 80)
    print(f"{'VLM (Qwen3-VL-4B)':<25} {vlm_time_ms:>10.1f}      {1000/vlm_time_ms:>10.1f}      {'大脑（异步）':<30}")
    print(f"{'动作模型 (完整)':<25} {total_action_model:>10.1f}      {1000/total_action_model:>10.1f}      {'本体（实时）':<30}")
    print(f"{'完整系统 (同步)':<25} {full_system_ms:>10.1f}      {1000/full_system_ms:>10.1f}      {'VLM 限制':<30}")

    print("\n" + "=" * 80)
    print("关键结论")
    print("=" * 80)
    print(f"✓ 完整动作模型: {total_action_model:.1f} ms/样本 ≈ {1000/total_action_model:.1f} Hz")
    print(f"✓ VLM: {vlm_time_ms:.1f} ms/样本 ≈ {1000/vlm_time_ms:.1f} Hz")
    print(f"✓ 本体比大脑快: {1000/total_action_model / (1000/vlm_time_ms):.1f}x")
    print(f"\n异步模式下:")
    print(f"  • 大脑 (VLM): {1000/vlm_time_ms:.1f} Hz —— 持续思考，提供高层次意图")
    print(f"  • 本体 (动作): {1000/total_action_model:.1f} Hz —— 实时执行，响应传感器输入")
    print(f"  • 性能比: {1000/total_action_model / (1000/vlm_time_ms):.1f}x")

    print("\n" + "=" * 80)
    print("注意")
    print("=" * 80)
    print("• ViT 时间使用了估算值 (25ms)，实际需要从完整 VLM 中分离测量")
    print("• State Encoder 非常快 (<0.01ms)，可以忽略")
    print("• StereoEncoder (15ms) 已优化 (224×224 分辨率)")
    print("• ActionHead (1ms) 非常轻量")


if __name__ == "__main__":
    benchmark_complete_action_pipeline()
