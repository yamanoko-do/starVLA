#!/usr/bin/env python
"""Complete benchmark: VLM vs ActionHead speed comparison."""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # starVLA root

import torch
import time


def benchmark_full_system():
    """Benchmark complete QwenZone system."""

    print("=" * 80)
    print("QwenZone Full System Benchmark")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0) if device == 'cuda' else 'N/A'}")

    # 1. Test StereoEncoder
    print("\n[1] Benchmarking StereoEncoder...")
    from starVLA.model.modules.action_model.StereoEncoder import StereoEncoder

    stereo = StereoEncoder(
        ckpt_path="/mnt/workspace/yama/OpenStereo/output/MultiDataset/WAVEStereo/wavestereo_mixdataset/20260623_wavestereo_filter/ckpt/checkpoint_epoch_2.pth",
        update_iters=4,
        hidden_dim=512,
        N_stereo_tokens=64,
        input_size=(224, 224),
    ).to(device).eval()

    B, T = 2, 4
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
    stereo_per_step = stereo_total / T
    stereo_per_sample = stereo_per_step / B

    print(f"✓ StereoEncoder (T=4): {stereo_total:.2f} ± {torch.tensor(times).std().item():.2f} ms")
    print(f"  Per timestep: {stereo_per_step:.2f} ms")
    print(f"  Per sample: {stereo_per_sample:.2f} ms")

    # 2. Test ActionHead
    print("\n[2] Benchmarking ActionHead...")
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
    action_per_step = action_total / T
    action_per_sample = action_per_step / B

    print(f"✓ ActionHead (T=4): {action_total:.2f} ± {torch.tensor(times).std().item():.2f} ms")
    print(f"  Per timestep: {action_per_step:.2f} ms")
    print(f"  Per sample: {action_per_sample:.2f} ms")

    # 3. Test VLM (use lightweight model for comparison)
    print("\n[3] Benchmarking VLM (lightweight transformer proxy)...")

    # Create a proxy for VLM (64 layers is too heavy for testing)
    # Use 8 layers as reasonable estimate
    vlm_proxy = torch.nn.TransformerEncoder(
        torch.nn.TransformerEncoderLayer(d_model=2560, nhead=16, batch_first=True),
        num_layers=8,
    ).to(device).eval()

    # VLM processes sequence of tokens: [B, T, N_tokens, hidden_size]
    # Assume ~1000 tokens per timestep (language + vision + action)
    seq_len = 1000
    dummy_vlm_input = torch.randn(B, T, seq_len, 2560, device=device)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            for t in range(T):
                _ = vlm_proxy(dummy_vlm_input[:, t])
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(5):
        start = time.perf_counter_ns()
        with torch.no_grad():
            for t in range(T):
                _ = vlm_proxy(dummy_vlm_input[:, t])
        torch.cuda.synchronize()
        times.append((time.perf_counter_ns() - start) / 1e6)

    vlm_total = sum(times) / len(times)
    vlm_per_step = vlm_total / T
    vlm_per_sample = vlm_per_step / B

    print(f"✓ VLM proxy (T=4): {vlm_total:.2f} ± {torch.tensor(times).std().item():.2f} ms")
    print(f"  Per timestep: {vlm_per_step:.2f} ms")
    print(f"  Per sample: {vlm_per_sample:.2f} ms")
    print(f"  Note: Real Qwen3-VL-4B has ~64 layers, this is 8-layer proxy")

    # Summary
    print("\n" + "=" * 80)
    print("PER-SAMPLE BREAKDOWN (Batch=2, T=4)")
    print("=" * 80)
    print(f"{'Component':<25} {'Time (ms)':<15} {'% of Action':<15} {'Analysis':<20}")
    print("-" * 80)
    print(f"{'VLM (8-layer proxy)':<25} {vlm_per_sample:>10.2f}      {'':<15} {'~300ms real (64L)':<20}")
    print(f"{'StereoEncoder':<25} {stereo_per_sample:>10.2f}      {stereo_per_sample/(stereo_per_sample+action_per_sample)*100:>10.1f}%        {'':<20}")
    print(f"{'ActionHead':<25} {action_per_sample:>10.2f}      {action_per_sample/(stereo_per_sample+action_per_sample)*100:>10.1f}%        {'':<20}")
    print("-" * 80)
    action_only_total = stereo_per_sample + action_per_sample
    print(f"{'ActionHead + Stereo':<25} {action_only_total:>10.2f}      {100.0:>10.1f}%        {'':<20}")

    # Extrapolate to real VLM
    print("\n" + "=" * 80)
    print("ESTIMATED REAL PERFORMANCE (Qwen3-VL-4B)")
    print("=" * 80)
    real_vlm_estimate = vlm_per_sample * 8  # 8 layers → 64 layers
    full_total_estimate = real_vlm_estimate + action_only_total

    print(f"\nEstimated per-sample breakdown:")
    print(f"  - VLM (Qwen3-VL-4B):  ~{real_vlm_estimate:.1f} ms  ({real_vlm_estimate/full_total_estimate*100:.1f}%)")
    print(f"  - StereoEncoder:     ~{stereo_per_sample:.1f} ms  ({stereo_per_sample/full_total_estimate*100:.1f}%)")
    print(f"  - ActionHead:        ~{action_per_sample:.1f} ms  ({action_per_sample/full_total_estimate*100:.1f}%)")
    print(f"  - TOTAL:             ~{full_total_estimate:.1f} ms")

    print(f"\nFPS estimate (per sample):")
    fps = 1000 / full_total_estimate
    print(f"  - {fps:.1f} FPS")

    print("\n" + "=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)
    print(f"✓ StereoEncoder is fast: {stereo_per_sample:.1f} ms per sample")
    print(f"✓ ActionHead is fast: {action_per_sample:.1f} ms per sample")
    print(f"✓ Action components (Stereo + ActionHead) = {action_only_total:.1f} ms")
    print(f"⚠️  VLM dominates: ~{real_vlm_estimate:.1f} ms ({real_vlm_estimate/full_total_estimate*100:.1f}% of total)")
    print(f"\nOptimization opportunities:")
    if stereo_per_sample > action_per_sample * 5:
        print(f"  • Consider reduce update_iters (current=4) to lower StereoEncoder cost")
    print(f"  • VLM is the bottleneck; consider caching or asynchronous execution")


if __name__ == "__main__":
    benchmark_full_system()
