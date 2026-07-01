#!/usr/bin/env python
"""Simple benchmark for QwenZone components using loaded checkpoint."""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # starVLA root

import torch
import time
from torch import nn


class DummyVLM(nn.Module):
    """Dummy VLM for testing."""
    def __init__(self, hidden_size=2560):
        super().__init__()
        self.hidden_size = hidden_size
        self.vit = nn.Sequential(
            nn.Conv2d(3, 32, 16, 16),  # 224 -> 14
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((14, 14)),
        )
        self.llm = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=16, batch_first=True),
            num_layers=32,
        )
        self.processor = None

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw=None, output_hidden_states=True):
        # Simple forward: extract vision features and run through LLM
        B, T, C, H, W = pixel_values.shape
        # Flatten sequence
        pixel_values = pixel_values.view(B * T, C, H, W)

        # Vision features
        vision_feat = self.vit(pixel_values)  # [B*T, 32, 14, 14]
        vision_feat = vision_feat.flatten(1).unsqueeze(1)  # [B*T, 1, 32*14*14]

        # Project to hidden size
        vision_feat = nn.Linear(32*14*14, self.hidden_size).to(vision_feat.device)(vision_feat)

        # LLM processing
        seq_len = input_ids.shape[1] + 1  # +1 for vision token
        dummy_input = torch.randn(B, seq_len, self.hidden_size, device=input_ids.device)

        output = self.llm(dummy_input)

        if output_hidden_states:
            return type('obj', (object,), {'hidden_states': [output]})()
        return output


def benchmark_components():
    """Benchmark individual components."""

    print("=" * 80)
    print("QwenZone Component Speed Benchmark")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # 1. Test ViT (Vision Transformer)
    print("\n[1] Benchmarking ViT (vision encoder)...")
    vit = nn.Sequential(
        nn.Conv2d(3, 32, 16, 16),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((14, 14)),
    ).to(device).eval()

    dummy_img = torch.randn(2, 4, 3, 224, 224, device=device)  # B=2, T=4

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = vit(dummy_img.view(2*4, 3, 224, 224))
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(20):
        start = time.perf_counter_ns()
        with torch.no_grad():
            _ = vit(dummy_img.view(2*4, 3, 224, 224))
        torch.cuda.synchronize()
        times.append((time.perf_counter_ns() - start) / 1e6)

    vit_time = sum(times) / len(times)
    print(f"✓ ViT: {vit_time:.2f} ± {torch.tensor(times).std().item():.2f} ms")

    # 2. Test StereoEncoder
    print("\n[2] Benchmarking StereoEncoder...")
    from starVLA.model.modules.action_model.StereoEncoder import StereoEncoder

    stereo = StereoEncoder(
        ckpt_path="/mnt/workspace/yama/OpenStereo/output/MultiDataset/WAVEStereo/wavestereo_mixdataset/20260623_wavestereo_filter/ckpt/checkpoint_epoch_2.pth",
        update_iters=4,
        hidden_dim=512,
        N_stereo_tokens=64,
        input_size=(224, 224),
    ).to(device).eval()

    dummy_left = torch.randn(2, 4, 3, 224, 224, device=device)
    dummy_right = torch.randn(2, 4, 3, 224, 224, device=device)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            for t in range(4):
                _ = stereo(dummy_left[:, t], dummy_right[:, t])
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(10):
        start = time.perf_counter_ns()
        with torch.no_grad():
            for t in range(4):
                _ = stereo(dummy_left[:, t], dummy_right[:, t])
        torch.cuda.synchronize()
        times.append((time.perf_counter_ns() - start) / 1e6)

    stereo_time_total = sum(times) / len(times)
    stereo_time_per_step = stereo_time_total / 4
    print(f"✓ StereoEncoder (T=4): {stereo_time_total:.2f} ± {torch.tensor(times).std().item():.2f} ms")
    print(f"  Per timestep: {stereo_time_per_step:.2f} ms")

    # 3. Test ActionHead
    print("\n[3] Benchmarking ActionHead...")
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

    dummy_vision_tokens = torch.randn(2, 4, 196, 512, device=device)  # VLM vision tokens
    dummy_action_intent = torch.randn(2, 4, 16, 512, device=device)  # Action intent tokens
    dummy_stereo_tokens = torch.randn(2, 4, 64, 512, device=device)  # Stereo tokens
    dummy_state = torch.randn(2, 4, 28, device=device)

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

    action_time = sum(times) / len(times)
    print(f"✓ ActionHead (T=4): {action_time:.2f} ± {torch.tensor(times).std().item():.2f} ms")
    print(f"  Per timestep: {action_time/4:.2f} ms")

    # Summary
    print("\n" + "=" * 80)
    print("COMPONENT BREAKDOWN (per sample, per timestep)")
    print("=" * 80)
    print(f"{'Component':<25} {'Total (ms)':<15} {'Per step (ms)':<15} {'% of Total':<15}")
    print("-" * 80)

    vit_per_sample = vit_time / 8  # B=2, T=4 => 8 samples
    stereo_per_sample = stereo_time_per_step / 2  # B=2
    action_per_sample = (action_time / 4) / 2  # T=4, B=2

    total = vit_per_sample + stereo_per_sample + action_per_sample

    print(f"{'ViT (vision encoder)':<25} {vit_per_sample:>10.2f}       {vit_per_sample:>10.2f}        {vit_per_sample/total*100:>10.1f}%")
    print(f"{'StereoEncoder':<25} {stereo_per_sample:>10.2f}       {stereo_per_sample:>10.2f}        {stereo_per_sample/total*100:>10.1f}%")
    print(f"{'ActionHead':<25} {action_per_sample:>10.2f}       {action_per_sample:>10.2f}        {action_per_sample/total*100:>10.1f}%")
    print("-" * 80)
    print(f"{'TOTAL (ViT + Stereo + Action)':<25} {total:>10.2f}       {total:>10.2f}        {100.0:>10.1f}%")
    print("=" * 80)

    # Add LLM estimate
    print("\nNOTE: This does NOT include LLM (the slow part!)")
    print("Full QwenZone = ViT + LLM + StereoEncoder + ActionHead")
    print(f"If LLM ≈ 80ms, full forward ≈ {vit_per_sample + 80 + stereo_per_sample + action_per_sample:.1f} ms per sample")


if __name__ == "__main__":
    benchmark_components()
