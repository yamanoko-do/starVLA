#!/usr/bin/env python
"""Benchmark QwenZone inference speed: VLM vs ActionHead."""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # starVLA root

import torch
import time
import yaml
from typing import Dict

from starVLA.model.framework.VLM4A.QwenZone import Qwenvl_Zone
from starVLA.dataloader.gr00t_lerobot.hdf5_robotwin_dataset import HDF5RobotwinDataset


def benchmark_inference(
    num_warmup: int = 3,
    num_runs: int = 10,
    task_name: str = "click_bell",
    config_path: str = None,
):
    """Benchmark VLM and ActionHead inference speed.

    Measures:
        1. VLM forward time (ViT + LLM)
        2. Full forward time (ViT + StereoEncoder + ActionHead)
        3. ActionHead time (Full - VLM)
    """
    print("=" * 80)
    print("QwenZone Inference Speed Benchmark")
    print("=" * 80)

    # Load dataset
    print("\n[1] Loading dataset...")
    dataset = HDF5RobotwinDataset(
        dataset_path=f"/mnt/workspace/yama/RoboTwin/data/{task_name}",
        T=4,  # Use T=4 for realistic sequence
        H=50,
        img_size=224,
        stereo_size=(224, 224),
    )
    sample = dataset[0]
    print(f"✓ Dataset loaded, sample keys: {list(sample.keys())}")

    # Load QwenZone model
    print("\n[2] Loading QwenZone model...")

    # Load config from file if provided
    if config_path is None:
        config_path = "/mnt/workspace/yama/starVLA/examples/Robotwin/train_files/starvla_qwenzone_robotwin.yaml"

    print(f"  Loading config from: {config_path}")
    with open(config_path) as f:
        model_config = SimpleConfig(yaml.safe_load(f))

    model = Qwenvl_Zone(model_config).cuda().eval()
    print(f"✓ QwenZone loaded")

    # Get model stats
    vlm_params = sum(p.numel() for p in model.vlm.parameters()) / 1e6
    action_params = sum(p.numel() for p in model.action_head.parameters()) / 1e6
    stereo_params = sum(p.numel() for p in model.stereo_encoder.parameters()) / 1e6 if hasattr(model, 'stereo_encoder') else 0
    print(f"  - VLM params: {vlm_params:.1f}M")
    print(f"  - StereoEncoder params: {stereo_params:.1f}M")
    print(f"  - ActionHead params: {action_params:.1f}M")

    # Prepare input
    print("\n[3] Preparing input...")
    batch = {
        k: [sample[k]] * 2 for k in sample.keys()  # Batch size 2
    }

    # Move to CUDA and format
    from starVLA.datautils.process_starvla import process_batch
    formatted_batch = process_batch(batch, model.qwenvl.processor, model.device)

    print(f"✓ Input prepared")
    print(f"  - Batch size: 2")
    print(f"  - train_seq_len: {model.qwenzone['train_seq_len']}")

    # Warmup
    print("\n[4] Warmup runs...")
    with torch.no_grad():
        for i in range(num_warmup):
            _ = model(formatted_batch)
            torch.cuda.synchronize()
    print(f"✓ {num_warmup} warmup runs completed")

    # Benchmark VLM only
    print("\n[5] Benchmarking VLM (ViT + LLM)...")
    vlm_times = []
    with torch.no_grad():
        for i in range(num_runs):
            # Time VLM forward
            start = time.perf_counter_ns()

            # Extract inputs for VLM
            pixel_values = formatted_batch["pixel_values"].to(model.device)
            input_ids = formatted_batch["input_ids"].to(model.device)
            attention_mask = formatted_batch["attention_mask"].to(model.device)
            image_grid_thw = formatted_batch.get("image_grid_thw")

            # VLM forward
            _ = model.vlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
            )

            torch.cuda.synchronize()
            end = time.perf_counter_ns()
            vlm_times.append((end - start) / 1e6)  # Convert to ms

    vlm_time_ms = sum(vlm_times) / len(vlm_times)
    vlm_time_std = torch.tensor(vlm_times).std().item()

    print(f"✓ VLM time: {vlm_time_ms:.2f} ± {vlm_time_std:.2f} ms")

    # Benchmark full forward (VLM + ActionHead)
    print("\n[6] Benchmarking Full Forward (VLM + StereoEncoder + ActionHead)...")
    full_times = []
    with torch.no_grad():
        for i in range(num_runs):
            start = time.perf_counter_ns()

            # Full forward
            _ = model(formatted_batch)

            torch.cuda.synchronize()
            end = time.perf_counter_ns()
            full_times.append((end - start) / 1e6)  # Convert to ms

    full_time_ms = sum(full_times) / len(full_times)
    full_time_std = torch.tensor(full_times).std().item()

    print(f"✓ Full time: {full_time_ms:.2f} ± {full_time_std:.2f} ms")

    # Calculate ActionHead time
    action_time_ms = full_time_ms - vlm_time_ms
    action_time_std = (full_time_std**2 + vlm_time_std**2)**0.5  # Error propagation

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS (Batch Size = 2, train_seq_len = 4)")
    print("=" * 80)
    print(f"{'Component':<30} {'Time (ms)':<15} {'Std (ms)':<15} {'Percentage':<15}")
    print("-" * 80)
    print(f"{'VLM (ViT + LLM)':<30} {vlm_time_ms:>10.2f} ± {vlm_time_std:>6.2f}    {100.0:>10.1f}%")
    print(f"{'ActionHead + Stereo':<30} {action_time_ms:>10.2f} ± {action_time_std:>6.2f}    {action_time_ms/full_time_ms*100:>10.1f}%")
    print(f"{'--- StereoEncoder':<30} {'~?':<15} {'':<15} {'':<15}")
    print(f"{'--- ActionHead (transformer)':<30} {'~?':<15} {'':<15} {'':<15}")
    print("-" * 80)
    print(f"{'TOTAL (VLM + Action)':<30} {full_time_ms:>10.2f} ± {full_time_std:>6.2f}    {100.0:>10.1f}%")
    print("=" * 80)

    # Detailed breakdown estimate
    print("\nEstimated Breakdown (assuming StereoEncoder ≈ 40ms from previous profiling):")
    stereo_est_ms = 40.0
    action_head_est_ms = action_time_ms - stereo_est_ms
    print(f"  - StereoEncoder:  ~{stereo_est_ms:.1f} ms")
    print(f"  - ActionHead:     ~{action_head_est_ms:.1f} ms")
    print(f"  - Total Action:    ~{action_time_ms:.1f} ms")

    # Per-sample breakdown
    print("\nPer-Sample Breakdown (Batch Size = 2):")
    print(f"  - VLM per sample:  {vlm_time_ms/2:.2f} ms")
    print(f"  - Action per sample: {action_time_ms/2:.2f} ms")
    print(f"  - Total per sample: {full_time_ms/2:.2f} ms")

    # FPS estimate
    fps_batch2 = 1000 / full_time_ms * 2  # Batch size 2
    fps_single = 1000 / (full_time_ms / 2)  # Per sample
    print(f"\nFPS Estimate:")
    print(f"  - Batch size 2: {fps_batch2:.1f} FPS")
    print(f"  - Single sample: {fps_single:.1f} FPS")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    if vlm_time_ms > full_time_ms * 0.6:
        print("⚠️  VLM dominates inference time (>60%)")
        print("   → Consider reducing sequence length or using smaller VLM")
    if action_time_ms < vlm_time_ms * 0.3:
        print("✓ ActionHead is lightweight (<30% of VLM time)")
        print("   → Good! Stereo + ActionHead doesn't add much overhead")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark QwenZone inference speed")
    parser.add_argument("--num_warmup", type=int, default=3, help="Number of warmup runs")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument("--task", type=str, default="click_bell", help="Task name")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")

    args = parser.parse_args()

    benchmark_inference(
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
        task_name=args.task,
        config_path=args.config,
    )
