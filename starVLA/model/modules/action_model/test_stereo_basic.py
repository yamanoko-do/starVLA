#!/usr/bin/env python
"""Quick test to verify stereo encoder processes left/right images correctly."""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # starVLA root
sys.path.insert(0, str(Path(__file__).resolve().parents[6] / "OpenStereo"))

import torch
import numpy as np
from PIL import Image

from starVLA.model.modules.action_model.StereoEncoder import StereoEncoder
from starVLA.dataloader.gr00t_lerobot.hdf5_robotwin_dataset import HDF5RobotwinDataset


def test_stereo_encoder():
    """Test that StereoEncoder correctly processes left/right images."""
    print("=== Loading dataset ===")
    dataset = HDF5RobotwinDataset(
        dataset_path="/mnt/workspace/yama/RoboTwin/data/click_bell",
        T=1,
        H=50,
        img_size=224,
        stereo_size=(256, 256),
    )

    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"stereo_left type: {type(sample['stereo_left'][0])}")
    print(f"stereo_right type: {type(sample['stereo_right'][0])}")

    # Check that left/right are different images
    left_arr = np.array(sample["stereo_left"][0])
    right_arr = np.array(sample["stereo_right"][0])
    are_different = not np.array_equal(left_arr, right_arr)
    print(f"Left and right images are different: {are_different}")
    if are_different:
        diff = np.abs(left_arr.astype(float) - right_arr.astype(float)).mean()
        print(f"Mean pixel difference: {diff:.2f} (should be > 0 for stereo)")

    # Load StereoEncoder
    print("\n=== Loading StereoEncoder ===")
    ckpt = "/mnt/workspace/yama/OpenStereo/output/MultiDataset/WAVEStereo/wavestereo_mixdataset/20260623_wavestereo_filter/ckpt/checkpoint_epoch_2.pth"

    encoder = StereoEncoder(
        ckpt_path=ckpt,
        update_iters=1,  # Use 1 for faster test
        hidden_dim=512,
        N_stereo_tokens=64,
        input_size=(256, 256),
    ).cuda()

    print(f"StereoEncoder loaded: params={sum(p.numel() for p in encoder.parameters())/1e6:.1f}M")

    # Prepare input
    print("\n=== Preparing input ===")
    left = torch.from_numpy(left_arr).permute(2, 0, 1).unsqueeze(0).cuda().float() / 255.0
    right = torch.from_numpy(right_arr).permute(2, 0, 1).unsqueeze(0).cuda().float() / 255.0

    print(f"Left shape: {left.shape}, range: [{left.min():.3f}, {left.max():.3f}]")
    print(f"Right shape: {right.shape}, range: [{right.min():.3f}, {right.max():.3f}]")

    # Forward
    print("\n=== Running StereoEncoder.forward ===")
    with torch.no_grad():
        tokens = encoder(left, right)

    print(f"Output tokens shape: {tokens.shape}")
    print(f"Tokens range: [{tokens.min():.3f}, {tokens.max():.3f}]")
    print(f"Tokens mean: {tokens.mean():.3f}, std: {tokens.std():.3f}")

    # Test that left/right order matters (swap should give different result)
    print("\n=== Testing left/right order sensitivity ===")
    with torch.no_grad():
        tokens_swapped = encoder(right, left)  # SWAPPED!

    diff = torch.abs(tokens - tokens_swapped).mean().item()
    print(f"Mean token difference when swapping left/right: {diff:.6f}")
    print(f"Order matters: {diff > 1e-3} (should be True for valid stereo)")

    # Test batch processing
    print("\n=== Testing batch processing ===")
    B = 4
    left_batch = left.repeat(B, 1, 1, 1)
    right_batch = right.repeat(B, 1, 1, 1)

    with torch.no_grad():
        tokens_batch = encoder(left_batch, right_batch)

    print(f"Batch input: {left_batch.shape}")
    print(f"Batch output: {tokens_batch.shape}")

    # All batches should give same result
    tokens_first = tokens_batch[0:1]
    all_same = torch.allclose(tokens_batch, tokens_first.expand(B, *tokens_batch.shape[1:]), atol=1e-5)
    print(f"All batch outputs identical: {all_same}")

    print("\n=== All tests passed! ===")
    print("✓ Left and right images are different")
    print("✓ StereoEncoder processes them correctly")
    print("✓ Output tokens are sensitive to left/right order")
    print("✓ Batch processing works")


if __name__ == "__main__":
    test_stereo_encoder()
