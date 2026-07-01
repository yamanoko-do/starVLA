#!/usr/bin/env python
"""Batch visualize disparity maps from RoboTwin dataset samples."""

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


def visualize_dataset_samples(
    num_samples: int = 10,
    output_dir: str = "/tmp/stereo_disp_viz",
    task_name: str = "click_bell",
):
    """Load dataset samples and save disparity visualizations.

    Args:
        num_samples: Number of samples to visualize
        output_dir: Directory to save visualizations
        task_name: RoboTwin task name
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Visualizing {num_samples} samples from {task_name}")
    print("=" * 70)

    # Load dataset
    print("\n[1] Loading dataset...")
    dataset = HDF5RobotwinDataset(
        dataset_path=f"/mnt/workspace/yama/RoboTwin/data/{task_name}",
        T=1,
        H=50,
        img_size=224,
        stereo_size=(224, 224),
    )
    print(f"✓ Dataset loaded: {len(dataset)} samples")

    # Load StereoEncoder with disparity output
    print("\n[2] Loading StereoEncoder...")
    ckpt = "/mnt/workspace/yama/OpenStereo/output/MultiDataset/WAVEStereo/wavestereo_mixdataset/20260623_wavestereo_filter/ckpt/checkpoint_epoch_2.pth"

    encoder = StereoEncoder(
        ckpt_path=ckpt,
        update_iters=4,
        hidden_dim=512,
        N_stereo_tokens=64,
        input_size=(224, 224),
        return_disp=True,
    ).cuda()
    print(f"✓ StereoEncoder ready")

    # Process each sample
    print(f"\n[3] Processing {num_samples} samples...")
    for idx in range(min(num_samples, len(dataset))):
        sample = dataset[idx]

        # Get stereo images
        left_arr = np.array(sample["stereo_left"][0])
        right_arr = np.array(sample["stereo_right"][0])

        # Prepare tensors
        left = torch.from_numpy(left_arr).permute(2, 0, 1).unsqueeze(0).cuda().float() / 255.0
        right = torch.from_numpy(right_arr).permute(2, 0, 1).unsqueeze(0).cuda().float() / 255.0

        # Forward pass
        with torch.no_grad():
            tokens, disparity = encoder(left, right)

        # Convert to depth
        depth = encoder.disparity_to_depth(disparity, focal_length=500, baseline=0.06)

        # Save visualization
        save_path = output_dir / f"sample_{idx:03d}_disp.png"

        # Extract data for visualization
        left_cpu = left.cpu()
        disparity_cpu = disparity.cpu()
        depth_cpu = depth.cpu()

        # Create visualization with matplotlib
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Left image
        left_img = left_cpu[0].numpy().transpose(1, 2, 0)
        axes[0, 0].imshow(left_img)
        axes[0, 0].set_title("Left Image")
        axes[0, 0].axis("off")

        # Right image
        right_img = right.cpu()[0].numpy().transpose(1, 2, 0)
        axes[0, 1].imshow(right_img)
        axes[0, 1].set_title("Right Image")
        axes[0, 1].axis("off")

        # Disparity
        disp_map = disparity_cpu[0, 0].numpy()
        disp_vis = (disp_map - disp_map.min()) / (disp_map.max() - disp_map.min() + 1e-6)
        im1 = axes[1, 0].imshow(disp_vis, cmap="jet")
        axes[1, 0].set_title(f"Disparity (pixels)\nRange: [{disp_map.min():.1f}, {disp_map.max():.1f}]")
        axes[1, 0].axis("off")
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)

        # Depth
        depth_map = depth_cpu[0, 0].numpy()
        depth_vis = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
        im2 = axes[1, 1].imshow(depth_vis, cmap="jet")
        axes[1, 1].set_title(f"Depth (meters)\nRange: [{depth_map.min():.2f}, {depth_map.max():.2f}]")
        axes[1, 1].axis("off")
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"  [{idx+1}/{num_samples}] Saved: {save_path.name}")

    print(f"\n✓ Done! Visualizations saved to: {output_dir}")
    print(f"  Total: {num_samples} samples")

    # Create a summary grid
    print(f"\n[4] Creating summary grid...")
    create_summary_grid(output_dir, num_samples)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ Processed {num_samples} samples")
    print(f"✓ Individual visualizations: {output_dir}/sample_XXX_disp.png")
    print(f"✓ Summary grid: {output_dir}/summary_grid.png")
    print("\nCheck the visualizations to verify:")
    print("  • Disparity shows smooth depth structure")
    print("  • Closer objects have higher disparity (warmer colors)")
    print("  • Farther objects have lower disparity (cooler colors)")


def create_summary_grid(output_dir: Path, num_samples: int):
    """Create a summary grid with all samples."""

    import matplotlib.pyplot as plt
    from PIL import Image

    # Load all disparity maps
    nrows = int(np.ceil(num_samples / 5))
    ncols = min(5, num_samples)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for idx in range(num_samples):
        row = idx // ncols
        col = idx % ncols

        img_path = output_dir / f"sample_{idx:03d}_disp.png"
        if img_path.exists():
            img = Image.open(img_path)
            axes[row, col].imshow(img)
            axes[row, col].set_title(f"Sample {idx}")
            axes[row, col].axis("off")

    # Hide empty subplots
    for idx in range(num_samples, nrows * ncols):
        row = idx // ncols
        col = idx % cols
        axes[row, col].axis("off")

    plt.tight_layout()
    summary_path = output_dir / "summary_grid.png"
    plt.savefig(summary_path, dpi=150)
    plt.close()

    print(f"  ✓ Summary grid saved: {summary_path.name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize disparity from dataset")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to visualize")
    parser.add_argument("--output_dir", type=str, default="/tmp/stereo_disp_viz", help="Output directory")
    parser.add_argument("--task", type=str, default="click_bell", help="Task name")

    args = parser.parse_args()

    visualize_dataset_samples(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        task_name=args.task,
    )
