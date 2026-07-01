# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""StereoEncoder with depth output for validation/debug.

Extends the original StereoEncoder to optionally output disparity/depth maps
for validating the stereo pipeline correctness.
"""

import sys
from pathlib import Path

_OPENSTEREO_ROOT = Path(__file__).resolve().parents[5] / "OpenStereo"
if str(_OPENSTEREO_ROOT) not in sys.path:
    sys.path.insert(0, str(_OPENSTEREO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from starVLA.model.modules.action_model.StereoEncoder import (
    StereoEncoder, ResConvBlock, _OPENSTEREO_ROOT
)
from stereo.modeling.disp_pred.disp_regression import disparity_regression


class StereoEncoderDebug(StereoEncoder):
    """StereoEncoder with optional disparity/depth output for validation.

    Usage:
        encoder = StereoEncoderDebug(..., return_disp=True)
        tokens, disp = encoder(left, right)  # disp: [B, 1, H, W] disparity map
        # Or convert to depth:
        depth = encoder.disparity_to_depth(disp, focal_length, baseline)

    The disparity output uses the same forward path as WaveStereo, including
    all update_iters iterations, so it reflects the actual stereo quality.
    """

    def __init__(
        self,
        ckpt_path: str,
        config_path: str | None = None,
        update_iters: int = 4,
        hidden_dim: int = 512,
        N_stereo_tokens: int = 64,
        input_size: tuple = (256, 256),
        return_disp: bool = False,  # NEW: whether to return disparity map
    ):
        super().__init__(
            ckpt_path=ckpt_path,
            config_path=config_path,
            update_iters=update_iters,
            hidden_dim=hidden_dim,
            N_stereo_tokens=N_stereo_tokens,
            input_size=input_size,
        )
        self.return_disp = return_disp

        # Load upsample modules from original WaveStereo (needed for full-res disparity)
        if return_disp:
            from easydict import EasyDict
            from stereo.utils.common_utils import config_loader
            from stereo.modeling.models.wavestereo.wavestereo import WAVEStereo

            if config_path is None:
                config_path = str(_OPENSTEREO_ROOT / "cfgs/wavestereo/wavestereo_mixdataset.yaml")
            raw_cfg = config_loader(config_path)
            full_model = WAVEStereo(EasyDict(raw_cfg["MODEL"]))
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            full_model.load_state_dict(ckpt["model_state"], strict=False)

            # Extract upsample modules
            self.stem_2 = full_model.stem_2
            self.stem_1 = full_model.stem_1
            self.spx_2_gru = full_model.spx_2_gru
            self.spx = full_model.spx
            self.spx_1_gru = full_model.spx_1_gru
            self.spx_gru = full_model.spx_gru

            for p in self.stem_2.parameters():
                p.requires_grad = False
            for p in self.stem_1.parameters():
                p.requires_grad = False
            for p in self.spx_2_gru.parameters():
                p.requires_grad = False
            for p in self.spx.parameters():
                p.requires_grad = False
            for p in self.spx_1_gru.parameters():
                p.requires_grad = False
            for p in self.spx_gru.parameters():
                p.requires_grad = False

    def forward(
        self, left_img, right_img
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward with optional disparity output.

        Args:
            left_img: [B, 3, H, W]
            right_img: [B, 3, H, W]

        Returns:
            if return_disp=False: tokens [B, N_stereo_tokens, hidden_dim]
            if return_disp=True: (tokens [B, N_stereo_tokens, hidden_dim],
                                 disparity [B, 1, H_out, W_out])
            where H_out, W_out match the input resolution (after padding).
        """
        B = left_img.shape[0]
        model = self._get_model()

        if next(model.parameters()).device != left_img.device:
            model = model.to(left_img.device)

        self.float()

        with torch.amp.autocast("cuda", enabled=False):
            left_img = left_img.float()
            right_img = right_img.float()

            from stereo.modeling.models.wavestereo.utils import InputPadder
            padder = InputPadder(left_img.shape, divis_by=32)
            left_img, right_img = padder.pad(left_img_right=None, left_img=left_img, right_img=right_img)

            # === WaveStereo forward (same as original) ===
            features_left = model.backbone(left_img)
            features_right = model.backbone(right_img)

            from stereo.modeling.cost_volume.cost_volume import correlation_volume
            cost_volume = correlation_volume(
                features_left[0], features_right[0], model.max_disp // 4
            )
            encoding_volume = model.cost_agg(cost_volume, features_left)

            hidden = model.hnet(features_left[0])
            net = torch.tanh(hidden)
            context = list(
                model.context_zqr_conv(features_left[0]).split(
                    split_size=model.hidden_dim, dim=1
                )
            )

            unsqueezed_encoding = encoding_volume[0].reshape(
                B, -1, encoding_volume[0].size(1),
                encoding_volume[0].size(2), encoding_volume[0].size(3),
            )
            prob = F.softmax(encoding_volume[0], dim=1)
            init_disp = disparity_regression(prob, model.max_disp // 4)

            geo_fn = model.Geo_Encoding_Volume(
                unsqueezed_encoding.float(),
                radius=model.corr_radius,
                num_levels=model.corr_levels,
            )

            disp = init_disp
            mask_feat = None  # Will be set in loop
            for itr in range(self.update_iters):
                disp = disp.detach()
                corr = geo_fn(disp)
                net, delta_disp, mask_feat = model.update_block(
                    net, context,
                    feat_left=features_left[0], feat_right=features_right[0],
                    disp=disp, corr=corr, itr=itr,
                )
                disp = disp + delta_disp

            # === Extract stereo tokens (same as original) ===
            fused_in = torch.cat([encoding_volume[0], net], dim=1)  # [B, 112, H/4, W/4]
            fused = self.fuse(fused_in)  # [B, 512, H/8, W/8]
            pooled = F.adaptive_avg_pool2d(fused, (self.grid, self.grid))  # [B, 512, 8, 8]
            tokens = pooled.permute(0, 2, 3, 1).flatten(1, 2)  # [B, 64, 512]

            if not self.return_disp:
                return tokens

            # === Upsample disparity to full resolution ===
            # stem features needed for upsampling
            stem_2x = self.stem_2(left_img)
            stem_1x = self.stem_1(left_img)

            # Up sample disparity (same as WaveStereo.upsample_disp)
            xspx = self.spx_2_gru(mask_feat, stem_2x)
            xspx = self.spx(xspx)
            xspx = self.spx_1_gru(xspx, stem_1x)
            spx_pred = self.spx_gru(xspx)
            spx_pred = F.softmax(spx_pred, 1)

            up_disp = model.context_upsample(disp * 4., spx_pred).unsqueeze(1)  # [B, 1, H, W]

            # Unpad to original size
            H, W = left_img.shape[2], left_img.shape[3]
            up_disp = padder.unpad(up_disp)
            up_disp = F.interpolate(up_disp, (H, W), mode='bilinear', align_corners=True)

            return tokens, up_disp

    @staticmethod
    def disparity_to_depth(
        disparity: torch.Tensor,
        focal_length: float,
        baseline: float = 0.06,  # 6cm baseline for typical stereo rigs
    ):
        """Convert disparity map to depth map.

        Args:
            disparity: [B, 1, H, W] in pixels
            focal_length: focal length in pixels
            baseline: baseline distance in meters

        Returns:
            depth: [B, 1, H, W] in meters
        """
        # depth = (focal_length * baseline) / disparity
        depth = (focal_length * baseline) / (disparity + 1e-6)  # Add eps to avoid div/0
        return depth

    @staticmethod
    def visualize_disparity(
        disparity: torch.Tensor,
        left_img: torch.Tensor = None,
        save_path: str = None,
    ):
        """Visualize disparity map (for debug).

        Args:
            disparity: [B, 1, H, W] or [H, W]
            left_img: [B, 3, H, W] or [3, H, W] (optional, for side-by-side)
            save_path: if provided, save to file
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if disparity.dim() == 4:
            disparity = disparity[0, 0].cpu().numpy()
        else:
            disparity = disparity.cpu().numpy()

        # Normalize for visualization
        disp_vis = (disparity - disparity.min()) / (disparity.max() - disparity.min() + 1e-6)

        if left_img is not None:
            if left_img.dim() == 4:
                left_img = left_img[0].cpu().numpy()
                left_img = left_img.transpose(1, 2, 0)
            else:
                left_img = left_img.cpu().numpy()
                left_img = left_img.transpose(1, 2, 0)

            # denormalize if needed
            if left_img.max() <= 1.0:
                left_img = (left_img * 255).astype(np.uint8)
            else:
                left_img = left_img.astype(np.uint8)

        fig, axes = plt.subplots(1, 2 if left_img is not None else 1, figsize=(12, 5))

        if left_img is not None:
            axes[0].imshow(left_img)
            axes[0].set_title("Left Image")
            axes[0].axis("off")
            axes[1].imshow(disp_vis, cmap="jet")
            axes[1].set_title("Disparity")
            axes[1].axis("off")
        else:
            axes.imshow(disp_vis, cmap="jet")
            axes.set_title("Disparity")
            axes.axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    """Test stereo encoding and visualize disparity."""
    ckpt = "/mnt/workspace/yama/OpenStereo/output/MultiDataset/WAVEStereo/wavestereo_mixdataset/20260623_wavestereo_filter/ckpt/checkpoint_epoch_2.pth"

    print("Testing StereoEncoderDebug...")

    # Test with return_disp=True
    encoder = StereoEncoderDebug(
        ckpt_path=ckpt,
        update_iters=4,
        hidden_dim=512,
        N_stereo_tokens=64,
        input_size=(256, 256),
        return_disp=True,
    ).cuda()

    # Load a real stereo pair from dataset
    from PIL import Image
    import torch
    from starVLA.dataloader.gr00t_lerobot.hdf5_robotwin_dataset import HDF5RobotwinDataset

    dataset = HDF5RobotwinDataset(
        dataset_path="/mnt/workspace/yama/RoboTwin/data/click_bell",
        T=1,
        H=50,
        img_size=224,
        stereo_size=(256, 256),
    )

    # Get one sample
    sample = dataset[0]
    import numpy as np
    left = torch.from_numpy(np.array(sample["stereo_left"][0])).permute(2, 0, 1).unsqueeze(0).cuda().float() / 255.0
    right = torch.from_numpy(np.array(sample["stereo_right"][0])).permute(2, 0, 1).unsqueeze(0).cuda().float() / 255.0

    print(f"Input shape: left={left.shape}, right={right.shape}")

    # Forward with disparity
    with torch.no_grad():
        tokens, disparity = encoder(left, right)

    print(f"Output: tokens={tokens.shape}, disparity={disparity.shape}")
    print(f"Disparity range: [{disparity.min():.3f}, {disparity.max():.3f}] pixels")

    # Visualize
    save_path = "/tmp/stereo_debug_output.png"
    encoder.visualize_disparity(disparity, left, save_path)
    print(f"Visualization saved to {save_path}")

    # Test depth conversion (assuming focal_length ≈ 500 pixels, baseline = 0.06m)
    depth = encoder.disparity_to_depth(disparity, focal_length=500, baseline=0.06)
    print(f"Depth range: [{depth.min():.3f}, {depth.max():.3f}] meters")
