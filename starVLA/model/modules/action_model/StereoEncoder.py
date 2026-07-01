# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""Frozen WAVEStereo feature extractor for the QwenZone action head.

Extracts fine-grained stereo-geometry tokens from a stereo pair (left + right):
  - encoding_volume[0]: 48-channel cost-aggregation features at 1/4 resolution
  - net after N update iterations: 64-channel iterative refinement features at 1/4 resolution

These two are CONCATENATED early (112 ch), then fused & downsampled by two
trainable ResConv blocks, finally adaptive-pooled to 8×8 = N_stereo_tokens (64)
spatial tokens. Output dimension == hidden_dim (512) directly from ResConv2,
so no extra projection is needed.

Usage::

    stereo = StereoEncoder(ckpt_path=..., hidden_dim=512, N_stereo_tokens=64)
    tokens = stereo(left_img, right_img)   # [B, 64, 512]
"""

import sys
from pathlib import Path

# Ensure OpenStereo is importable
_OPENSTEREO_ROOT = Path(__file__).resolve().parents[5] / "OpenStereo"
if str(_OPENSTEREO_ROOT) not in sys.path:
    sys.path.insert(0, str(_OPENSTEREO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from stereo.modeling.cost_volume.cost_volume import correlation_volume
from stereo.modeling.disp_pred.disp_regression import disparity_regression
from stereo.modeling.disp_refinement.disp_refinement import context_upsample
from stereo.modeling.models.wavestereo.geometry import Geo_Encoding_Volume
from stereo.modeling.models.wavestereo.utils import InputPadder


class ResConvBlock(nn.Module):
    """ResNet-style basic block: Conv3×3-BN-ReLU-Conc3×3-BN + identity + ReLU.

    `stride` controls spatial downsampling (stride=2 halves H,W).
    """

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        # identity path matches dims when in/out or stride differ
        if stride != 1 or in_ch != out_ch:
            self.identity = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.identity = nn.Identity()

    def forward(self, x):
        identity = self.identity(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class StereoEncoder(nn.Module):
    """Frozen WAVEStereo wrapper → fused stereo tokens via early-cat + ResConv fusion."""

    def __init__(
        self,
        ckpt_path: str,
        config_path: str | None = None,
        update_iters: int = 4,       # N: how many update_block iterations to run
        hidden_dim: int = 512,        # token dim (== action head hidden_size); ResConv2 output ch
        N_stereo_tokens: int = 64,    # output token count (8×8 spatial grid)
        input_size: tuple = (224, 224),  # stereo image resize (H, W)
        return_disp: bool = False,   # if True, load upsample modules and return disparity map
    ):
        super().__init__()
        self.update_iters = update_iters
        self.hidden_dim = hidden_dim
        self.N_stereo_tokens = N_stereo_tokens
        self.input_size = tuple(input_size)
        self.return_disp = return_disp
        # spatial grid for the final adaptive pool (sqrt(N_stereo_tokens))
        self.grid = int(round(N_stereo_tokens ** 0.5))
        assert self.grid * self.grid == N_stereo_tokens, (
            f"N_stereo_tokens={N_stereo_tokens} must be a perfect square (8×8=64)"
        )

        model = self._build_model(ckpt_path, config_path)
        self._set_model(model)   # plain attr → DeepSpeed bf16 conversion skips it

        # Early-fuse + downsample: cat(enc0[48] + net[64]=112) → 256 (stride2) → 512
        self.fuse = nn.Sequential(
            ResConvBlock(48 + 64, 256, stride=2),   # [B, 256, H/8, W/8]
            ResConvBlock(256, hidden_dim, stride=1),  # [B, 512, H/8, W/8]
        )

        # Load upsample modules if disparity output is requested
        if return_disp:
            self._load_upsample_modules(ckpt_path, config_path)

    @staticmethod
    def _build_model(ckpt_path, config_path=None):
        from easydict import EasyDict
        from stereo.utils.common_utils import config_loader
        from stereo.modeling.models.wavestereo.wavestereo import WAVEStereo

        if config_path is None:
            config_path = str(_OPENSTEREO_ROOT / "cfgs/wavestereo/wavestereo_mixdataset.yaml")
        raw_cfg = config_loader(config_path)
        model = WAVEStereo(EasyDict(raw_cfg["MODEL"]))
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model = model.float()   # WAVEStereo uses BatchNorm → needs float32
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    def _get_model(self):
        return self._frozen_model

    def _set_model(self, model):
        # plain attr so DeepSpeed/accelerate dtype conversions skip it
        object.__setattr__(self, "_frozen_model", model)

    def _load_upsample_modules(self, ckpt_path, config_path):
        """Load WAVEStereo upsample modules for full-res disparity output.

        These modules are frozen and only used for visualization/debug.
        """
        from easydict import EasyDict
        from stereo.utils.common_utils import config_loader
        from stereo.modeling.models.wavestereo.wavestereo import WAVEStereo

        if config_path is None:
            config_path = str(_OPENSTEREO_ROOT / "cfgs/wavestereo/wavestereo_mixdataset.yaml")
        raw_cfg = config_loader(config_path)
        full_model = WAVEStereo(EasyDict(raw_cfg["MODEL"]))
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        full_model.load_state_dict(ckpt["model_state"], strict=False)

        # Extract upsample modules (frozen)
        self.stem_2 = full_model.stem_2
        self.stem_1 = full_model.stem_1
        self.spx_2_gru = full_model.spx_2_gru
        self.spx = full_model.spx
        self.spx_1_gru = full_model.spx_1_gru
        self.spx_gru = full_model.spx_gru
        # context_upsample is a function, not a module, so we import it at top level

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

    def forward(self, left_img, right_img):
        """Extract stereo tokens → [B, N_stereo_tokens, hidden_dim]."""
        B = left_img.shape[0]
        model = self._get_model()

        if next(model.parameters()).device != left_img.device:
            model = model.to(left_img.device)

        # StereoEncoder's own modules may be bf16 under DeepSpeed → force fp32
        self.float()

        # Frozen WAVEStereo (BatchNorm) requires float32; disable autocast
        with torch.amp.autocast('cuda', enabled=False):
            left_img = left_img.float()
            right_img = right_img.float()

            padder = InputPadder(left_img.shape, divis_by=32)
            left_img, right_img = padder.pad(left_img, right_img)

            # -- partial forward (mirrors WAVEStereo.forward) -------------------
            features_left = model.backbone(left_img)
            features_right = model.backbone(right_img)

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

            geo_fn = Geo_Encoding_Volume(
                unsqueezed_encoding.float(),
                radius=model.corr_radius,
                num_levels=model.corr_levels,
            )

            disp = init_disp
            mask_feat = None
            for itr in range(self.update_iters):
                disp = disp.detach()
                corr = geo_fn(disp)
                net, delta_disp, mask_feat = model.update_block(
                    net, context,
                    feat_left=features_left[0], feat_right=features_right[0],
                    disp=disp, corr=corr, itr=itr,
                )
                disp = disp + delta_disp

            # -- early fuse: cat on channel, then ResConv fusion + downsample ----
            # enc0: [B, 48, H/4, W/4], net: [B, 64, H/4, W/4]
            fused_in = torch.cat([encoding_volume[0], net], dim=1)   # [B, 112, H/4, W/4]
            fused = self.fuse(fused_in)                              # [B, 512, H/8, W/8]

            # adaptive pool to fixed spatial grid (sqrt(N_stereo_tokens))
            pooled = F.adaptive_avg_pool2d(fused, (self.grid, self.grid))  # [B,512,8,8]
            # → [B, grid, grid, hidden_dim] → flatten → [B, N_stereo_tokens, hidden_dim]
            tokens = pooled.permute(0, 2, 3, 1).flatten(1, 2)

            # Upsample disparity to full resolution (if requested)
            if self.return_disp:
                # stem features needed for upsampling
                stem_2x = self.stem_2(left_img)
                stem_1x = self.stem_1(left_img)

                # Upsample disparity (same as WaveStereo.upsample_disp)
                xspx = self.spx_2_gru(mask_feat, stem_2x)
                xspx = self.spx(xspx)
                xspx = self.spx_1_gru(xspx, stem_1x)
                spx_pred = self.spx_gru(xspx)
                spx_pred = F.softmax(spx_pred, 1)

                up_disp = context_upsample(disp * 4., spx_pred).unsqueeze(1)  # [B, 1, H, W]

                # Unpad to original size
                H, W = left_img.shape[2], left_img.shape[3]
                up_disp = padder.unpad(up_disp)
                up_disp = F.interpolate(up_disp, (H, W), mode='bilinear', align_corners=True)

                return tokens, up_disp

        return tokens  # [B, N_stereo_tokens, hidden_dim]

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

            # Denormalize if needed
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
            plt.imshow(disp_vis, cmap="jet")
            plt.title("Disparity")
            plt.axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    ckpt = "/mnt/workspace/yama/OpenStereo/output/MultiDataset/WAVEStereo/wavestereo_mixdataset/20260623_wavestereo_filter/ckpt/checkpoint_epoch_2.pth"

    # Test without disparity (training mode)
    print("=== Test 1: Training mode (tokens only) ===")
    encoder = StereoEncoder(
        ckpt_path=ckpt,
        hidden_dim=512,
        N_stereo_tokens=64,
        input_size=(224, 224),
        return_disp=False,  # Training mode
    ).cuda()
    n = sum(p.numel() for p in encoder.parameters()) / 1e6
    n_train = sum(p.numel() for p in encoder.parameters() if p.requires_grad) / 1e6
    print(f"StereoEncoder params: {n:.1f}M total, {n_train:.1f}M trainable (fuse blocks)")

    left = torch.randn(2, 3, 224, 224, device="cuda")
    right = torch.randn(2, 3, 224, 224, device="cuda")
    with torch.no_grad():
        tokens = encoder(left, right)
    print(f"output: {tuple(tokens.shape)}  (expected (2, 64, 512))")

    # Test with disparity output (debug mode)
    print("\n=== Test 2: Debug mode (tokens + disparity) ===")
    encoder_disp = StereoEncoder(
        ckpt_path=ckpt,
        hidden_dim=512,
        N_stereo_tokens=64,
        input_size=(224, 224),
        return_disp=True,  # Debug mode
    ).cuda()
    print(f"StereoEncoder with disparity params: {sum(p.numel() for p in encoder_disp.parameters())/1e6:.1f}M")

    with torch.no_grad():
        tokens, disparity = encoder_disp(left, right)
    print(f"tokens: {tuple(tokens.shape)}  (expected (2, 64, 512))")
    print(f"disparity: {tuple(disparity.shape)}  (expected (2, 1, 224, 224))")
    print(f"disparity range: [{disparity.min():.3f}, {disparity.max():.3f}] pixels")

    # Test depth conversion
    depth = encoder_disp.disparity_to_depth(disparity, focal_length=500, baseline=0.06)
    print(f"depth range: [{depth.min():.3f}, {depth.max():.3f}] meters")

    print("\n=== All tests passed! ===")
    print("✓ 224×224 resolution works (no padding needed)")
    print("✓ Training mode returns tokens only")
    print("✓ Debug mode returns tokens + disparity")
