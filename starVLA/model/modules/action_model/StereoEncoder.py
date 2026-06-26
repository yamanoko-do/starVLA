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
        input_size: tuple = (256, 256),  # stereo image resize (H, W)
    ):
        super().__init__()
        self.update_iters = update_iters
        self.hidden_dim = hidden_dim
        self.N_stereo_tokens = N_stereo_tokens
        self.input_size = tuple(input_size)
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
            for itr in range(self.update_iters):
                disp = disp.detach()
                corr = geo_fn(disp)
                net, delta_disp, _mask_feat = model.update_block(
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

        return tokens  # [B, N_stereo_tokens, hidden_dim]


if __name__ == "__main__":
    ckpt = "/mnt/workspace/yama/OpenStereo/output/MultiDataset/WAVEStereo/wavestereo_mixdataset/20260623_wavestereo_filter/ckpt/checkpoint_epoch_2.pth"
    encoder = StereoEncoder(ckpt_path=ckpt, hidden_dim=512, N_stereo_tokens=64, input_size=(256, 256)).cuda()
    n = sum(p.numel() for p in encoder.parameters()) / 1e6
    n_train = sum(p.numel() for p in encoder.parameters() if p.requires_grad) / 1e6
    print(f"StereoEncoder params: {n:.1f}M total, {n_train:.1f}M trainable (fuse blocks)")

    left = torch.randn(2, 3, 256, 256, device="cuda")
    right = torch.randn(2, 3, 256, 256, device="cuda")
    with torch.no_grad():
        tokens = encoder(left, right)
    print(f"output: {tuple(tokens.shape)}  (expected (2, 64, 512))")
