# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""Frozen WAVEStereo feature extractor for the QwenZone action head.

Extracts fine-grained stereo-geometry tokens from a stereo pair (left + right):
  - encoding_volume[0]: 48-channel cost-aggregation features at 1/4 resolution
  - net after N update iterations: 64-channel iterative refinement features at 1/4 resolution

Both are pooled to a fixed spatial grid, projected, and concatenated into
``N_stereo_tokens`` of ``hidden_dim`` each, fed to the action-head transformer.

Usage::

    stereo = StereoEncoder(cfg=stereo_cfg)
    tokens = stereo(left_img, right_img)          # [B, N_stereo, hidden_dim]
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


class StereoEncoder(nn.Module):
    """Frozen WAVEStereo wrapper that extracts intermediate features as learnable tokens."""

    def __init__(
        self,
        ckpt_path: str,
        config_path: str | None = None,
        update_iters: int = 4,       # N: how many update_block iterations to run
        pool_size: int = 8,           # K: spatial grid for pooling feature maps
        hidden_dim: int = 512,        # token embedding dimension (must match action head)
        N_stereo_tokens: int = 8,     # output token count
    ):
        super().__init__()
        self.update_iters = update_iters
        self.pool_size = pool_size
        self.hidden_dim = hidden_dim
        self.N_stereo_tokens = N_stereo_tokens

        model = self._build_model(ckpt_path, config_path)
        self._set_model(model)   # plain attr → DeepSpeed bf16 conversion skips it

        # encoding_volume[0]: 48 channels → project
        self.encoding_proj = nn.Sequential(
            nn.LayerNorm(48),
            nn.Linear(48, hidden_dim),
        )
        # net (after N iters): 64 channels → project
        self.net_proj = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, hidden_dim),
        )
        # merge pooled encoding + net tokens → N_stereo tokens
        total_spatial = 2 * pool_size * pool_size
        self.merger = nn.Sequential(
            nn.LayerNorm(total_spatial * hidden_dim),
            nn.Linear(total_spatial * hidden_dim, N_stereo_tokens * hidden_dim),
        )

    @staticmethod
    def _build_model(ckpt_path, config_path=None):
        """Build WAVEStereo and load checkpoint weights."""
        from easydict import EasyDict
        from stereo.utils.common_utils import config_loader
        from stereo.modeling.models.wavestereo.wavestereo import WAVEStereo

        if config_path is None:
            config_path = str(
                _OPENSTEREO_ROOT
                / "cfgs/wavestereo/wavestereo_mixdataset.yaml"
            )

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
        """Access the frozen WAVEStereo (stored as plain attr to avoid DeepSpeed bf16)."""
        return self._frozen_model

    def _set_model(self, model):
        # Keep as plain attribute so DeepSpeed/accelerate dtype conversions skip it
        object.__setattr__(self, "_frozen_model", model)

    def forward(self, left_img, right_img):
        """Extract stereo-geometry tokens."""
        B = left_img.shape[0]
        model = self._get_model()  # frozen WAVEStereo (plain attr, not nn.Module child)

        # Ensure model is on the same device as input (plain attr skips .cuda())
        if next(model.parameters()).device != left_img.device:
            model = model.to(left_img.device)

        # DeepSpeed bf16 training converts all nn.Parameters to bf16. The frozen
        # WAVEStereo (plain attr) escapes this, but StereoEncoder's own projection
        # layers don't. Force everything to float32 for this forward pass.
        self.float()

        # Frozen WAVEStereo (BatchNorm) requires float32. Disable autocast so the
        # entire partial forward runs in fp32 regardless of the caller's context.
        with torch.amp.autocast('cuda', enabled=False):
            left_img = left_img.float()
            right_img = right_img.float()

            # pad to dimensions divisible by 32 (FPN downsampling requirement)
            padder = InputPadder(left_img.shape, divis_by=32)
            left_img, right_img = padder.pad(left_img, right_img)

            # -- partial forward (mirrors WAVEStereo.forward lines 75--105) ----------
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

            # -- extract tokens --------------------------------------------------
            enc0 = encoding_volume[0]
            enc_pool = F.adaptive_avg_pool2d(enc0, (self.pool_size, self.pool_size))
            enc_pool = enc_pool.permute(0, 2, 3, 1).flatten(1, 2)
            enc_tok = self.encoding_proj(enc_pool)    # [B, K², hd] — fp32 here

            net_pool = F.adaptive_avg_pool2d(net, (self.pool_size, self.pool_size))
            net_pool = net_pool.permute(0, 2, 3, 1).flatten(1, 2)
            net_tok = self.net_proj(net_pool)          # [B, K², hd] — fp32 here

        all_tokens = torch.cat([enc_tok, net_tok], dim=1)           # [B, 2K², hd]
        merged = self.merger(all_tokens.flatten(1))                 # [B, N_stereo*hd]
        return merged.view(B, self.N_stereo_tokens, self.hidden_dim)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/mnt/workspace/yama/OpenStereo")

    ckpt = "/mnt/workspace/yama/OpenStereo/output/MultiDataset/WAVEStereo/wavestereo_mixdataset/20260623_wavestereo_filter/ckpt/checkpoint_epoch_2.pth"
    encoder = StereoEncoder(ckpt_path=ckpt).cuda()
    n = sum(p.numel() for p in encoder.parameters()) / 1e6
    print(f"StereoEncoder params: {n:.1f}M (frozen + trainable)")

    # dummy stereo pair (ImageNet-normalised)
    left = torch.randn(2, 3, 256, 512, device="cuda")
    right = torch.randn(2, 3, 256, 512, device="cuda")

    with torch.no_grad():
        tokens = encoder(left, right)
    print(f"output shape: {tuple(tokens.shape)}  (expected: (2, {encoder.N_stereo_tokens}, {encoder.hidden_dim}))")
