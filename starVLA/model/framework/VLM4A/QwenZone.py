# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025].
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-Zone Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions.
Zone variant: temporally-aware multi-frame video input with configurable timestamp strategies.
Flow-matching header is copyright from GR00T N1.5,
"""

import re
import sys
from pathlib import Path

# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers.video_utils import VideoMetadata

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.Zone_ActionHeader import ZoneActionHeadTransformer, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenZone
#  - Documents every framework-level parameter with type + description
#  - YAML values override these defaults; extra YAML keys are preserved
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenZoneDefaultConfig:
    """QwenZone framework default parameters.

    All fields can be overridden by the corresponding key in the YAML
    ``framework:`` section.  Extra YAML keys not listed here are kept
    as-is (Config-as-API flexibility).
    """

    # --- Registry identifier ---
    name: str = "QwenZone"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            # Path to base VLM checkpoint (local or HF hub id)
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            # Attention implementation: "flash_attention_2" | "eager" | "sdpa"
            "attn_implementation": "flash_attention_2",
            # VLM hidden dimension (used for cross-attention alignment)
            "vl_hidden_dim": 2048,
        }
    )

    # === Video / multi-frame input ===
    video: dict = field(
        default_factory=lambda: {
            # Whether to treat multi-frame image lists as video (<video> token)
            # instead of independent <image> tokens. Requires Qwen3-VL / Qwen3.5-VL.
            "use_video_token": True,
            # Invert timestamps: count down from end instead of up from start
            "invert_timestamps": False,
            # Video fps (used when examples don't carry per-sample fps)
            "fps": 30.0,
            # Prevent processor from internally resampling pre-sampled frames
            "do_sample_frames": False,
        }
    )

    # === Action head (Flow-matching / DiT diffusion) ===
    action_model: dict = field(
        default_factory=lambda: {
            # DiT model size: "DiT-B" | "DiT-L" | "DiT-XL"
            "action_model_type": "DiT-B",
            # Hidden dim for action model (auto-aligned at runtime)
            "action_hidden_dim": 1024,
            "hidden_size": 1024,
            # Whether to add positional embeddings in the action head
            "add_pos_embed": True,
            "max_seq_len": 1024,
            # Dimensionality of each action vector (e.g., 7 for 6-DoF + gripper)
            "action_dim": 7,
            # State dimension (proprioception input)
            "state_dim": 7,
            # Canonical chunk length (number of action steps the head predicts).
            # Legacy YAMLs may use future_action_window_size = action_horizon - 1;
            # apply_config_compat normalises both directions.
            "action_horizon": 8,
            # Repeat factor for flow-matching loss (more noise samples per batch)
            "repeated_diffusion_steps": 8,
            # Beta distribution params for noise schedule
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            # Inference denoising steps
            "num_inference_timesteps": 4,
            # Number of vision tokens fed to action head
            "num_target_vision_tokens": 32,
            # === DiT Transformer sub-config ===
            "diffusion_model_cfg": {
                # Cross-attention dim (aligned to VLM hidden_size at runtime)
                "cross_attention_dim": 2048,
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "num_layers": 16,
                "output_dim": 1024,
                "positional_embeddings": None,
            },
        }
    )


@FRAMEWORK_REGISTRY.register("QwenZone")
class Qwen_Zone(baseframework):
    """
    Multimodal vision-language-action model (Zone variant — temporally-aware).

    Components:
      - Qwen2.5-VL / Qwen3-VL backbone for fused language/vision token embeddings
      - Flow-matching (DiT) diffusion head for continuous action sequence modeling

    Focus: Predict future continuous actions conditioned on multi-frame video + instruction.
    Supports configurable timestamp strategies via monkey-patched processor.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        # Merge framework defaults with YAML config (YAML wins on conflicts)
        self.config = merge_framework_config(QwenZoneDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        self.action_model: ZoneActionHeadTransformer = get_action_model(config=self.config)

        # `action_horizon` is the single source of truth for chunk length.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

        # --- hook up video-aware processor patches ---
        self._video_cfg = self.config.framework.get("video", {})
        self._use_video = self._video_cfg.get("use_video_token", False)
        self._invert_ts = self._video_cfg.get("invert_timestamps", False)
        self._default_fps = self._video_cfg.get("fps", 30.0)
        self._do_sample = self._video_cfg.get("do_sample_frames", False)

        if self._use_video:
            self._install_processor_patches()

    def _install_processor_patches(self):
        """Monkey-patch the processor for accurate video timestamps and grid_thw."""
        proc = self.qwen_vl_interface.processor
        if not hasattr(proc, '_calculate_timestamps'):
            return
        _orig_calc_ts = proc._calculate_timestamps
        invert = self._invert_ts

        def _exact_ts(indices, video_fps, merge_size=2):
            real_indices = getattr(proc, '_sample_indices', None)
            real_fps = getattr(proc, '_real_fps', None)
            if real_indices is not None and real_fps is not None:
                # real_indices are offsets from current frame (e.g. [-21, -13, ..., 0]).
                # Convert to positive "seconds ago":  0.0 = now, 0.42 = 0.42s ago.
                frame_ts = [-real_indices[idx] / real_fps for idx in indices]
            else:
                frame_ts = [idx / video_fps for idx in indices]
            return [(frame_ts[i] + frame_ts[i + merge_size - 1]) / 2
                    for i in range(0, len(frame_ts), merge_size)]

        proc._calculate_timestamps = _exact_ts

    # ── helper: detect whether input is multi-frame video ──────────────

    @staticmethod
    def _is_video_input(imgs):
        """imgs is one sample's image list.  True when first element is itself a list of frames."""
        return len(imgs) > 0 and isinstance(imgs[0], (list, tuple))

    # ── video-aware input builder (replaces build_qwenvl_inputs for video) ──

    def _build_zone_inputs(self, images, instructions, solutions=None,
                           fps_list=None, sample_indices_list=None):
        """
        Build processor inputs, routing to <video> tokens when input is multi-frame.

        Args:
            images:  List[List[PIL.Image]] — B samples, each a flat image list (image mode)
                     or List[List[List[PIL.Image]]] — B samples, each a camera×frames list (video mode)
            instructions:  List[str]
            solutions:  Optional[List[str]]
            fps_list:  Optional[List[float]] — per-sample fps for video mode
            sample_indices_list:  Optional[List[List[int]]] — per-sample frame indices (for exact timestamps)

        Returns:
            BatchFeature dict on the correct device.
        """
        if any(self._is_video_input(imgs) for imgs in images):
            return self._build_video_inputs(
                images, instructions, solutions, fps_list, sample_indices_list)
        # fallback to original image-only path
        return self.qwen_vl_interface.build_qwenvl_inputs(
            images=images, instructions=instructions, solutions=solutions)

    def _build_video_inputs(self, images, instructions, solutions=None,
                            fps_list=None, sample_indices_list=None):
        """Build inputs with <video> tokens — one <video> block per camera."""
        proc = self.qwen_vl_interface.processor
        has_solutions = solutions is not None

        default_fps = self._default_fps
        messages = []
        for b in range(len(images)):
            imgs = images[b]
            instruction = instructions[b]
            per_cam_frames = imgs if self._is_video_input(imgs) else [imgs]
            n_frames = len(per_cam_frames[0]) if per_cam_frames else 0
            fps = fps_list[b] if fps_list else default_fps

            # Store for monkey-patched _calculate_timestamps
            if sample_indices_list and b < len(sample_indices_list):
                proc._sample_indices = sample_indices_list[b]
            else:
                proc._sample_indices = list(range(n_frames))
            proc._real_fps = fps

            duration = n_frames / fps if fps > 0 else 0
            effective_fps = n_frames / duration if duration > 0 else fps

            content = []
            for cam_frames in per_cam_frames:
                nf = len(cam_frames) if isinstance(cam_frames, (list, tuple)) else 1
                meta = VideoMetadata(
                    total_num_frames=nf,
                    fps=effective_fps,
                    frames_indices=list(range(nf)),
                )
                content.append({
                    "type": "video",
                    "video": cam_frames,
                    "fps": effective_fps,
                })
                # Attach metadata to last video entry for processor
                content[-1]["video_metadata"] = meta

            if "CoT_prompt" in self.config.datasets.vla_data:
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction
            content.append({"type": "text", "text": prompt})

            msg = [{"role": "user", "content": content}]
            if has_solutions:
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solutions[b]}]})
            messages.append(msg)

        # Build with processor
        video_metadata_flat = []
        for msg in messages:
            for item in msg[0]["content"]:
                if isinstance(item, dict) and item.get("type") == "video":
                    vm = item.pop("video_metadata", None)
                    if vm:
                        video_metadata_flat.append(vm)

        batch_inputs = proc.apply_chat_template(
            messages, tokenize=True, padding=True,
            add_generation_prompt=True, return_dict=True, return_tensors="pt",
            do_sample_frames=self._do_sample,
            video_metadata=video_metadata_flat if video_metadata_flat else None,
        )

        # Fix: split video_grid_thw so each temporal chunk has its own row
        if "video_grid_thw" in batch_inputs:
            vg = batch_inputs["video_grid_thw"]
            new_rows = []
            for row in vg:
                T = int(row[0].item())
                for _ in range(T):
                    new_rows.append([1, int(row[1].item()), int(row[2].item())])
            batch_inputs["video_grid_thw"] = torch.tensor(
                new_rows, dtype=vg.dtype, device=vg.device)

        # Label masking (same logic as build_qwenvl_inputs)
        if has_solutions:
            action_token_min = 248077
            action_token_max = 248077 + 2047
            labels = batch_inputs["input_ids"].clone()
            for i in range(labels.size(0)):
                seq = labels[i]
                mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                nonzero = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero.numel() > 0:
                    seq[:nonzero[0].item()] = -100
                else:
                    seq[:] = -100
            labels[labels == proc.tokenizer.pad_token_id] = -100
            batch_inputs["labels"] = labels

        return batch_inputs.to(self.qwen_vl_interface.model.device)

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """ """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]
        fps_list = [example.get("fps", self._default_fps) for example in examples]
        sample_indices_list = [example.get("sample_indices", None) for example in examples]

        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        # Step 1: QWenVL input format (auto-detects image vs video)
        qwen_inputs = self._build_zone_inputs(
            images=batch_images, instructions=instructions,
            fps_list=fps_list, sample_indices_list=sample_indices_list)
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        # Step 4: Action Head (MLP — no diffusion)
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )
            actions_target = actions[:, -self.action_horizon :, :]  # (B, action_horizon, action_dim)

            state_tensor = None
            if state is not None:
                state_tensor = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)  # [B, 1, D]
                # Tile current state to pseudo-history (until dataloader provides real history)
                state_tensor = state_tensor.repeat(1, self.action_model.state_history_len, 1)  # [B, 50, D]

            action_loss = self.action_model(
                last_hidden, actions_target, state_tensor,
                encoder_attention_mask=backbone_attention_mask,
            )

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        """
        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        fps_list = [example.get("fps", self._default_fps) for example in examples]
        sample_indices_list = [example.get("sample_indices", None) for example in examples]

        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # Step 1: QWenVL input format (auto-detects image vs video)
        qwen_inputs = self._build_zone_inputs(
            images=batch_images, instructions=instructions,
            fps_list=fps_list, sample_indices_list=sample_indices_list)
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        state = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )
        if state is not None and state.dim() == 3 and state.shape[1] == 1:
            state = state.repeat(1, self.action_model.state_history_len, 1)  # [B, 1, D] → [B, 50, D]

        # Step 4: Action Head (transformer, one-shot)
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                last_hidden, state=state, encoder_attention_mask=backbone_attention_mask
            )  # (B, action_horizon, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    model: Qwen_Zone = Qwen_Zone(cfg)
    #print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
        "lang": "This is a fake instruction for testing.",
    }
    sample2 = sample.copy()
    sample2["lang"] = "Another fake instruction for testing."

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")

    predict_output = model.predict_action(examples=[sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"Unnormalized Action: {normalized_actions}")

    print("Finished")