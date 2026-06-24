# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""
QwenZone Framework — memory-token VLA with block-wise attention (Phase 1).

Design (see tt.md): introduce history via memory tokens so the model can distinguish
identical-looking observations that require different future actions, and decouple
"action intent" (encoded by VLM) from "action execution" (action head).

Sequence per sample (T timesteps):
    S = [M_init, V_0, L, A_0, M_0, V_1, A_1, M_1, ..., V_{T-1}, A_{T-1}, M_{T-1}]
  - M_init : N_mem memory-init tokens (emoji 🌱) at the head
  - V_t    : vision tokens of step t (3 cameras)
  - L      : language tokens
  - A_t    : N_act action-intent tokens (emoji 🔍), no supervision (driven by action loss)
  - M_t    : N_mem memory tokens (emoji 🧠), the only cross-step information channel

Attention is block-wise (see block_attention.py): bidirectional within a timestep block,
controlled across timesteps (history flows only through Memory). Implemented as a custom
4D additive mask fed to Qwen3-VL (transformers 4.57.0 supports custom 4D masks natively).

Action head (ZoneMemoryActionHead): non-autoregressive, predicts an H=50 step action chunk
at each timestep from h(A_t) + V_t + state, where:
  - h(A_t): action-intent from VLM last hidden (high-level, slow VLM forward)
  - V_t: raw vision features from ViT→Perceiver (fine-grained, pre-LLM, high-frequency candidate)
  - state: proprioception.
Decoupling V_t (raw vision) from the VLM LLM layers is intentional: the action head is meant
to combine coarse intent with fine-grained, real-time visual observations.

Phase 1 scope: single-pass forward + full-history inference. (Pass2/RNN/async delay = Phase 2.)
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

IMAGE_TOKEN_INDEX = 151655  # <image> placeholder id (aligned with QWen3.py)

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.framework.VLM4A.block_attention import build_block_attention_mask
from starVLA.model.modules.action_model.ZoneMemory_ActionHeader import get_action_model
from starVLA.model.modules.action_model.StereoEncoder import StereoEncoder
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.training.trainer_utils.trainer_tools import resize_images


def gather_token_hidden(last_hidden, input_ids, token_id, expected_count):
    """Gather the hidden states at positions where input_ids == token_id, in position order.

    last_hidden: [B, L, H]; input_ids: [B, L].
    Returns [B, expected_count, H]. Raises if any sample has fewer matches than expected.
    """
    B, L, H = last_hidden.shape
    mask = input_ids == token_id  # [B, L]
    counts = mask.sum(dim=1)
    if int(counts.min()) < expected_count:
        raise RuntimeError(
            f"gather_token_hidden(token={token_id}): need {expected_count} per sample, "
            f"got min={int(counts.min())}"
        )
    idx = torch.arange(L, device=last_hidden.device).unsqueeze(0).expand(B, L)
    masked_pos = torch.where(mask, idx, torch.full_like(idx, L))  # non-matches -> L (large)
    topk_pos = masked_pos.topk(k=expected_count, dim=-1, largest=False).values  # [B, k]
    topk_pos = topk_pos.sort(dim=-1).values
    expanded = topk_pos.unsqueeze(-1).expand(-1, -1, H)
    return last_hidden.gather(1, expanded)


# ──────────────────────────────────────────────────────────────────────
#  Default Config
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenZoneDefaultConfig:
    name: str = "QwenZone"

    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct-MemoryAction",
            "attn_implementation": "sdpa",  # 4D additive mask requires sdpa (not flash)
        }
    )

    qwenzone: dict = field(
        default_factory=lambda: {
            "T_obs": 4,        # sequence time steps
            "N_act": 8,        # action-intent tokens per step
            "N_mem": 8,        # memory tokens per step (and for M_init)
            "max_history": 16, # inference: cap history length
        }
    )

    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "ZoneMemory",
            "action_dim": 14,
            "state_dim": 14,
            "action_horizon": 50,
            "action_hidden_dim": 2560,  # overwritten by VLM hidden_size at runtime
            "hidden_size": 512,
            "N_act": 8,
            "N_vis_tokens": 8,
            "N_state_tokens": 4,
            "nhead": 8,
            "num_transformer_layers": 4,
        }
    )


@FRAMEWORK_REGISTRY.register("QwenZone")
class Qwenvl_Zone(baseframework):
    """QwenZone memory-token VLA (Phase 1)."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenZoneDefaultConfig, config)

        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.processor = self.qwen_vl_interface.processor
        hidden_size = self.qwen_vl_interface.model.config.hidden_size
        self.config.framework.action_model.action_hidden_dim = hidden_size

        self.action_model = get_action_model(config=self.config)

        qz = self.config.framework.qwenzone
        self.T_obs = int(qz.T_obs)
        self.N_act = int(qz.N_act)
        self.N_mem = int(qz.N_mem)
        self.max_history = int(qz.get("max_history", 16))
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

        # consistency: sequence N_act must equal action head N_act
        assert int(self.config.framework.action_model.get("N_act", self.N_act)) == self.N_act, (
            "qwenzone.N_act must equal action_model.N_act"
        )

        tok = self.processor.tokenizer
        self.action_token_id = self._single_token_id(tok, "🔍")
        self.memory_token_id = self._single_token_id(tok, "🧠")
        self.memory_init_id = self._single_token_id(tok, "🌱")
        self.image_token_id = IMAGE_TOKEN_INDEX
        self.pad_token_id = tok.pad_token_id

        # --- frozen stereo encoder (optional) ---
        stereo_cfg = qz.get("stereo", {}) or {}
        if stereo_cfg.get("ckpt_path"):
            from starVLA.model.modules.action_model.StereoEncoder import StereoEncoder
            self.stereo_encoder = StereoEncoder(
                ckpt_path=stereo_cfg["ckpt_path"],
                update_iters=int(stereo_cfg.get("update_iters", 4)),
                pool_size=int(stereo_cfg.get("pool_size", 8)),
                hidden_dim=int(self.config.framework.action_model.get("hidden_size", 512)),
                N_stereo_tokens=int(stereo_cfg.get("N_stereo_tokens", 8)),
            )
        else:
            self.stereo_encoder = None

        self.l1_loss = nn.L1Loss()
        self._infer_history: Optional[List[dict]] = None

    @staticmethod
    def _single_token_id(tokenizer, emoji):
        ids = tokenizer(emoji, add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, f"emoji {emoji} must be a single token, got {ids}"
        return ids[0]

    # ── sequence construction ──────────────────────────────────────────
    def _build_sequence(self, example, images_by_cam):
        """Build input_ids + token_meta + image tensors for one sample.

        images_by_cam: [num_cam][T] PIL images (cam-major, time-minor).
        Returns dict with input_ids (list[int]), token_meta (list[(type,step)]),
        pixel_values, image_grid_thw.
        """
        tok = self.processor.tokenizer
        num_cam = len(images_by_cam)
        T = len(images_by_cam[0])

        # flatten images as [s0_c0, s0_c1, ..., s0_c{C-1}, s1_c0, ...] (time-major, cam-minor)
        flat_images = [images_by_cam[c][t] for t in range(T) for c in range(num_cam)]

        img_inputs = self.processor.image_processor(images=flat_images, return_tensors="pt")
        pixel_values = img_inputs["pixel_values"]
        image_grid_thw = img_inputs["image_grid_thw"]
        per_image_tokens = (image_grid_thw.prod(dim=-1) // 4).tolist()
        # assume all images share the same resolution (RoboTwin 224x224)
        assert len(set(per_image_tokens)) == 1, (
            f"QwenZone expects uniform image resolution, got token counts {per_image_tokens}"
        )
        self._per_image_tokens = per_image_tokens[0]

        L_ids = tok(example["lang"], add_special_tokens=False)["input_ids"]

        input_ids = []
        token_meta = []  # (type, step)
        # M_init at the head (step -1)
        input_ids += [self.memory_init_id] * self.N_mem
        token_meta += [("M_init", -1)] * self.N_mem

        img_idx = 0
        for t in range(T):
            for _c in range(num_cam):
                n = per_image_tokens[img_idx]
                input_ids += [self.image_token_id] * n
                token_meta += [("V", t)] * n
                img_idx += 1
            if t == 0:
                input_ids += L_ids
                token_meta += [("L", 0)] * len(L_ids)
            input_ids += [self.action_token_id] * self.N_act
            token_meta += [("A", t)] * self.N_act
            input_ids += [self.memory_token_id] * self.N_mem
            token_meta += [("M", t)] * self.N_mem

        assert img_idx == len(flat_images), "image token block count mismatch"

        return {
            "input_ids": input_ids,
            "token_meta": token_meta,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "num_cam": num_cam,
        }

    def _assemble_batch(self, seqs, device):
        """Left-pad input_ids to common length and stack a 4D block attention mask."""
        Lmax = max(len(s["input_ids"]) for s in seqs)
        neg = torch.finfo(torch.bfloat16).min

        ids_batch = []
        attn_list = []
        for s in seqs:
            Li = len(s["input_ids"])
            pad_len = Lmax - Li
            ids_padded = [self.pad_token_id] * pad_len + s["input_ids"]
            ids_batch.append(ids_padded)

            types = [m[0] for m in s["token_meta"]]
            steps = [m[1] for m in s["token_meta"]]
            m_i = build_block_attention_mask(types, steps, dtype=torch.bfloat16, device=device)  # [Li, Li]
            full = torch.full((Lmax, Lmax), neg, dtype=torch.bfloat16, device=device)
            full[pad_len:, pad_len:] = m_i  # real tokens block at bottom-right; pad rows/cols stay masked
            attn_list.append(full)

        input_ids = torch.tensor(ids_batch, dtype=torch.long, device=device)  # [B, Lmax]
        attn_4d = torch.stack(attn_list, dim=0).unsqueeze(1)  # [B, 1, Lmax, Lmax]

        pixel_values = torch.cat([s["pixel_values"] for s in seqs], dim=0).to(device)
        image_grid_thw = torch.cat([s["image_grid_thw"] for s in seqs], dim=0).to(device)
        return input_ids, attn_4d, pixel_values, image_grid_thw, Lmax

    def _gather_hA(self, last_hidden, input_ids, T):
        """Gather action-intent hidden states from VLM last layer, reshaped per timestep."""
        B = last_hidden.shape[0]
        H = last_hidden.shape[-1]
        h_A = gather_token_hidden(last_hidden, input_ids, self.action_token_id, T * self.N_act)
        return h_A.reshape(B, T, self.N_act, H)

    def _split_raw_vis_by_step(self, raw_vis, image_grid_thw, B, T, num_cam):
        """Split raw vision-encoder features (pre-LLM, ViT→Perceiver output) into [B,T,N_v,H].

        raw_vis: [total_tokens, H] from model.visual(), concatenated per-image across batch.
        image_grid_thw: [B*num_cam*T, 3] in flat_images order [s0_c0, s0_c1, ..., s0_c{C-1}, s1_c0, ...].
        """
        per_img_tokens = (image_grid_thw.prod(dim=-1) // 4).tolist()
        raw_per_img = list(raw_vis.split(per_img_tokens, dim=0))  # list of [n_i, H]

        idx = 0
        batch_feats = []
        for _b in range(B):
            step_feats = []
            for _t in range(T):
                step_parts = [raw_per_img[idx + c] for c in range(num_cam)]
                step_feats.append(torch.cat(step_parts, dim=0))
                idx += num_cam
            batch_feats.append(torch.stack(step_feats, dim=0))
        return torch.stack(batch_feats, dim=0)  # [B, T, N_v, H]

    def _extract_stereo(self, examples, device):
        """Run frozen stereo encoder on left/right image pairs per timestep.

        Each example is expected to carry ``stereo_left`` / ``stereo_right`` keys,
        each a list of T PIL images. Returns [B, T, N_stereo, hidden_dim].
        Falls back to a zero-filled tensor if no stereo images are provided
        (for backward compatibility with non-stereo configs).
        """
        if self.stereo_encoder is None:
            return None

        # determine T from the data (may differ between training and inference)
        B = len(examples)
        has_stereo = all("stereo_left" in ex and "stereo_right" in ex for ex in examples)
        if not has_stereo:
            return None

        # collect all stereo pairs across batch & timestep, then forward in one go
        all_left, all_right = [], []
        T = len(examples[0]["stereo_left"])
        for ex in examples:
            for t in range(T):
                all_left.append(ex["stereo_left"][t])
                all_right.append(ex["stereo_right"][t])

        # PIL → ImageNet-normalised tensor
        left_t = torch.stack([self._pil_to_imagenet(im) for im in all_left]).to(device)
        right_t = torch.stack([self._pil_to_imagenet(im) for im in all_right]).to(device)

        stereo_tokens = self.stereo_encoder(left_t, right_t)  # [B*T, N_stereo, hd]
        return stereo_tokens.reshape(B, T, *stereo_tokens.shape[1:])

    @staticmethod
    def _pil_to_imagenet(pil_img):
        """Convert a PIL image to an ImageNet-normalised tensor [3, H, W]."""
        import torchvision.transforms.functional as TF
        t = TF.to_tensor(pil_img)  # [0,1] range
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (t - mean) / std

    # ── training forward ───────────────────────────────────────────────
    def forward(self, examples: List[dict], **kwargs):
        device = next(self.qwen_vl_interface.parameters()).device
        T = self.T_obs
        H = self.action_horizon

        seqs = []
        for ex in examples:
            images_by_cam = [[to_pil_preserve(im) for im in cam_frames] for cam_frames in ex["image"]]
            seqs.append(self._build_sequence(ex, images_by_cam))

        input_ids, attn_4d, pixel_values, image_grid_thw, Lmax = self._assemble_batch(seqs, device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            # V_t: raw vision encoder output (ViT→Perceiver, pre-LLM) — fine-grained, high-frequency
            raw_vis = self.qwen_vl_interface.model.visual(pixel_values, grid_thw=image_grid_thw)[0]
            # VLM forward — encodes action intent h(A_t) via block-wise attention
            outputs = self.qwen_vl_interface(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                attention_mask=attn_4d,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]  # [B, Lmax, H]

        with torch.autocast("cuda", dtype=torch.float32):
            h_A = self._gather_hA(last_hidden, input_ids, T)   # [B,T,N_act,H] — action intent
            num_cam = seqs[0].get("num_cam", 3)
            h_V = self._split_raw_vis_by_step(raw_vis, image_grid_thw, len(examples), T, num_cam)

            # state: [B, T, state_dim]
            state = torch.tensor(
                np.stack([np.asarray(ex["state"], dtype=np.float32) for ex in examples]),
                device=device,
            )
            # action chunks: each ex action [T+H-1, dim] -> [T, H, dim]
            chunks = []
            for ex in examples:
                act = np.asarray(ex["action"], dtype=np.float32)
                chunk = np.stack([act[t : t + H] for t in range(T)], axis=0)  # [T, H, dim]
                chunks.append(chunk)
            actions_target = torch.tensor(np.stack(chunks), device=device)  # [B, T, H, dim]

            stereo_tokens = self._extract_stereo(examples, device)
            action_loss = self.action_model(h_A, h_V, state, actions_target, stereo=stereo_tokens)

        return {"action_loss": action_loss}

    # ── inference (full-history mode) ──────────────────────────────────
    def reset_history(self):
        self._infer_history = None

    @torch.inference_mode()
    def predict_action(self, examples, **kwargs):
        """Full-history inference. Each example carries the CURRENT step observation
        (image: [num_cam PIL], state: [state_dim]); we accumulate history internally and
        forward the whole sequence, returning the current step's H-step chunk.
        """
        if not isinstance(examples, list):
            examples = [examples]

        # Reset history when the task changes (new episode) so server-side history
        # doesn't leak across episodes.
        cur_lang = examples[0].get("lang", "") if examples else ""
        if self._infer_history is None or getattr(self, "_last_lang", None) != cur_lang:
            self._infer_history = [{"imgs": [], "states": []} for _ in examples]
            self._last_lang = cur_lang
        for i, ex in enumerate(examples):
            cur_imgs = [to_pil_preserve(im) for im in ex["image"]]  # [num_cam] current step
            self._infer_history[i]["imgs"].append(cur_imgs)
            self._infer_history[i]["states"].append(np.asarray(ex["state"], dtype=np.float32))
            if len(self._infer_history[i]["imgs"]) > self.max_history:
                self._infer_history[i]["imgs"] = self._infer_history[i]["imgs"][-self.max_history :]
                self._infer_history[i]["states"] = self._infer_history[i]["states"][-self.max_history :]

        device = next(self.qwen_vl_interface.parameters()).device
        T = len(self._infer_history[0]["imgs"])

        # build history-augmented examples (cam-major, time-minor) [num_cam][T]
        hist_examples = []
        hist_states = []
        for i, ex in enumerate(examples):
            h = self._infer_history[i]
            images_by_cam = [[h["imgs"][t][c] for t in range(T)] for c in range(len(h["imgs"][0]))]
            hist_examples.append({"image": images_by_cam, "lang": ex["lang"]})
            hist_states.append(np.stack(h["states"]))  # [T, state_dim]

        seqs = [self._build_sequence(he, he["image"]) for he in hist_examples]
        input_ids, attn_4d, pixel_values, image_grid_thw, Lmax = self._assemble_batch(seqs, device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            # V_t: raw vision encoder output (ViT→Perceiver, pre-LLM) — fine-grained, high-frequency
            raw_vis = self.qwen_vl_interface.model.visual(pixel_values, grid_thw=image_grid_thw)[0]
            # VLM forward — encodes action intent
            outputs = self.qwen_vl_interface(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                attention_mask=attn_4d,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            h_A = self._gather_hA(last_hidden, input_ids, T)   # [B,T,N_act,H] — action intent
            num_cam = seqs[0].get("num_cam", 3)
            h_V = self._split_raw_vis_by_step(raw_vis, image_grid_thw, len(hist_examples), T, num_cam)
            state = torch.tensor(np.stack(hist_states), device=device)  # [B, T, state_dim]
            stereo_tokens = self._extract_stereo(hist_examples, device)
            pred = self.action_model.predict_action(h_A, h_V, state, stereo=stereo_tokens)  # [B, T, H, dim]

        # return only the current (last) step's chunk
        cur = pred[:, -1, :, :].detach().cpu().numpy()  # [B, H, dim]
        return {"normalized_actions": cur}


if __name__ == "__main__":
    # tiny end-to-end test: forward + backward on random data (needs GPU + base ckpt)
    from omegaconf import OmegaConf

    T, H_ACTION = 2, 50
    ckpt = "/mnt/workspace/yama/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct-MemoryAction"
    cfg = OmegaConf.create(
        {
            "framework": {
                "name": "QwenZone",
                "qwenvl": {"base_vlm": ckpt, "attn_implementation": "sdpa"},
                "qwenzone": {"T_obs": T, "N_act": 4, "N_mem": 4, "max_history": 16},
                "action_model": {
                    "action_model_type": "ZoneMemory",
                    "action_dim": 14,
                    "state_dim": 14,
                    "action_horizon": H_ACTION,
                    "action_hidden_dim": 2560,
                    "hidden_size": 512,
                    "N_act": 4,
                    "N_vis_tokens": 8,
                    "N_state_tokens": 4,
                    "nhead": 8,
                    "num_transformer_layers": 4,
                },
            },
            "datasets": {"vla_data": {"obs_image_size": [224, 224]}},
        }
    )

    model = Qwenvl_Zone(cfg).cuda()
    print(f"[init] hidden_size used by action head: {model.config.framework.action_model.action_hidden_dim}")

    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    def make_sample(lang):
        return {
            "action": np.random.uniform(-1, 1, (T + H_ACTION - 1, 14)).astype(np.float16),
            "image": [[img] * T, [img] * T, [img] * T],  # [3 cam][T]
            "lang": lang,
            "state": np.random.uniform(-1, 1, (T, 14)).astype(np.float16),
        }

    batch = [make_sample("pick up the red block"), make_sample("open the drawer")]
    out = model(batch)
    print(f"[forward] action_loss = {out['action_loss'].item():.4f}")
    out["action_loss"].backward()
    print("[backward] OK")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    losses = []
    for step in range(5):
        opt.zero_grad()
        o = model(batch)
        o["action_loss"].backward()
        opt.step()
        losses.append(o["action_loss"].item())
    print(f"[train] losses over 5 steps: {[round(x, 4) for x in losses]}")
