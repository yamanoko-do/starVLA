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
at each timestep from h(A_δ) + V_t + state, where:
  - h(A_δ): action-intent from VLM last hidden at a sampled delay step δ ~ U{t-d_max, t}
    (async training: makes A_t a "delayable intent" to match slow-VLM inference).
  - V_t: raw vision features from ViT→Perceiver (fine-grained, pre-LLM, high-frequency candidate)
  - state: proprioception.
Decoupling V_t (raw vision) from the VLM LLM layers is intentional: the action head is meant
to combine coarse intent with fine-grained, real-time visual observations.

Phase 2 (full tt.md design): dual-Pass training + RNN inference + async delay.
  - Pass 1: standard parallel forward over the full T-step sequence; extracts m̄_t = h_L(M_t).
  - Pass 2: B*T independent 1-step RNN forwards  S_t^rnn = [m_prev, V_t, L, A_t, M_t]  where
    m_prev = m̄_{t-1} (detached, injected via a forward_pre_hook on the LLM at the m_prev/🌱
    slot — after image features are merged and mrope position_ids are computed). Reuses
    Pass1's h_V/state/stereo. Joint loss L = L_parallel + lambda_rnn * L_rnn.
  - Inference mode-1 (full history) and mode-2 (RNN, identical sequence structure to Pass2 →
    no train/infer shift). infer_mode selects between them.
"""

from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import List, Optional
import logging

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
            "train_seq_len": 4,        # sequence time steps
            "N_act": 16,       # cmd (action-intent) tokens per step
            "N_mem": 4,        # memory tokens per step (and for M_init)
            "max_history": 16, # inference: cap history length
            # ── Phase 2 ──
            "lambda_rnn": 0.1,    # Pass2 (RNN) loss weight; raise to ~0.3 once stable
            "d_max": 5,           # async action delay: δ ~ U{t-d_max, ..., t}
            "infer_mode": "full", # "full" (mode-1, full history) | "rnn" (mode-2, single-step + memory injection)
            "vlm_stride": 0,      # 0 = run the LLM every step (synchronous); >0 = refresh the LLM every K steps (K-1 ≤ d_max)
            "stereo": {
                "update_iters": 4,
                "N_stereo_tokens": 64,
                "input_size": [256, 256],
            },
        }
    )

    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "ZoneMemory",
            "action_dim": 14,
            "state_dim": 28,
            "action_horizon": 50,
            "action_hidden_dim": 2560,  # overwritten by VLM hidden_size at runtime
            "hidden_size": 512,
            "N_state_tokens": 4,
            "nhead": 8,
            "num_transformer_layers": 4,
            # N_act / N_stereo / stereo_dim read from qwenzone.* (single source of truth)
            # vis is NOT pooled (raw N_v tokens)
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
        self.train_seq_len = int(qz.train_seq_len)
        self.N_act = int(qz.N_act)
        self.N_mem = int(qz.N_mem)
        self.max_history = int(qz.get("max_history", 16))
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        # ── Phase 2 knobs ──
        self.lambda_rnn = float(qz.get("lambda_rnn", 0.0))  # 0 → single-Pass (Phase 1) behavior
        self.d_max = int(qz.get("d_max", 0))                # 0 → no async delay (δ == t)
        self.infer_mode = str(qz.get("infer_mode", "full")) # "full" | "rnn" | "async"
        self.vlm_stride = int(qz.get("vlm_stride", 0))      # 0 = sync (LLM every step); >0 = refresh LLM every K steps
        # vit token遮蔽参数
        self.vis_mask_probability = float(self.config.framework.action_model.get("vis_mask_probability", 0.0))
        self.vis_mask_ratio = float(self.config.framework.action_model.get("vis_mask_ratio", 0.0))
        self.cmd_mask_probability = float(self.config.framework.action_model.get("cmd_mask_probability", 0.0))
        self.cmd_mask_ratio = float(self.config.framework.action_model.get("cmd_mask_ratio", 0.0))

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
                hidden_dim=int(self.config.framework.action_model.get("hidden_size", 512)),
                N_stereo_tokens=int(stereo_cfg.get("N_stereo_tokens", 64)),
                input_size=tuple(stereo_cfg.get("input_size", [256, 256])),
            )
        else:
            self.stereo_encoder = None

        self.l1_loss = nn.L1Loss()
        self._infer_history: Optional[List[dict]] = None
        # RNN inference state: per-example running memory hidden state m_state.
        self._rnn_state: Optional[List] = None
        self._last_lang: Optional[str] = None
        # Async stride state (shared by full & rnn when vlm_stride > 0):
        # cached action intent + step counter controlling how often the LLM refreshes.
        self._cached_hA: Optional[torch.Tensor] = None  # [B, N_act, H]
        self._vlm_step: int = 0

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

    def _gather_m_bar(self, last_hidden, input_ids, T):
        """Gather per-step memory hidden states m̄_t = h_L(M_t) from a Pass1 forward.
        Returns [B, T, N_mem, H]. Caller decides on .detach()."""
        B = last_hidden.shape[0]
        H = last_hidden.shape[-1]
        m_bar = gather_token_hidden(last_hidden, input_ids, self.memory_token_id, T * self.N_mem)
        return m_bar.reshape(B, T, self.N_mem, H)

    def _lang_ids(self, ex):
        """Tokenize the task language (shared across timesteps)."""
        return self.processor.tokenizer(ex["lang"], add_special_tokens=False)["input_ids"]

    def _mem_init_embed(self):
        """Learnable M_init (🌱) embedding [H], used as the RNN initial state m̄_{-1}.
        Used identically at Pass2 t=0 and inference step 0 → zero train/infer shift
        (see tt.md "m̄_{-1} = M_init")."""
        device = self.qwen_vl_interface.model.device
        with torch.no_grad():
            return self.qwen_vl_interface.model.get_input_embeddings()(
                torch.tensor([self.memory_init_id], device=device)
            )[0]  # [H]

    def _async_delta_hA(self, h_A, d_max):
        """Async action delay: for each step t, use h(A_{δ_t}) to predict chunk_t,
        δ_t ~ U{max(0, t-d_max), ..., t}  (tt.md 266-278). h_A:[B,T,N_act,H] -> [B,T,N_act,H].
        d_max<=0 -> identity (δ_t = t, i.e. Phase-1 behavior)."""
        if d_max <= 0:
            return h_A
        B, T = h_A.shape[:2]
        device = h_A.device
        deltas = torch.empty(T, B, dtype=torch.long, device=device)
        for t in range(T):
            low = max(0, t - d_max)
            deltas[t] = t if t == low else torch.randint(low, t + 1, (B,), device=device)
        idx = deltas.t().unsqueeze(-1).unsqueeze(-1).expand(B, T, self.N_act, h_A.shape[-1])  # [B,T,N_act,H]
        return h_A.gather(1, idx)

    # ── Phase 2: hidden-state injection (RNN path) ─────────────────────
    def _rnn_inject_hook(self):
        """forward_pre_hook for model.model.language_model: scatter the detached m_prev
        hidden state into the 🌱 (M_init) positions of inputs_embeds. Images are already
        merged and mrope position_ids already computed at this hook (verified by probe).
        Uses torch.where (differentiable, no in-place autograd hazard)."""
        def hook(module, args, kwargs):
            ie = kwargs.get("inputs_embeds")
            if ie is None and args and args[0] is not None:
                ie = args[0]
            H = ie.shape[-1]
            inject_full = torch.zeros_like(ie)
            inject_full[self._rnn_inject_mask2d] = self._rnn_inject_values.reshape(-1, H)
            ie2 = torch.where(self._rnn_inject_mask2d.unsqueeze(-1), inject_full, ie)
            if "inputs_embeds" in kwargs:
                kwargs["inputs_embeds"] = ie2
            else:
                args = (ie2,) + tuple(args[1:])
            return args, kwargs
        return hook

    @contextmanager
    def _injection_ctx(self):
        """Register the injection hook for the duration of one RNN forward, then remove it."""
        lm = self.qwen_vl_interface.model.model.language_model
        handle = lm.register_forward_pre_hook(self._rnn_inject_hook(), with_kwargs=True)
        try:
            yield
        finally:
            handle.remove()

    def _build_rnn_sequence(self, images_step, lang_ids):
        """Build one 1-step RNN sequence  S_t^rnn = [M_init(🌱), V_t, L, A_t(🔍), M_t(🧠)].
        images_step: [num_cam] PIL for one timestep; lang_ids: list[int] (shared language).
        The m_prev slot is labelled ("M_init",-1) so the standard block mask (T=1 form)
        yields exactly the tt.md RNN visibility (V/A/M read m_prev via prev_block)."""
        num_cam = len(images_step)
        img_inputs = self.processor.image_processor(images=images_step, return_tensors="pt")
        pixel_values = img_inputs["pixel_values"]
        image_grid_thw = img_inputs["image_grid_thw"]
        per_image_tokens = (image_grid_thw.prod(dim=-1) // 4).tolist()
        assert len(set(per_image_tokens)) == 1, "RNN seq expects uniform image resolution"

        input_ids = []
        token_meta = []
        input_ids += [self.memory_init_id] * self.N_mem                 # m_prev slot (placeholder, overwritten by hook)
        token_meta += [("M_init", -1)] * self.N_mem
        img_idx = 0
        for _c in range(num_cam):
            n = per_image_tokens[img_idx]
            input_ids += [self.image_token_id] * n
            token_meta += [("V", 0)] * n
            img_idx += 1
        input_ids += list(lang_ids)
        token_meta += [("L", 0)] * len(lang_ids)
        input_ids += [self.action_token_id] * self.N_act
        token_meta += [("A", 0)] * self.N_act
        input_ids += [self.memory_token_id] * self.N_mem
        token_meta += [("M", 0)] * self.N_mem
        assert img_idx == num_cam, "RNN image token block count mismatch"
        return {
            "input_ids": input_ids,
            "token_meta": token_meta,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "num_cam": num_cam,
        }

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
        has_stereo = all(ex.get("stereo_left") and ex.get("stereo_right") for ex in examples)
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
        # OPTIMIZATION: use torchvision for faster PIL→tensor conversion (14.58ms vs 20.64ms)
        import torchvision.transforms.functional as TF
        all_left_tf = []
        all_right_tf = []
        for ex in examples:
            for t in range(T):
                all_left_tf.append(ex["stereo_left"][t])
                all_right_tf.append(ex["stereo_right"][t])
        # Batch convert PIL→tensor on CPU, then single GPU transfer
        left_t = torch.stack([TF.to_tensor(im).float().div_(255.0) for im in all_left_tf]).to(device)
        right_t = torch.stack([TF.to_tensor(im).float().div_(255.0) for im in all_right_tf]).to(device)

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
        T = self.train_seq_len
        H = self.action_horizon
        B = len(examples)

        seqs = []
        lang_ids_per_ex = []
        for ex in examples:
            images_by_cam = [[to_pil_preserve(im) for im in cam_frames] for cam_frames in ex["image"]]
            seqs.append(self._build_sequence(ex, images_by_cam))
            lang_ids_per_ex.append(self._lang_ids(ex))

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
            h_V = self._split_raw_vis_by_step(raw_vis, image_grid_thw, B, T, num_cam)  # [B,T,N_v,H]

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

            # Pass 1: parallel action loss with async δ (δ applied to action intent only)
            h_A_delta = self._async_delta_hA(h_A, self.d_max)
            # 随机遮蔽部分视觉token，减轻对视觉的过度依赖
            L_parallel = self.action_model(h_A_delta, h_V, state, actions_target, stereo=stereo_tokens,
                                    vis_mask_probability=self.vis_mask_probability, vis_mask_ratio=self.vis_mask_ratio,
                                    cmd_mask_probability=self.cmd_mask_probability, cmd_mask_ratio=self.cmd_mask_ratio)

            # Pass 2: RNN serial forward (only when lambda_rnn > 0)
            if self.lambda_rnn > 0:
                m_bar = self._gather_m_bar(last_hidden, input_ids, T).detach()  # [B,T,N_mem,H]
                mem_init = self._mem_init_embed().detach()                       # [H]
                L_rnn = self._pass2_rnn_loss(
                    examples, lang_ids_per_ex, m_bar, mem_init,
                    h_V, state, actions_target, stereo_tokens, device,
                )
            else:
                L_rnn = torch.zeros((), device=device)

        total = L_parallel + self.lambda_rnn * L_rnn
        return {
            "action_loss": total,
            "parallel_loss": L_parallel.detach(),
            "rnn_loss": L_rnn.detach() if torch.is_tensor(L_rnn) else torch.tensor(float(L_rnn)),
        }

    def _pass2_rnn_loss(self, examples, lang_ids_per_ex, m_bar, mem_init,
                        h_V, state, actions_target, stereo_tokens, device):
        """Pass 2: B*T independent 1-step RNN forwards with injected m_prev = m̄_{t-1}.
        Reuses Pass1's h_V/state/stereo/target (identical per timestep). Returns scalar.

        The m_prev slot (🌱) embedding is overwritten by the detached m_prev hidden via
        the injection hook (no grad flows back into Pass1 — tt.md 194).
        """
        B = len(examples)
        T = self.train_seq_len

        # build B*T RNN sequences (one per timestep; identical length → no padding)
        rnn_seqs = []
        for b, ex in enumerate(examples):
            images_by_cam = [[to_pil_preserve(im) for im in cam_frames] for cam_frames in ex["image"]]
            for t in range(T):
                images_step = [images_by_cam[c][t] for c in range(len(images_by_cam))]
                rnn_seqs.append(self._build_rnn_sequence(images_step, lang_ids_per_ex[b]))
        input_ids_rnn, attn_4d_rnn, pixel_values_rnn, image_grid_thw_rnn, _ = self._assemble_batch(
            rnn_seqs, device
        )

        # inject values [B*T, N_mem, H]: t=0 -> M_init embed (all N_mem slots); t>0 -> m̄_{t-1}
        Hd = mem_init.shape[-1]
        inject = torch.empty(B, T, self.N_mem, Hd, device=device, dtype=m_bar.dtype)
        inject[:, 0] = mem_init                              # broadcast [H] -> [B, N_mem, H]
        if T > 1:
            inject[:, 1:] = m_bar[:, :-1]
        self._rnn_inject_values = inject.reshape(B * T, self.N_mem, Hd).detach()
        self._rnn_inject_mask2d = input_ids_rnn == self.memory_init_id   # [B*T, Lmax]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            with self._injection_ctx():
                outputs = self.qwen_vl_interface(
                    input_ids=input_ids_rnn,
                    pixel_values=pixel_values_rnn,
                    image_grid_thw=image_grid_thw_rnn,
                    attention_mask=attn_4d_rnn,
                    output_hidden_states=True,
                    return_dict=True,
                )
            last_hidden_rnn = outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            h_A_rnn = self._gather_hA(last_hidden_rnn, input_ids_rnn, 1).reshape(B, T, self.N_act, -1)
            h_A_rnn_delta = self._async_delta_hA(h_A_rnn, self.d_max)
            # Pass 2 也使用随机遮蔽visual和cmd token
            return self.action_model(h_A_rnn_delta, h_V, state, actions_target, stereo=stereo_tokens,
                                    vis_mask_probability=self.vis_mask_probability, vis_mask_ratio=self.vis_mask_ratio,
                                    cmd_mask_probability=self.cmd_mask_probability, cmd_mask_ratio=self.cmd_mask_ratio)


    # ── inference ──────────────────────────────────────────────────────
    def reset_history(self):
        """Reset internal state (full-history buffer, RNN m_state, async stride cache). Call on episode start."""
        self._infer_history = None
        self._rnn_state = None
        self._last_lang = None
        self._cached_hA = None
        self._vlm_step = 0

    def _should_refresh_vlm(self) -> bool:
        """Advance the step counter and decide whether this step must run the (slow) LLM.

        - vlm_stride == 0: synchronous → always refresh (run the LLM every step).
        - vlm_stride  > 0: async → refresh on the first step and every K-th step afterwards;
          in between the action head reuses the cached (possibly stale) action intent with a
          fresh observation. (``% 0`` is guarded by the ``<= 0`` short-circuit.)
        """
        self._vlm_step += 1
        if self._cached_hA is None:
            return True
        if self.vlm_stride <= 0:
            return True
        return (self._vlm_step - 1) % self.vlm_stride == 0

    @torch.inference_mode()
    def predict_action(self, examples, **kwargs):
        """Inference entry. Two modes, each honoring the vlm_stride async knob:
          - "full": full-history mode (mode-1), accumulates history, forwards the whole sequence.
          - "rnn":  RNN mode (mode-2), one short forward per step with running m_state.
          vlm_stride: 0 = run the LLM every step (synchronous); >0 = refresh the LLM every K
          steps and reuse the cached action intent in between (both modes support this).
        """
        if self.infer_mode == "rnn":
            return self.predict_action_rnn(examples, **kwargs)
        return self.predict_action_full(examples, **kwargs)

    @torch.inference_mode()
    def predict_action_full(self, examples, **kwargs):
        """Full-history inference (mode-1). Each example carries the CURRENT step observation
        (image: [num_cam PIL], state: [state_dim]); we accumulate history internally and
        forward the whole sequence, returning the current step's H-step chunk.

        With vlm_stride > 0 the (slow) LLM refreshes only every K steps over the full history;
        in between, only the current step's ViT runs and the action head reuses the cached
        (stale) action intent with a fresh observation (the δ-trained staleness).
        vlm_stride == 0 runs the LLM every step.
        """
        if not isinstance(examples, list):
            examples = [examples]

        # Reset history when the task changes (new episode) so server-side history
        # doesn't leak across episodes.
        cur_lang = examples[0].get("lang", "") if examples else ""
        if self._infer_history is None or getattr(self, "_last_lang", None) != cur_lang:
            self._infer_history = [
                {"imgs": [], "states": [], "stereo_left": [], "stereo_right": []} for _ in examples
            ]
            self._last_lang = cur_lang
            self._cached_hA = None
            self._vlm_step = 0
        # Always accumulate the current observation so the next refresh can absorb it.
        for i, ex in enumerate(examples):
            cur_imgs = [to_pil_preserve(im) for im in ex["image"]]  # [num_cam] current step
            self._infer_history[i]["imgs"].append(cur_imgs)
            self._infer_history[i]["states"].append(np.asarray(ex["state"], dtype=np.float32))
            # stereo: accumulate per-step left/right pair (must match training _extract_stereo)
            if self.stereo_encoder is not None and ex.get("stereo_left") and ex.get("stereo_right"):
                self._infer_history[i]["stereo_left"].append(to_pil_preserve(ex["stereo_left"][0]))
                self._infer_history[i]["stereo_right"].append(to_pil_preserve(ex["stereo_right"][0]))
            if len(self._infer_history[i]["imgs"]) > self.max_history:
                for k in ("imgs", "states", "stereo_left", "stereo_right"):
                    if self._infer_history[i][k]:
                        self._infer_history[i][k] = self._infer_history[i][k][-self.max_history :]

        device = next(self.qwen_vl_interface.parameters()).device
        B = len(examples)
        refresh = self._should_refresh_vlm()

        if refresh:
            # ---- refresh: forward the full accumulated history through ViT + LLM ----
            T = len(self._infer_history[0]["imgs"])
            logging.info(f"[FULL] refresh T={T}, max_history={self.max_history}, vlm_stride={self.vlm_stride}")
            # build history-augmented examples (cam-major, time-minor) [num_cam][T]
            hist_examples = []
            hist_states = []
            for i, ex in enumerate(examples):
                h = self._infer_history[i]
                images_by_cam = [[h["imgs"][t][c] for t in range(T)] for c in range(len(h["imgs"][0]))]
                hist_examples.append({"image": images_by_cam, "lang": ex["lang"],
                                      "stereo_left": h["stereo_left"], "stereo_right": h["stereo_right"]})
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
                h_V = self._split_raw_vis_by_step(raw_vis, image_grid_thw, B, T, num_cam)  # [B,T,N_v,H]
                state = torch.tensor(np.stack(hist_states), device=device)  # [B, T, state_dim]
                stereo_tokens = self._extract_stereo(hist_examples, device)  # [B,T,...] | None
                # cache the LAST step's intent for reuse on non-refresh steps
                self._cached_hA = h_A[:, -1].detach()  # [B, N_act, H]
                stereo_last = stereo_tokens[:, -1:] if stereo_tokens is not None else None
                # current-step inputs = last step of the full history
                pred = self.action_model.predict_action(
                    h_A[:, -1:], h_V[:, -1:], state[:, -1:], stereo=stereo_last)  # [B,1,H,dim]
        else:
            # ---- non-refresh: only the current step's ViT + action head (cached intent) ----
            logging.info(f"[FULL] skip-LLM (cached intent), vlm_stride={self.vlm_stride}")
            seqs = []
            for ex in examples:
                images_step = [to_pil_preserve(im) for im in ex["image"]]
                seqs.append(self._build_rnn_sequence(images_step, self._lang_ids(ex)))  # single-step image builder
            input_ids, attn_4d, pixel_values, image_grid_thw, _ = self._assemble_batch(seqs, device)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                raw_vis = self.qwen_vl_interface.model.visual(pixel_values, grid_thw=image_grid_thw)[0]

            with torch.autocast("cuda", dtype=torch.float32):
                num_cam = seqs[0]["num_cam"]
                h_V = self._split_raw_vis_by_step(raw_vis, image_grid_thw, B, 1, num_cam)  # [B,1,N_v,H]
                state = torch.tensor(
                    np.stack([np.asarray(ex["state"], dtype=np.float32) for ex in examples]),
                    device=device,
                )[:, None, :]  # [B,1,state_dim]
                stereo_tokens = self._extract_stereo(examples, device)  # [B,1,...] | None
                h_A = self._cached_hA[:, None, :, :]  # [B,1,N_act,H] (cached, possibly stale)
                pred = self.action_model.predict_action(h_A, h_V, state, stereo=stereo_tokens)  # [B,1,H,dim]

        cur = pred[:, 0, :, :].detach().cpu().numpy()  # [B, H, dim]
        logging.info(f"[FULL] action[0,0,:5]={cur[0,0,:5]}")
        return {"normalized_actions": cur}

    @torch.inference_mode()
    def predict_action_rnn(self, examples, **kwargs):
        """RNN inference (mode-2). One short forward per step; the running m_state
        (last-layer hidden of the previous M_t) carries memory. Zero distribution shift
        vs Pass2 (identical sequence structure).

        With vlm_stride > 0 the (slow) LLM refreshes only every K steps; in between the action
        head runs every step with a FRESH observation (ViT/stereo/state) but the cached (possibly
        stale) action intent — the δ-trained staleness (max staleness = vlm_stride-1 ≤ d_max).
        vlm_stride == 0 runs the LLM every step. Returns the current step's H-step chunk.
        """
        if not isinstance(examples, list):
            examples = [examples]
        device = next(self.qwen_vl_interface.parameters()).device
        B = len(examples)

        # reset on episode change (new language)
        cur_lang = examples[0].get("lang", "") if examples else ""
        if self._rnn_state is None or self._last_lang != cur_lang:
            mem_init = self._mem_init_embed().detach()                       # [H]
            self._rnn_state = [mem_init.view(1, -1).expand(self.N_mem, -1) for _ in examples]
            self._last_lang = cur_lang
            self._cached_hA = None
            self._vlm_step = 0

        refresh = self._should_refresh_vlm()

        # build the 1-step RNN sequence for the CURRENT observation (always needed: fresh ViT)
        rnn_seqs = []
        for ex in examples:
            images_step = [to_pil_preserve(im) for im in ex["image"]]
            rnn_seqs.append(self._build_rnn_sequence(images_step, self._lang_ids(ex)))
        logging.info(f"[RNN] T=1, B={B}, refresh={refresh}, vlm_stride={self.vlm_stride}")
        input_ids, attn_4d, pixel_values, image_grid_thw, _ = self._assemble_batch(rnn_seqs, device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            # fresh raw vision features every step (ViT — cheap relative to the LLM)
            raw_vis = self.qwen_vl_interface.model.visual(pixel_values, grid_thw=image_grid_thw)[0]
            # LLM refresh (the expensive part) only when due
            if refresh:
                # inject the running m_state into the 🌱 (m_prev) slot
                self._rnn_inject_values = torch.stack(self._rnn_state, dim=0).to(device)  # [B, N_mem, H]
                self._rnn_inject_mask2d = input_ids == self.memory_init_id
                with self._injection_ctx():
                    outputs = self.qwen_vl_interface(
                        input_ids=input_ids, pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw, attention_mask=attn_4d,
                        output_hidden_states=True, return_dict=True,
                    )
                last_hidden = outputs.hidden_states[-1]
                self._cached_hA = self._gather_hA(last_hidden, input_ids, 1)[:, 0].detach()  # [B, N_act, H]
                m_bar = self._gather_m_bar(last_hidden, input_ids, 1)[:, 0]                  # [B, N_mem, H]
                self._rnn_state = [m_bar[i].detach() for i in range(B)]

        with torch.autocast("cuda", dtype=torch.float32):
            h_V = self._split_raw_vis_by_step(raw_vis, image_grid_thw, B, 1, rnn_seqs[0]["num_cam"])  # [B,1,N_v,H]
            state = torch.tensor(
                np.stack([np.asarray(ex["state"], dtype=np.float32) for ex in examples]),
                device=device,
            )[:, None, :]                                                          # [B,1,state_dim]
            stereo_tokens = self._extract_stereo(examples, device)                 # [B,1,N_stereo,hd] | None
            h_A = self._cached_hA[:, None, :, :]                                   # [B,1,N_act,H] (cached, possibly stale)
            pred = self.action_model.predict_action(h_A, h_V, state, stereo=stereo_tokens)  # [B,1,H,dim]

        cur = pred[:, 0, :, :].detach().cpu().numpy()   # [B, H, dim]
        logging.info(f"[RNN] action[0,0,:5]={cur[0,0,:5]}")
        return {"normalized_actions": cur}


if __name__ == "__main__":
    # tiny end-to-end test: dual-Pass forward + backward on random data (needs GPU + base ckpt)
    from omegaconf import OmegaConf

    T, H_ACTION = 3, 50
    ckpt = "/mnt/workspace/yama/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct-MemoryAction"
    cfg = OmegaConf.create(
        {
            "framework": {
                "name": "QwenZone",
                "qwenvl": {"base_vlm": ckpt, "attn_implementation": "sdpa"},
                "qwenzone": {
                    "train_seq_len": T, "N_act": 4, "N_mem": 4, "max_history": 16,
                    "lambda_rnn": 0.1, "d_max": 2, "infer_mode": "full",
                },
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
    print(f"[cfg ] lambda_rnn={model.lambda_rnn} d_max={model.d_max} infer_mode={model.infer_mode}")

    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    def make_sample(lang):
        return {
            "action": np.random.uniform(-1, 1, (T + H_ACTION - 1, 14)).astype(np.float16),
            "image": [[img] * T, [img] * T, [img] * T],  # [3 cam][T]
            "lang": lang,
            "state": np.random.uniform(-1, 1, (T, 14)).astype(np.float16),
        }

    batch = [make_sample("pick up the red block"), make_sample("open the drawer")]

    # ── dual-Pass forward + backward ──
    out = model(batch)
    print(f"[forward] total={out['action_loss'].item():.4f} "
          f"parallel={out['parallel_loss'].item():.4f} rnn={out['rnn_loss'].item():.4f}")
    out["action_loss"].backward()
    print("[backward] OK")

    # ── tiny training: both losses should decrease ──
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    p_losses, r_losses = [], []
    for step in range(5):
        opt.zero_grad()
        o = model(batch)
        o["action_loss"].backward()
        opt.step()
        p_losses.append(o["parallel_loss"].item())
        r_losses.append(o["rnn_loss"].item())
    print(f"[train] parallel losses: {[round(x, 4) for x in p_losses]}")
    print(f"[train] rnn      losses: {[round(x, 4) for x in r_losses]}")

    # ── hook-equivalence: injecting the *original* 🌱 embedding must reproduce the
    #    no-injection RNN forward exactly (proves the injection mechanism is a no-op
    #    when the injected value equals the placeholder embedding). ──
    model.eval()
    mem_init_emb = model._mem_init_embed().detach()  # [H] == the 🌱 embedding lookup
    ex = batch[0]
    images_by_cam = [[to_pil_preserve(im) for im in cam] for cam in ex["image"]]
    rnn_seqs = [model._build_rnn_sequence([images_by_cam[c][t] for c in range(3)], model._lang_ids(ex))
                for t in range(T)]
    input_ids, attn_4d, pixel_values, image_grid_thw, _ = model._assemble_batch(
        rnn_seqs, model.qwen_vl_interface.model.device
    )

    def rnn_forward_with_inject(inject_values):
        model._rnn_inject_values = inject_values
        model._rnn_inject_mask2d = input_ids == model.memory_init_id
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with model._injection_ctx():
                o = model.qwen_vl_interface(input_ids=input_ids, pixel_values=pixel_values,
                                            image_grid_thw=image_grid_thw, attention_mask=attn_4d,
                                            output_hidden_states=True, return_dict=True)
        return o.hidden_states[-1]

    # baseline: inject exactly the 🌱 embedding (N_mem copies) into every row → must equal no-inject path
    H_emb = mem_init_emb.shape[-1]
    inject_same = mem_init_emb.view(1, 1, -1).expand(input_ids.shape[0], model.N_mem, H_emb).contiguous()
    hs_inject_same = rnn_forward_with_inject(inject_same)

    # corrupt the 🌱 embedding → must DIFFER (proves the hook actually takes effect)
    inject_diff = inject_same + 0.5
    hs_inject_diff = rnn_forward_with_inject(inject_diff)

    # no-injection reference (hook that copies inputs_embeds unchanged)
    ref_hook = lambda m, a, k: None
    model._rnn_inject_mask2d = input_ids == model.memory_init_id
    h_no = model.qwen_vl_interface.model.model.language_model.register_forward_pre_hook(ref_hook, with_kwargs=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o_ref = model.qwen_vl_interface(input_ids=input_ids, pixel_values=pixel_values,
                                        image_grid_thw=image_grid_thw, attention_mask=attn_4d,
                                        output_hidden_states=True, return_dict=True)
    h_no.remove()
    hs_ref = o_ref.hidden_states[-1]

    d_same = (hs_inject_same.float() - hs_ref.float()).abs().mean().item()
    d_diff = (hs_inject_diff.float() - hs_ref.float()).abs().mean().item()
    print(f"[equiv] |Δ| inject=🌱emb vs no-inject: {d_same:.2e} (should be ~0)")
    print(f"[equiv] |Δ| inject=corrupt  vs no-inject: {d_diff:.2e} (should be >0)")
    assert d_same < 1e-3, "injecting the original embedding should be a no-op"
    assert d_diff > 1e-3, "corrupting the injected value must change the output"
    print("[equiv] PASS")

    # ── RNN inference smoke test (mode-2) ──
    model.infer_mode = "rnn"
    model.reset_history()
    cur_ex = {"image": [img, img, img], "lang": "pick up the red block",
              "state": np.random.uniform(-1, 1, (14,)).astype(np.float16)}
    pred = model.predict_action([cur_ex])
    print(f"[infer-rnn] out shape = {pred['normalized_actions'].shape} (expect (1, {H_ACTION}, 14))")
    # second step uses the carried m_state
    pred2 = model.predict_action([cur_ex])
    print(f"[infer-rnn] step-2 OK, shape = {pred2['normalized_actions'].shape}")

    # ── Async-stride inference smoke test (rnn + vlm_stride > 0) ──
    model.infer_mode = "rnn"
    model.vlm_stride = 3
    model.reset_history()
    import time
    # run vlm_stride+2 steps; the LLM refreshes at step 1 (cache empty) and step 1+stride,
    # and reuses the cached intent in between.
    stride_preds = []
    t0 = time.time()
    for s in range(model.vlm_stride + 2):
        p = model.predict_action([cur_ex])
        stride_preds.append(p["normalized_actions"])
    dt = time.time() - t0
    print(f"[infer-rnn-stride] {len(stride_preds)} steps, out shape = {stride_preds[0].shape}, "
          f"total {dt:.3f}s ({dt/len(stride_preds)*1000:.1f} ms/step)")
    print(f"[infer-rnn-stride] step counter = {model._vlm_step}, cache present = {model._cached_hA is not None}")
    assert all(a.shape == (1, H_ACTION, 14) for a in stride_preds), "rnn-stride output shape mismatch"
    assert model._cached_hA is not None, "cached action intent should be populated"
    print("[infer-rnn-stride] PASS")
    print("\n=> ALL TESTS PASSED")

