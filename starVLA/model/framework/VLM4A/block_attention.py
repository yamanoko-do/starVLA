# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Block-wise attention mask for the QwenZone memory-token framework.
#
# Implements the visibility rules from the design doc (tt.md):
#   - within a timestep block (V_t, A_t, M_t): bidirectional
#   - across timesteps: controlled — history only flows through Memory tokens
#   - M_init sits at the head of the sequence as the initial memory (timestep -1)
#
# Returns a [L, L] additive mask (0 = visible, finfo(dtype).min = masked) that can
# be fed directly as a 4D attention mask to Qwen3-VL (transformers 4.57.0 supports
# custom 4D masks via masking_utils.create_causal_mask early-exit).

import torch


def build_block_attention_mask(
    token_types,
    token_steps,
    dtype=torch.bfloat16,
    device="cpu",
):
    """Build a [L, L] additive attention mask from per-token (type, timestep) labels.

    Visibility rules (query -> key), with M_{-1} == M_init:
      * M_init  : {L, M_init}                       (initial memory reads task + self)
      * L       : {L}                                (language tokens only see language)
      * V_t     : {L, M_{t-1}, V_t}
      * A_t     : {L, M_{t-1}, V_t, A_t}             (different timesteps fully isolated)
      * M_t     : {L, M_{t-1}, V_t, A_t, M_t}        (same-group memory bidirectional)

    Args:
        token_types: list[str] of length L, each in {"M_init", "V", "L", "A", "M"}.
        token_steps: list[int] of length L, timestep; M_init uses -1, L arbitrary.
        dtype: additive mask dtype (0=visible, finfo.min=masked). Must be floating.
        device: torch device.
    Returns:
        torch.Tensor of shape [L, L]. Diagonal is 0 for every real token (each token
        sees itself), which Qwen3-VL relies on to derive the 2D padding mask.
    """
    assert len(token_types) == len(token_steps), "types and steps must align"
    L = len(token_types)

    is_Minit = torch.tensor([t == "M_init" for t in token_types], device=device)
    is_V = torch.tensor([t == "V" for t in token_types], device=device)
    is_A = torch.tensor([t == "A" for t in token_types], device=device)
    is_M = torch.tensor([t == "M" for t in token_types], device=device)
    is_L = torch.tensor([t == "L" for t in token_types], device=device)
    steps = torch.tensor(token_steps, device=device, dtype=torch.long)

    qs = steps.unsqueeze(1)  # [L, 1] query timestep
    ks = steps.unsqueeze(0)  # [1, L] key timestep
    same_block = ks == qs          # [L, L]
    prev_block = ks == (qs - 1)    # [L, L]  (qs-1 == -1 matches M_init)

    # key-side predicates, shape [1, L]
    k_L = is_L.unsqueeze(0)
    k_V = is_V.unsqueeze(0)
    k_A = is_A.unsqueeze(0)
    k_M = is_M.unsqueeze(0)
    k_Minit = is_Minit.unsqueeze(0)
    k_M_or_init = (is_M | is_Minit).unsqueeze(0)

    # Every query sees all language tokens (and L sees L via this too)
    visible = k_L.expand(L, L).clone()

    # query-side masks, shape [L, 1]
    q_Minit = is_Minit.unsqueeze(1)
    q_V = is_V.unsqueeze(1)
    q_A = is_A.unsqueeze(1)
    q_M = is_M.unsqueeze(1)

    # M_init query: sees itself
    visible |= q_Minit & k_Minit
    # V query: {V same-block, M_{t-1}}
    visible |= q_V & (k_V & same_block)
    visible |= q_V & (k_M_or_init & prev_block)
    # A query: {V same-block, A same-block, M_{t-1}}
    visible |= q_A & (k_V & same_block)
    visible |= q_A & (k_A & same_block)
    visible |= q_A & (k_M_or_init & prev_block)
    # M query: {V same-block, A same-block, M same-block, M_{t-1}}
    visible |= q_M & (k_V & same_block)
    visible |= q_M & (k_A & same_block)
    visible |= q_M & (k_M & same_block)
    visible |= q_M & (k_M_or_init & prev_block)

    neg = torch.finfo(dtype).min
    mask = torch.where(
        visible,
        torch.zeros((), device=device, dtype=dtype),
        torch.full((), neg, device=device, dtype=dtype),
    )
    return mask  # [L, L]


if __name__ == "__main__":
    # ---- unit test: a T=2 mini-sequence ----
    # layout: M_init | V0 V0 | L L | A0 | M0 | V1 V1 | A1 | M1
    types = ["M_init", "V", "V", "L", "L", "A", "M", "V", "V", "A", "M"]
    steps = [-1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    # indices:        0   1  2  3  4  5  6  7  8  9  10
    m = build_block_attention_mask(types, steps, dtype=torch.float32)
    MIN = torch.finfo(torch.float32).min

    def vis(q, k):
        return m[q, k].item() == 0.0

    def masked(q, k):
        return m[q, k].item() == MIN

    checks = {
        "V1 cannot see V0 (no history vision)": masked(7, 1),
        "V1 can see M0 (history via memory)": vis(7, 6),
        "A0 cannot see A1 (cross-step A isolated)": masked(5, 9),
        "A0 cannot see same-block M0": masked(5, 6),
        "M0 can see itself (diagonal)": vis(6, 6),
        "M0 can see A0 (same block)": vis(6, 5),
        "M1 can see M0 (prev block)": vis(10, 6),
        "M1 can see V1 (same block)": vis(10, 7),
        "M_init can see L": vis(0, 3),
        "M_init cannot see V0": masked(0, 1),
        "V0 can see M_init (prev block at t=0)": vis(1, 0),
        "A0 can see M_init (prev block at t=0)": vis(5, 0),
        "L can see L": vis(3, 4),
        "L cannot see V": masked(3, 1),
        "M_init diagonal is 0": vis(0, 0),
        "V0 diagonal is 0": vis(1, 1),
        "A0 diagonal is 0": vis(5, 5),
        "L diagonal is 0": vis(3, 3),
    }

    all_ok = True
    for name, ok in checks.items():
        flag = "OK " if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{flag}] {name}")

    print("\n=> ALL PASSED" if all_ok else "\n=> SOME FAILED")
    assert all_ok, "block attention mask unit test failed"
