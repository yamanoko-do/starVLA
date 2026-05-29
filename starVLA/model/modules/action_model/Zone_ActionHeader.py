# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class StateEncoder(nn.Module):
    """Temporal CNN encoder: compresses [B, T_in, action_dim] → [B, T_out, hidden_dim].

    Uses stacked Conv1d with stride=2 to progressively halve the temporal dimension,
    then a final adaptive pooling to hit the exact target token count.
    """

    def __init__(self, in_dim=7, hidden_dim=512, out_tokens=8, in_len=50):
        super().__init__()
        self.out_tokens = out_tokens
        # Layer 1: T_in → T_in/2
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim // 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, hidden_dim // 4),
            nn.SiLU(),
        )
        # Layer 2: T_in/2 → T_in/4
        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim // 4, hidden_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.SiLU(),
        )
        # Layer 3: T_in/4 → T_in/8
        self.conv3 = nn.Sequential(
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, hidden_dim),
            nn.SiLU(),
        )
        # Adaptive pooling to exact out_tokens (handles cases where strides don't hit exactly)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(out_tokens)

    def forward(self, x):
        # x: [B, T_in, D] → [B, D, T_in] for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adaptive_pool(x)  # [B, hidden_dim, out_tokens]
        x = x.permute(0, 2, 1)      # [B, out_tokens, hidden_dim]
        return x


class ZoneActionHeadTransformer(nn.Module):
    """Transformer-based Action Head using VLM output h as Prefix Context.

    Core Idea:
    VLM representation h is projected into a few 'Command Tokens' at the start 
    of the sequence. State history s is projected into 'State Tokens'. 
    Through Causal Self-Attention, the state tokens are conditioned on the 
    command tokens, and future actions are auto-regressively generated.
    The command tokens act as a persistent 'motor mode' set by the brain.
    """

    def __init__(
        self,
        input_dim=2048,
        hidden_dim=512,
        action_dim=7,
        NUM_ACTIONS_CHUNK=8,
        state_history_len=50,
        num_cmd_tokens=4,     # Number of tokens representing the VLM command h
        num_state_tokens=8,   # Compressed state tokens (output of CNN encoder)
        nhead=8,
        num_transformer_layers=4,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.NUM_ACTIONS_CHUNK = NUM_ACTIONS_CHUNK
        self.state_history_len = state_history_len
        self.hidden_dim = hidden_dim
        self.num_cmd_tokens = num_cmd_tokens
        self.num_state_tokens = num_state_tokens

        # Learnable query tokens for fast one-shot inference (no autoregressive loop)
        self.action_query = nn.Parameter(torch.randn(1, NUM_ACTIONS_CHUNK, hidden_dim) * 0.02)

        # --- 1. Projectors ---
        # VLM h -> Command Tokens
        self.vlm_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, num_cmd_tokens * hidden_dim),
            nn.SiLU(),
        )

        # State history encoder: temporal conv compression (50 frames → num_state_tokens)
        self.state_encoder = StateEncoder(
            in_dim=action_dim,
            hidden_dim=hidden_dim,
            out_tokens=num_state_tokens,
            in_len=state_history_len,
        )

        # Previous action -> Action Tokens (for autoregressive generation)
        self.action_proj = nn.Sequential(
            nn.LayerNorm(action_dim),
            nn.Linear(action_dim, hidden_dim),
        )

        # --- 2. Positional Encoding ---
        max_len = num_cmd_tokens + num_state_tokens + NUM_ACTIONS_CHUNK
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, hidden_dim) * 0.02)

        # --- 3. Transformer Engine ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm is generally more stable
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # --- 4. Action Decoder ---
        self.action_decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim),
        )

    def _pool_vl_embs(self, vl_embs, encoder_attention_mask=None):
        if encoder_attention_mask is not None:
            mask = encoder_attention_mask.float().unsqueeze(-1)
            pooled = (vl_embs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = vl_embs.mean(dim=1)
        return pooled

    def predict_action(self, vl_embs, state=None, encoder_attention_mask=None):
        """
        Fast one-shot inference: learnable query tokens attend to [CMD, STATE]
        and predict all actions in a single forward pass (no autoregressive loop).
        """
        B = vl_embs.shape[0]
        device = vl_embs.device

        # 1. Command + State tokens (encode once)
        vlm_feat = self._pool_vl_embs(vl_embs, encoder_attention_mask)
        cmd_tokens = self.vlm_proj(vlm_feat).view(B, self.num_cmd_tokens, self.hidden_dim)

        if state is not None:
            state_tokens = self.state_encoder(state)
        else:
            state_tokens = torch.zeros(B, self.num_state_tokens, self.hidden_dim, device=device)

        # 2. Learnable action query tokens (shared across batch)
        action_q = self.action_query.expand(B, -1, -1)  # [B, chunk, D]

        # 3. Full sequence: [CMD, STATE, ACTION_QUERY]
        seq = torch.cat([cmd_tokens, state_tokens, action_q], dim=1)
        L = seq.shape[1]
        seq = seq + self.pos_embedding[:, :L, :]

        causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=device)
        out = self.transformer(seq, mask=causal_mask, is_causal=True)

        # Decode action positions
        start_idx = self.num_cmd_tokens + self.num_state_tokens
        action_out = out[:, start_idx:, :]
        return self.action_decoder(action_out)  # [B, chunk, action_dim]

    def forward(self, vl_embs, actions=None, state=None, encoder_attention_mask=None):
        """
        Training forward pass uses Teacher Forcing (parallel prediction).
        """
        B = vl_embs.shape[0]
        device = vl_embs.device

        # 1. Command Tokens
        vlm_feat = self._pool_vl_embs(vl_embs, encoder_attention_mask)
        cmd_tokens = self.vlm_proj(vlm_feat).view(B, self.num_cmd_tokens, self.hidden_dim)

        # 2. State Tokens (compressed by CNN encoder)
        if state is not None:
            state_tokens = self.state_encoder(state)  # [B, num_state_tokens, hidden_dim]
        else:
            state_tokens = torch.zeros(B, self.num_state_tokens, self.hidden_dim, device=device)

        # 3. Target Action Tokens (Shifted Right for Teacher Forcing)
        if actions is not None:
            # Prepend the last known state to the actions
            if state is not None:
                first_input = state[:, -1, :]
            else:
                first_input = torch.zeros(B, self.action_dim, device=device)
            
            # shifted_actions: [s_t, a_t, a_{t+1}, ..., a_{T-1}]
            shifted_actions = torch.cat([first_input.unsqueeze(1), actions[:, :-1, :]], dim=1)
            action_tokens = self.action_proj(shifted_actions)

            # 4. Full Sequence: [CMD, STATE, SHIFTED_ACTIONS]
            seq = torch.cat([cmd_tokens, state_tokens, action_tokens], dim=1)
            L = seq.shape[1]
            seq = seq + self.pos_embedding[:, :L, :]

            # Causal Mask
            causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=device)

            # 5. Transformer Forward
            out = self.transformer(seq, mask=causal_mask, is_causal=True)

            # 6. Decode only the action part of the output
            # Indices corresponding to the action tokens
            start_idx = self.num_cmd_tokens + self.num_state_tokens
            action_out = out[:, start_idx:, :] 
            
            pred_actions = self.action_decoder(action_out) # [B, chunk, action_dim]
            
            return F.l1_loss(pred_actions, actions)
        else:
            return self.predict_action(vl_embs, state, encoder_attention_mask)


def get_action_model(config=None):
    """
    Factory: build ZoneActionHeadTransformer from global framework config.

    Args:
        config: Global config (expects config.framework.action_model namespace).
    Returns:
        ZoneActionHeadTransformer
    """
    action_cfg = config.framework.action_model
    action_dim = action_cfg.action_dim
    action_horizon = int(action_cfg.action_horizon)
    state_history_len = int(action_cfg.get("state_history_len", 50))
    num_cmd_tokens = int(action_cfg.get("num_cmd_tokens", 4))
    num_state_tokens = int(action_cfg.get("num_state_tokens", 8))
    nhead = int(action_cfg.get("nhead", 8))
    num_transformer_layers = int(action_cfg.get("num_transformer_layers", 4))
    hidden_dim = action_cfg.get("hidden_size", 512)

    return ZoneActionHeadTransformer(
        input_dim=action_cfg.action_hidden_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        NUM_ACTIONS_CHUNK=action_horizon,
        state_history_len=state_history_len,
        num_cmd_tokens=num_cmd_tokens,
        num_state_tokens=num_state_tokens,
        nhead=nhead,
        num_transformer_layers=num_transformer_layers,
    )


if __name__ == "__main__":
    import time

    # --- config ---
    B = 4                
    L = 115              
    H = 1024             
    ACTION_DIM = 7
    CHUNK = 8
    HIST = 50
    # --------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ZoneActionHeadTransformer(
        input_dim=H,
        hidden_dim=512,
        action_dim=ACTION_DIM,
        NUM_ACTIONS_CHUNK=CHUNK,
        state_history_len=HIST,
        num_cmd_tokens=4,
        nhead=8,
        num_transformer_layers=3,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_tok = model.num_cmd_tokens + model.num_state_tokens + CHUNK
    print(f"ZoneActionHeadTransformer params: {n_params:,}")
    print(f"Token breakdown: {model.num_cmd_tokens} cmd + {model.num_state_tokens} state (from {HIST}f CNN) + {CHUNK} action = {n_tok} tokens")
    print()

    vl_embs = torch.randn(B, L, H, device=device)
    actions = torch.randn(B, CHUNK, ACTION_DIM, device=device)
    state_hist = torch.randn(B, HIST, ACTION_DIM, device=device)
    mask = torch.ones(B, L, device=device)

    # Test Training (Teacher Forcing - Fast)
    loss = model(vl_embs, actions=actions, state=state_hist, encoder_attention_mask=mask)
    print(f"Training Loss: {loss.item():.4f}")

    # Test Inference
    with torch.no_grad():
        pred = model.predict_action(vl_embs, state=state_hist, encoder_attention_mask=mask)
    print(f"Inference shape: {pred.shape} (expected: [{B}, {CHUNK}, {ACTION_DIM}])")

    # --- speed benchmark ---
    N_WARMUP = 10
    N_BENCH = 200

    def bench(name, fn):
        for _ in range(N_WARMUP):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_BENCH):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        print(f"{name:>16s} ({N_BENCH} runs): {elapsed/N_BENCH*1000:.2f} ms/call  →  {N_BENCH/elapsed:.0f} Hz")

    bench("Training",   lambda: model(vl_embs, actions=actions, state=state_hist, encoder_attention_mask=mask))
    bench("Inference",  lambda: model.predict_action(vl_embs, state=state_hist, encoder_attention_mask=mask))
