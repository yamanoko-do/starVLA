# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""ZoneMemory Action Head — Phase 1 transformer decoder for the QwenZone framework.

Inputs (per timestep t, all timesteps processed in parallel by flattening T into batch):
  - h_A : [B, T, N_act, input_dim]   last-layer hidden states of the action-intent tokens A_t
  - h_V : [B, T, N_v,   input_dim]   last-layer hidden states of the vision tokens V_t
  - state : [B, T, state_dim]        proprioception

Output:
  - [B, T, H_action, action_dim]      predicted future H_action-step action chunk per step,
                                      non-autoregressive (learnable query tokens, like OFT).

Design: at each step a small bidirectional transformer runs over
  [cmd_tokens(h_A), vis_tokens(pooled h_V), state_tokens(s_t), action_query]
and the action_query positions are decoded into the action chunk.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZoneMemoryActionHead(nn.Module):
    def __init__(
        self,
        input_dim=2560,
        hidden_dim=512,
        action_dim=14,
        H_action=50,
        N_act=8,
        N_vis_tokens=8,
        N_state_tokens=4,
        N_stereo=0,            # 0 = no stereo; >0 = stereo tokens per step
        stereo_dim=512,        # input dim of stereo tokens (already projected)
        state_dim=14,
        nhead=8,
        num_layers=4,
        dropout=0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.H_action = H_action
        self.N_act = N_act
        self.N_vis_tokens = N_vis_tokens
        self.N_state_tokens = N_state_tokens
        self.N_stereo = N_stereo

        # h_A (N_act tokens) -> command tokens (keep fine-grained intent, no pooling)
        self.cmd_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
        )
        # h_V (N_v tokens) -> pool to N_vis_tokens, then project
        self.vis_pool = nn.AdaptiveAvgPool1d(N_vis_tokens)
        self.vis_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
        )
        # state -> N_state_tokens
        self.state_proj = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim * N_state_tokens),
        )

        # stereo tokens (already pooled+projected from StereoEncoder) -> optional fine-tune projection
        if N_stereo > 0:
            self.stereo_proj = nn.Sequential(
                nn.LayerNorm(stereo_dim),
                nn.Linear(stereo_dim, hidden_dim),
            )
        else:
            self.stereo_proj = None

        # learnable action queries -> one-shot non-autoregressive prediction
        self.action_query = nn.Parameter(torch.randn(1, H_action, hidden_dim) * 0.02)

        # per-step positional embedding over [cmd, vis, state, stereo?, query]
        seq_len = N_act + N_vis_tokens + N_state_tokens + max(N_stereo, 0) + H_action
        self.pos = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward_step(self, h_A, h_V, state, stereo=None):
        """One timestep. h_A:[B,N_act,Hin] h_V:[B,N_v,Hin] state:[B,state_dim] stereo:[B,N_stereo,hd]|None -> [B,H_action,action_dim]."""
        B = h_A.shape[0]
        cmd = self.cmd_proj(h_A)                                   # [B, N_act, hd]

        v = h_V.transpose(1, 2)                                    # [B, Hin, N_v]
        v = self.vis_pool(v)                                       # [B, Hin, N_vis]
        vis = self.vis_proj(v.transpose(1, 2))                     # [B, N_vis, hd]

        st = self.state_proj(state).view(B, self.N_state_tokens, -1)  # [B, N_state, hd]
        q = self.action_query.expand(B, -1, -1)                    # [B, H_action, hd]

        parts = [cmd, vis, st]
        if self.N_stereo > 0 and stereo is not None:
            stereo_proj = self.stereo_proj(stereo)                  # [B, N_stereo, hd]
            parts.append(stereo_proj)
        parts.append(q)
        seq = torch.cat(parts, dim=1)                               # [B, seq_len, hd]
        seq = seq + self.pos[:, : seq.shape[1], :]
        out = self.transformer(seq)                                 # bidirectional (no mask)

        action_out = out[:, -self.H_action :, :]                    # query positions
        return self.decoder(action_out)                             # [B, H_action, action_dim]

    def forward(self, h_A, h_V, state, actions_target=None, stereo=None):
        """
        h_A: [B, T, N_act, input_dim]
        h_V: [B, T, N_v,   input_dim]
        state: [B, T, state_dim]
        stereo: [B, T, N_stereo, stereo_dim] | None
        actions_target (optional): [B, T, H_action, action_dim]
        Returns: loss (if target given) else predictions [B, T, H_action, action_dim].
        """
        B, T = h_A.shape[:2]
        N_v = h_V.shape[2]
        h_A_flat = h_A.reshape(B * T, self.N_act, -1)
        h_V_flat = h_V.reshape(B * T, N_v, -1)
        st_flat = state.reshape(B * T, -1)
        stereo_flat = stereo.reshape(B * T, self.N_stereo, -1) if stereo is not None else None

        pred = self.forward_step(h_A_flat, h_V_flat, st_flat, stereo_flat)  # [B*T, H_action, action_dim]
        pred = pred.reshape(B, T, self.H_action, self.action_dim)

        if actions_target is not None:
            return F.l1_loss(pred, actions_target)
        return pred

    def predict_action(self, h_A, h_V, state, stereo=None):
        """Inference: returns [B, T, H_action, action_dim]."""
        return self.forward(h_A, h_V, state, actions_target=None, stereo=stereo)


def get_action_model(config=None):
    """Factory: build ZoneMemoryActionHead from global framework config."""
    am = config.framework.action_model
    assert am.action_model_type == "ZoneMemory", (
        f"ZoneMemory factory expects action_model_type=='ZoneMemory', got {am.action_model_type}"
    )
    return ZoneMemoryActionHead(
        input_dim=am.action_hidden_dim,
        hidden_dim=int(am.get("hidden_size", 512)),
        action_dim=am.action_dim,
        H_action=int(am.action_horizon),
        N_act=int(am.get("N_act", 8)),
        N_vis_tokens=int(am.get("N_vis_tokens", 8)),
        N_state_tokens=int(am.get("N_state_tokens", 4)),
        N_stereo=int(am.get("N_stereo", 8)),
        stereo_dim=int(am.get("stereo_dim", am.get("hidden_size", 512))),
        state_dim=int(am.get("state_dim", am.action_dim)),
        nhead=int(am.get("nhead", 8)),
        num_layers=int(am.get("num_transformer_layers", 4)),
    )


if __name__ == "__main__":
    B, T, H_in, N_v = 2, 4, 2560, 192
    N_act, ACTION_DIM, H_ACTION = 8, 14, 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ZoneMemoryActionHead(
        input_dim=H_in, hidden_dim=512, action_dim=ACTION_DIM, H_action=H_ACTION,
        N_act=N_act, N_vis_tokens=8, N_state_tokens=4, state_dim=ACTION_DIM,
        nhead=8, num_layers=4,
    ).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    h_A = torch.randn(B, T, N_act, H_in, device=device)
    h_V = torch.randn(B, T, N_v, H_in, device=device)
    state = torch.randn(B, T, ACTION_DIM, device=device)

    # training: loss is scalar
    target = torch.randn(B, T, H_ACTION, ACTION_DIM, device=device)
    loss = model(h_A, h_V, state, target)
    print(f"train loss: {loss.item():.4f} (scalar={loss.dim()==0})")

    # inference: shape
    pred = model.predict_action(h_A, h_V, state)
    print(f"pred shape: {tuple(pred.shape)} (expected ({B},{T},{H_ACTION},{ACTION_DIM}))")
    assert tuple(pred.shape) == (B, T, H_ACTION, ACTION_DIM)

    # backward sanity
    loss.backward()
    print("backward OK")
