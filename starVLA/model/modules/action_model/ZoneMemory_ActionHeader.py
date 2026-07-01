# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""ZoneMemory Action Head — transformer decoder for the QwenZone framework.

Inputs (per timestep t, all timesteps processed in parallel by flattening T into batch):
  - h_A : [B, T, N_act, input_dim]   last-layer hidden states of the action-intent tokens A_t
  - h_V : [B, T, N_v,   input_dim]   RAW vision tokens V_t (NOT pooled — full spatial info kept)
  - state : [B, T, state_dim]        proprioception (single-step per t, not historical)
  - stereo: [B, T, N_stereo, hidden_dim]  fused stereo tokens (already at hidden_dim)

Output:
  - [B, T, H_action, action_dim]      predicted future H_action-step action chunk per step,
                                      non-autoregressive (learnable query tokens, like OFT).

Sequence per step: [cmd(N_act) | vis(N_v raw) | state(N_state) | stereo(N_stereo) | query(H_action)]
N_v is dynamic (depends on image resolution) → pos embedding uses a max-len buffer, sliced per call.
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
        N_act=16,
        N_state_tokens=4,
        N_stereo=64,           # stereo tokens per step (0 = disable); already at hidden_dim
        state_dim=28,
        nhead=8,
        num_layers=4,
        dropout=0.1,
        pos_max_len=512,       # max sequence length for the positional-embedding buffer
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.H_action = H_action
        self.N_act = N_act
        self.N_state_tokens = N_state_tokens
        self.N_stereo = N_stereo

        # h_A (N_act tokens) -> command tokens (intent, no pooling)
        self.cmd_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
        )
        # h_V (N_v tokens, RAW unpooled) -> project to hidden_dim
        self.vis_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
        )
        # state -> N_state_tokens
        self.state_proj = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim * N_state_tokens),
        )
        # stereo tokens arrive already at hidden_dim (from StereoEncoder ResConv2)
        # → no extra projection; optional LayerNorm for stability
        if N_stereo > 0:
            self.stereo_norm = nn.LayerNorm(hidden_dim)
        else:
            self.stereo_norm = None

        # learnable action queries -> one-shot non-autoregressive prediction
        self.action_query = nn.Parameter(torch.randn(1, H_action, hidden_dim) * 0.02)

        # positional-embedding buffer (N_v is dynamic → use max-len and slice)
        self.pos = nn.Parameter(torch.randn(1, pos_max_len, hidden_dim) * 0.02)

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

    def forward_step(self, h_A, h_V, state, stereo=None,
                      vis_mask_probability=0.0, vis_mask_ratio=0.0,
                      cmd_mask_probability=0.0, cmd_mask_ratio=0.0):
        """One timestep.
        h_A:[B,N_act,Hin] h_V:[B,N_v,Hin] state:[B,state_dim] stereo:[B,N_stereo,hd]|None
        -> [B,H_action,action_dim].

        Args:
            vis_mask_probability: 应用视觉遮蔽的概率 (0.0-1.0), 默认0.0
            vis_mask_ratio: 视觉遮蔽比例的上限 (0.0-1.0), 默认0.0
            cmd_mask_probability: 应用cmd遮蔽的概率 (0.0-1.0), 默认0.0
            cmd_mask_ratio: cmd遮蔽比例的上限 (0.0-1.0), 默认0.0
        """
        B = h_A.shape[0]
        cmd = self.cmd_proj(h_A)                                       # [B, N_act, hd]
        vis = self.vis_proj(h_V)                                       # [B, N_v, hd]  (raw, N_v dynamic)

        # 训练时随机遮蔽部分视觉token
        # vis_mask_probability: 应用遮蔽的概率 (0.0-1.0)
        # vis_mask_ratio: 遮蔽比例的上限 (0.0-1.0)，实际遮蔽比例在 [0, vis_mask_ratio] 之间随机选择
        if self.training and vis_mask_probability > 0:
            if torch.rand(1).item() < vis_mask_probability:  # 决定是否应用遮蔽
                N_v = vis.shape[1]
                # 动态选择实际遮蔽比例：在 [0, vis_mask_ratio] 之间随机选择
                actual_mask_ratio = torch.rand(1).item() * vis_mask_ratio
                mask = torch.rand(B, N_v, device=vis.device) > actual_mask_ratio  # [B, N_v]
                mask = mask.unsqueeze(-1)  # [B, N, 1]
                vis = vis * mask.float()  # 遮蔽被选中的token

        # 训练时随机遮蔽部分cmd token
        # cmd_mask_probability: 应用遮蔽的概率 (0.0-1.0)
        # cmd_mask_ratio: 遮蔽比例的上限 (0.0-1.0)，实际遮蔽比例在 [0, cmd_mask_ratio] 之间随机选择
        if self.training and cmd_mask_probability > 0:
            if torch.rand(1).item() < cmd_mask_probability:  # 决定是否应用遮蔽
                N_cmd = cmd.shape[1]
                # 动态选择实际遮蔽比例：在 [0, cmd_mask_ratio] 之间随机选择
                actual_cmd_mask_ratio = torch.rand(1).item() * cmd_mask_ratio
                mask = torch.rand(B, N_cmd, device=cmd.device) > actual_cmd_mask_ratio  # [B, N_act]
                mask = mask.unsqueeze(-1)  # [B, N_act, 1]
                cmd = cmd * mask.float()  # 遮蔽被选中的token

        st = self.state_proj(state).view(B, self.N_state_tokens, -1)   # [B, N_state, hd]
        q = self.action_query.expand(B, -1, -1)                        # [B, H_action, hd]

        parts = [cmd, vis, st]
        if self.N_stereo > 0 and stereo is not None:
            parts.append(self.stereo_norm(stereo))                     # [B, N_stereo, hd]
        parts.append(q)
        seq = torch.cat(parts, dim=1)                                  # [B, seq_len, hd]
        seq = seq + self.pos[:, : seq.shape[1], :]                     # slice pos to seq_len
        out = self.transformer(seq)                                    # bidirectional (no mask)

        action_out = out[:, -self.H_action :, :]                       # query positions
        return self.decoder(action_out)                                # [B, H_action, action_dim]

    def forward(self, h_A, h_V, state, actions_target=None, stereo=None,
                vis_mask_probability=0.0, vis_mask_ratio=0.0,
                cmd_mask_probability=0.0, cmd_mask_ratio=0.0):
        """
        h_A: [B, T, N_act, input_dim]
        h_V: [B, T, N_v,   input_dim]
        state: [B, T, state_dim]
        stereo: [B, T, N_stereo, hidden_dim] | None
        actions_target (optional): [B, T, H_action, action_dim]
        vis_mask_probability: 应用视觉遮蔽的概率 (0.0-1.0), 默认0.0
        vis_mask_ratio: 视觉遮蔽比例的上限 (0.0-1.0), 默认0.0
        cmd_mask_probability: 应用cmd遮蔽的概率 (0.0-1.0), 默认0.0
        cmd_mask_ratio: cmd遮蔽比例的上限 (0.0-1.0), 默认0.0
        Returns: loss (if target given) else predictions [B, T, H_action, action_dim].
        """
        B, T = h_A.shape[:2]
        N_v = h_V.shape[2]
        h_A_flat = h_A.reshape(B * T, self.N_act, -1)
        h_V_flat = h_V.reshape(B * T, N_v, -1)
        st_flat = state.reshape(B * T, -1)
        stereo_flat = stereo.reshape(B * T, self.N_stereo, -1) if stereo is not None else None

        pred = self.forward_step(h_A_flat, h_V_flat, st_flat, stereo_flat,
                                 vis_mask_probability, vis_mask_ratio,
                                 cmd_mask_probability, cmd_mask_ratio)  # [B*T, H_action, action_dim]
        pred = pred.reshape(B, T, self.H_action, self.action_dim)

        if actions_target is not None:
            return F.l1_loss(pred, actions_target)
        return pred

    def predict_action(self, h_A, h_V, state, stereo=None):
        """Inference: returns [B, T, H_action, action_dim]."""
        return self.forward(h_A, h_V, state, actions_target=None, stereo=stereo)


def get_action_model(config=None):
    """Factory: build ZoneMemoryActionHead from global framework config.

    N_act / N_stereo are read from ``config.framework.qwenzone.*`` (single source of truth).
    vis is NOT pooled (raw N_v tokens), so N_vis_tokens is gone. stereo_dim == hidden_size.
    """
    am = config.framework.action_model
    qz = config.framework.qwenzone
    assert am.action_model_type == "ZoneMemory", (
        f"ZoneMemory factory expects action_model_type=='ZoneMemory', got {am.action_model_type}"
    )
    hidden_size = int(am.get("hidden_size", 512))
    n_stereo = int(qz.get("stereo", {}).get("N_stereo_tokens", 0)) if qz.get("stereo") else 0

    return ZoneMemoryActionHead(
        input_dim=am.action_hidden_dim,
        hidden_dim=hidden_size,
        action_dim=am.action_dim,
        H_action=int(am.action_horizon),
        N_act=int(qz.get("N_act", 16)),
        N_state_tokens=int(am.get("N_state_tokens", 4)),
        N_stereo=n_stereo,
        state_dim=int(am.get("state_dim", am.action_dim)),
        nhead=int(am.get("nhead", 8)),
        num_layers=int(am.get("num_transformer_layers", 4)),
    )


if __name__ == "__main__":
    B, T, H_in, N_v = 2, 4, 2560, 64   # N_v=64 (224×224 Qwen tokens), unpooled
    N_act, N_STEREO, ACTION_DIM, H_ACTION = 16, 64, 14, 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ZoneMemoryActionHead(
        input_dim=H_in, hidden_dim=512, action_dim=ACTION_DIM, H_action=H_ACTION,
        N_act=N_act, N_state_tokens=4, N_stereo=N_STEREO, state_dim=28,
        nhead=8, num_layers=4,
    ).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    seq_len = N_act + N_v + 4 + N_STEREO + H_ACTION
    print(f"per-step seq len: {seq_len} (cmd{N_act}+vis{N_v}+state4+stereo{N_STEREO}+query{H_ACTION})")

    h_A = torch.randn(B, T, N_act, H_in, device=device)
    h_V = torch.randn(B, T, N_v, H_in, device=device)
    state = torch.randn(B, T, 28, device=device)
    stereo = torch.randn(B, T, N_STEREO, 512, device=device)

    target = torch.randn(B, T, H_ACTION, ACTION_DIM, device=device)
    loss = model(h_A, h_V, state, target, stereo=stereo)
    print(f"train loss: {loss.item():.4f} (scalar={loss.dim()==0})")

    pred = model.predict_action(h_A, h_V, state, stereo=stereo)
    print(f"pred shape: {tuple(pred.shape)} (expected ({B},{T},{H_ACTION},{ACTION_DIM}))")
    loss.backward()
    print("backward OK")
