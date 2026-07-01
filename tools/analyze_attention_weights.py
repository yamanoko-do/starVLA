#!/usr/bin/env python3
"""
Analyze ZoneMemoryActionHead attention on REAL training data across timesteps.

Loads full framework (VLM + action head), runs inference on a real HDF5 episode,
captures per-layer attention weights at each timestep, reports per-modality
distribution and how it evolves over time.

Usage:
    python3 tools/analyze_attention_weights.py
    python3 tools/analyze_attention_weights.py --episode 5 --max_steps 20
    python3 tools/analyze_attention_weights.py --ckpt /path/to/checkpoint.pt --compare
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# CRITICAL: disable fast path so _sa_block monkey-patch gets called
torch.backends.mha.set_fastpath_enabled(False)

from starVLA.model.framework.base_framework import baseframework


def load_episode_data(episode_path, max_steps=20):
    """Load observation data from a single HDF5 episode.

    HDF5 structure:
      observation/
        head_camera_left/rgb:  [T] bytes (JPEG/PNG encoded)
        head_camera_right/rgb: [T] bytes
      endpose/
        left_endpose:   [T, 7]
        right_endpose:  [T, 7]
        left_gripper:   [T]
        right_gripper:  [T]
      joint_action/
        left_arm:       [T, 6]
        right_arm:      [T, 6]
        left_gripper:   [T]
        right_gripper:  [T]
    """
    import io

    f = h5py.File(episode_path, 'r')

    obs_grp = f['observation']
    endpose = f['endpose']
    joints = f['joint_action']

    # RGB images are stored as encoded bytes
    head_left_bytes = obs_grp['head_camera_left']['rgb']
    if 'head_camera_right' in obs_grp:
        head_right_bytes = obs_grp['head_camera_right']['rgb']
    else:
        head_right_bytes = None

    T = min(len(head_left_bytes), max_steps)

    frames = []
    for t in range(T):
        # Decode JPEG/PNG bytes → PIL Image
        head_left = Image.open(io.BytesIO(head_left_bytes[t])).convert('RGB')
        if head_right_bytes is not None:
            head_right = Image.open(io.BytesIO(head_right_bytes[t])).convert('RGB')
        else:
            head_right = head_left

        # Build state: endpose(7+7) + joint(6+6) + gripper(1+1) = 28
        state = np.concatenate([
            np.asarray(endpose['left_endpose'][t], dtype=np.float32),
            np.asarray(endpose['right_endpose'][t], dtype=np.float32),
            np.asarray(joints['left_arm'][t], dtype=np.float32),
            np.asarray(joints['right_arm'][t], dtype=np.float32),
            np.array([joints['left_gripper'][t]], dtype=np.float32),
            np.array([joints['right_gripper'][t]], dtype=np.float32),
        ])

        frames.append({
            'head_left': head_left,
            'head_right': head_right,
            'state': state,
        })

    f.close()
    return frames


def register_attention_hooks(action_model):
    """Patch _sa_block on all transformer layers to capture attention weights."""
    captured = []

    for layer_idx, layer in enumerate(action_model.transformer.layers):
        _lr = layer
        _lid = layer_idx

        def make_patch(lr, lid):
            def patched_sa(x, attn_mask, key_padding_mask, is_causal=False):
                attn_out, attn_weights = lr.self_attn(
                    x, x, x, attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=True, average_attn_weights=False, is_causal=is_causal)
                captured.append({'layer': lid, 'weights': attn_weights.detach().cpu()})
                return lr.dropout1(attn_out)
            return patched_sa

        _lr._sa_block = make_patch(_lr, _lid)

    return captured


def analyze_attention_for_step(captured, action_model, step_idx):
    """Parse captured attention weights for one timestep into per-modality stats."""
    N_act = action_model.N_act
    N_state_tokens = action_model.N_state_tokens
    N_stereo = action_model.N_stereo
    H_action = action_model.H_action

    step_stats = {'cmd': [], 'vis': [], 'state': [], 'stereo': [], 'query': []}

    for cap in captured:
        weights = cap['weights']  # [B, nhead, seq, seq]
        _, nhead, seq_len, _ = weights.shape
        N_v = seq_len - N_act - N_state_tokens - N_stereo - H_action

        attn = weights[0].mean(dim=0)  # [seq, seq] — avg over heads, batch=0
        query_start = seq_len - H_action
        q_attn = attn[query_start:, :]  # [H_action, seq]

        a_cmd    = q_attn[:, :N_act].mean().item()
        a_vis    = q_attn[:, N_act:N_act+N_v].mean().item()
        a_state  = q_attn[:, N_act+N_v:N_act+N_v+N_state_tokens].mean().item()
        a_stereo = q_attn[:, N_act+N_v+N_state_tokens:N_act+N_v+N_state_tokens+N_stereo].mean().item()
        a_query  = q_attn[:, query_start:].mean().item()

        total = a_cmd + a_vis + a_state + a_stereo + a_query
        for k, v in [('cmd', a_cmd), ('vis', a_vis), ('state', a_state),
                      ('stereo', a_stereo), ('query', a_query)]:
            step_stats[k].append(v / total)

    return step_stats


def run_episode_analysis(ckpt_path, episode_path, max_steps=16, device="cuda:0"):
    """Load model, run real episode, capture attention per timestep."""

    print(f"Loading model from {ckpt_path} ...")
    model = baseframework.from_pretrained(ckpt_path)
    model = model.to(device).eval()
    model.infer_mode = "full"
    model.vlm_stride = 0
    model.max_history = max_steps

    # Register hooks on action head
    captured = register_attention_hooks(model.action_model)
    n_layers = len(model.action_model.transformer.layers)
    print(f"  Hooks registered on {n_layers} transformer layers")

    # Load real episode data
    frames = load_episode_data(episode_path, max_steps=max_steps)
    print(f"  Loaded {len(frames)} frames from episode")

    N_act = model.action_model.N_act
    N_stereo = model.action_model.N_stereo
    H_action = model.action_model.H_action
    print(f"  Action head: N_act={N_act}, N_stereo={N_stereo}, H_action={H_action}")

    # Per-timestep results
    timeline = []  # list of per-step per-modality averages

    print(f"\nRunning inference for {len(frames)} steps ...")
    for step_idx, frame in enumerate(frames):
        captured.clear()

        # Build example dict (matches predict_action_full expectations)
        example = {
            "image": [frame['head_left']],
            "lang": "click the bell",
            "state": frame['state'],
            "stereo_left": [frame['head_right']],   # stereo uses right cam
            "stereo_right": [frame['head_right']],
        }

        try:
            with torch.no_grad():
                result = model.predict_action([example])
        except Exception as e:
            print(f"  Step {step_idx}: inference error: {e}")
            continue

        if not captured:
            print(f"  Step {step_idx}: no attention captured (VLM may have used fast path)")
            continue

        step_stats = analyze_attention_for_step(captured, model.action_model, step_idx)
        timeline.append(step_stats)

        # Quick per-step summary
        vis_pct = np.mean(step_stats['vis']) * 100
        cmd_pct = np.mean(step_stats['cmd']) * 100
        print(f"  Step {step_idx:2d}: vis={vis_pct:5.1f}%  cmd={cmd_pct:5.1f}%  "
              f"(layers={len(step_stats['vis'])})")

    return timeline, model


def print_report(timeline, model):
    """Print detailed attention analysis report."""
    n_steps = len(timeline)
    n_layers = model.action_model.transformer.layers

    if n_steps == 0:
        print("ERROR: No attention data captured!")
        return

    # Aggregate across all timesteps and layers
    agg = {'cmd': [], 'vis': [], 'state': [], 'stereo': [], 'query': []}
    for step_stats in timeline:
        for k in agg:
            agg[k].extend(step_stats[k])

    total_samples = len(agg['vis'])
    uniform_vis = 64 / 198 * 100  # 32.3%

    print(f"\n{'='*70}")
    print(f"RESULTS: {n_steps} timesteps × {len(n_layers)} layers = {total_samples} samples")
    print(f"{'='*70}")

    print(f"\n  Overall attention distribution (real episode data):")
    print(f"  {'Modality':<10s} {'Mean':>8s}  {'Std':>8s}  {'Min':>8s}  {'Max':>8s}  {'vs Uniform':>10s}")
    print(f"  {'-'*60}")

    uniform_ref = {'cmd': 16/198*100, 'vis': 64/198*100, 'state': 4/198*100,
                   'stereo': 64/198*100, 'query': 50/198*100}

    for k in ['cmd', 'vis', 'state', 'stereo', 'query']:
        vals = agg[k]
        mean_v = np.mean(vals) * 100
        std_v = np.std(vals) * 100
        min_v = np.min(vals) * 100
        max_v = np.max(vals) * 100
        bar = '█' * max(1, int(mean_v))
        delta = mean_v - uniform_ref[k]
        print(f"  {k:<10s} {mean_v:7.2f}% {std_v:7.2f}% {min_v:7.2f}% {max_v:7.2f}%  {delta:+8.2f}%  {bar}")

    # Per-layer breakdown
    print(f"\n  Per-layer breakdown (avg over all timesteps):")
    header = f"  {'Layer':<8s}"
    for k in ['cmd', 'vis', 'state', 'stereo', 'query']:
        header += f" {k:>8s}"
    print(header)
    print(f"  {'-'*50}")

    for lyr in range(len(n_layers)):
        lyr_stats = {k: [] for k in agg}
        for step_stats in timeline:
            if lyr < len(step_stats['vis']):  # each step contributes 1 value per layer per modality
                for k in agg:
                    vals = step_stats[k]
                    if lyr < len(vals):
                        lyr_stats[k].append(vals[lyr])
        parts = [f"  {lyr:<8d}"]
        for k in ['cmd', 'vis', 'state', 'stereo', 'query']:
            if lyr_stats[k]:
                parts.append(f" {np.mean(lyr_stats[k])*100:7.2f}%")
            else:
                parts.append(f" {'-':>7s}")
        print(''.join(parts))

    # Time evolution
    print(f"\n  Attention over time (avg across layers):")
    header = f"  {'Step':<6s}"
    for k in ['cmd', 'vis', 'state', 'stereo']:
        header += f" {k:>8s}"
    print(header)
    print(f"  {'-'*42}")

    for t, step_stats in enumerate(timeline):
        parts = [f"  {t:<6d}"]
        for k in ['cmd', 'vis', 'state', 'stereo']:
            parts.append(f" {np.mean(step_stats[k])*100:7.2f}%")
        # Mark when VLM refresh happens (every step when vlm_stride=0)
        print(''.join(parts))

    # Summary
    vis_mean = np.mean(agg['vis']) * 100
    cmd_mean = np.mean(agg['cmd']) * 100

    print(f"\n{'='*70}")
    print(f"Key metrics:")
    print(f"  Vision:  {vis_mean:.1f}%  (uniform baseline: {uniform_vis:.1f}%)")
    print(f"  cmd:     {cmd_mean:.1f}%")
    print(f"  Non-vis: {np.mean(agg['cmd']+agg['state']+agg['stereo'])*100:.1f}%")

    if vis_mean < uniform_vis:
        print(f"\n✅ Vision below uniform baseline — model does NOT over-rely on vision")
    elif vis_mean < uniform_vis * 1.3:
        print(f"\n⚠️  Vision slightly above uniform baseline — moderate reliance")
    else:
        print(f"\n❌ Vision dominates attention")


def main():
    parser = argparse.ArgumentParser(description="Analyze action head attention on real episode data")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to checkpoint.pt (default: latest vit-mask trained)")
    parser.add_argument("--episode", type=int, default=0,
                        help="Episode index to analyze (default: 0)")
    parser.add_argument("--max_steps", type=int, default=16,
                        help="Max timesteps to process (default: 16)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare pre vs post masking checkpoints")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # Default checkpoints
    ckpt_base = "/mnt/workspace/yama/starVLA/results/Checkpoints_QwenZone"
    pre_ckpt = f"{ckpt_base}/qwenzone_click_bell_v2phase2/checkpoints/steps_50000_pytorch_model.pt"
    post_ckpt = f"{ckpt_base}/qwenzone_click_bell_v2phase2_t8/checkpoints/steps_50000_pytorch_model.pt"

    episode_path = f"/mnt/workspace/yama/RoboTwin/data/click_bell/demo_randomized/data/episode{args.episode}.hdf5"

    if not os.path.exists(episode_path):
        print(f"ERROR: Episode not found: {episode_path}")
        sys.exit(1)

    if args.compare:
        # Compare both checkpoints on the same episode
        for ckpt, label in [(pre_ckpt, "PRE masking (no vit mask)"), (post_ckpt, "POST masking (vit+cmd mask)")]:
            if not os.path.exists(ckpt):
                print(f"ERROR: Checkpoint not found: {ckpt}")
                continue
            print(f"\n{'#'*70}")
            print(f"# {label}")
            print(f"{'#'*70}")
            timeline, model = run_episode_analysis(ckpt, episode_path,
                                                   max_steps=args.max_steps,
                                                   device=args.device)
            print_report(timeline, model)
            # Free GPU memory between models
            del model
            torch.cuda.empty_cache()
    else:
        ckpt = args.ckpt or post_ckpt
        if not os.path.exists(ckpt):
            print(f"ERROR: Checkpoint not found: {ckpt}")
            sys.exit(1)

        timeline, model = run_episode_analysis(ckpt, episode_path,
                                               max_steps=args.max_steps,
                                               device=args.device)
        print_report(timeline, model)

    print("\nDone!")


if __name__ == "__main__":
    main()
