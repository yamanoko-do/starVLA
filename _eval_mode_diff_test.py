"""Definitive test: does predict_action_rnn actually differ from predict_action_full?
Loads the TRAINED checkpoint (same path the eval server uses) and runs 3 sequential
predict_action calls in each mode (history/m_state grows across calls).

Expectation:
  - call 1 (step 0): IDENTICAL (both single-step [M_init, V_0, L, A_0, M_0]; rnn m_state = M_init embed).
  - call 2+ : DIVERGE (full grows history -> multi-step seq; rnn injects advanced m_state).

If they're identical across ALL calls -> rnn dispatch is broken. If they diverge at call 2+ ->
rnn works, and identical eval episodes are just short (<1 chunk = 50 steps).
"""
import os, sys
import numpy as np
from PIL import Image
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, "/mnt/workspace/yama/starVLA")
os.chdir("/mnt/workspace/yama/starVLA")

from starVLA.model.framework.base_framework import baseframework

CKPT = "results/Checkpoints_QwenZone/qwenzone_click_bell_v2phase2/checkpoints/steps_50000_pytorch_model.pt"
print("loading trained framework ...", flush=True)
fw = baseframework.from_pretrained(CKPT).cuda().eval()
print(f"loaded. infer_mode default = {fw.infer_mode}", flush=True)

img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

# FIXED observation sequence — identical inputs to both modes so any action
# difference is purely due to full vs rnn (history growth vs m_state recurrence).
rng = np.random.RandomState(0)
FIXED_OBS = []
for _ in range(3):
    g = Image.fromarray(rng.randint(0, 255, (224, 224, 3)).astype(np.uint8))
    s = rng.uniform(-1, 1, (28,)).astype(np.float32)
    FIXED_OBS.append({"image": [g, g, g], "lang": "click the bell", "state": s})

def run_mode(mode, n=3):
    fw.infer_mode = mode
    fw.reset_history()
    outs = []
    for i in range(n):
        o = fw.predict_action([FIXED_OBS[i]])
        outs.append(o["normalized_actions"].copy())
    return outs

print("\nrunning FULL mode (3 steps) ...", flush=True)
full_outs = run_mode("full", n=3)
print("running RNN mode (3 steps) ...", flush=True)
rnn_outs = run_mode("rnn", n=3)

print("\n=== per-call mean |full - rnn| (should be ~0 at call1, >0 at call2+) ===")
for i in range(3):
    d = np.abs(full_outs[i] - rnn_outs[i]).mean()
    print(f"  call {i+1}: mean|Δaction| = {d:.6e}   full[0,0,:3]={full_outs[i][0,0,:3]}  rnn[0,0,:3]={rnn_outs[i][0,0,:3]}")

# also report full-mode internal: does full's call2 differ from call1? (sanity that history grows)
print("\n=== sanity: full call2 vs call1 (should differ, history grew) ===")
d_ff = np.abs(full_outs[0] - full_outs[1]).mean()
print(f"  mean|full_call1 - full_call2| = {d_ff:.6e}")
print("\n=== sanity: rnn call2 vs call1 (should differ, m_state advanced) ===")
d_rr = np.abs(rnn_outs[0] - rnn_outs[1]).mean()
print(f"  mean|rnn_call1 - rnn_call2| = {d_rr:.6e}")
