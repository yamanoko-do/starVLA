"""Test client for starVLA policy server (no hardware required).

Sends synthetic observations to the server and prints the predicted actions.
Usage:
  python deployment/test_client.py --host 127.0.0.1 --port 10093
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from PIL import Image

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


def build_argparser():
    p = argparse.ArgumentParser(description="starVLA policy server test client")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=13467)
    p.add_argument("--lang", default="fold the cloth", help="task instruction")
    p.add_argument("--cameras", type=int, default=1, help="number of cameras")
    p.add_argument("--frames", type=int, default=8, help="frames per camera (1=image, >1=video)")
    p.add_argument("--state_history", type=int, default=50, help="state history length")
    p.add_argument("--action_dim", type=int, default=14, help="action dimension")
    p.add_argument("--img_size", type=int, default=224, help="image resolution")
    return p


def main():
    args = build_argparser().parse_args()

    # 1. Connect
    print(f"Connecting to {args.host}:{args.port} ...")
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    meta = client.get_server_metadata()
    print(f"Server: chunk={meta['action_chunk_size']}, "
          f"norm_key={meta.get('default_unnorm_key')}, "
          f"action_keys={meta.get('action_keys')}, "
          f"state_keys={meta.get('state_keys')}")

    # 2. Build test observation
    H, W = args.img_size, args.img_size
    images = []
    for _ in range(args.cameras):
        if args.frames == 1:
            # single image per camera
            images.append(np.random.randint(0, 255, (H, W, 3), dtype=np.uint8))
        else:
            # video: list of frames per camera
            frames = [np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
                      for _ in range(args.frames)]
            images.append(frames)

    state = np.random.randn(args.state_history, args.action_dim).astype(np.float32)

    payload = {
        "examples": [{
            "image": images,
            "lang": args.lang,
            "state": state,
        }],
    }

    # 3. Send and receive
    print(f"\nSending: {args.cameras} cam(s) × {args.frames} frame(s), "
          f"state={state.shape}, lang='{args.lang}'")
    result = client.predict_action(payload)
    # Response format: {"status": "ok", "data": {"actions": np.ndarray}, ...}
    actions = np.array(result["data"]["actions"])

    print(f"Actions: {actions.shape}  (batch={actions.shape[0]}, "
          f"chunk={actions.shape[1]}, dim={actions.shape[2]})")
    print(f"  Step 0: {np.array2string(actions[0, 0], precision=3, suppress_small=True)}")
    print(f"  Step 1: {np.array2string(actions[0, 1], precision=3, suppress_small=True)}")
    print(f"  ...")
    print(f"  Step {actions.shape[1]-1}: {np.array2string(actions[0, -1], precision=3, suppress_small=True)}")

    # 4. Timing
    import time
    N = 10
    t0 = time.perf_counter()
    for _ in range(N):
        client.predict_action(payload)
    elapsed = time.perf_counter() - t0
    print(f"\n{N} requests in {elapsed:.1f}s → {elapsed/N*1000:.0f} ms/req, {N/elapsed:.1f} Hz")

    client.close()
    print("Done.")


if __name__ == "__main__":
    main()
