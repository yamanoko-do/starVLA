"""Standalone test client for starVLA policy server.

Zero starVLA imports — only needs websockets + msgpack + numpy.
Usage:
  python test_client.py --host 8.145.57.160 --port 13467
"""

import argparse
import functools
import msgpack
import numpy as np
import websockets.sync.client


# ---- tiny msgpack-numpy glue (no dependency on starVLA code) ----
def _pack_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(),
                b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj

def _unpack_ndarray(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj

_packb = functools.partial(msgpack.packb, default=_pack_ndarray)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_ndarray)


class PolicyClient:
    """Minimal websocket client for starVLA policy server."""

    def __init__(self, host="127.0.0.1", port=10093, timeout=10):
        uri = f"ws://{host}:{port}"
        self._ws = websockets.sync.client.connect(
            uri, open_timeout=timeout, close_timeout=3,
            ping_interval=None, ping_timeout=60)
        self.meta = _unpackb(self._ws.recv())

    def predict(self, payload: dict) -> dict:
        self._ws.send(_packb(payload))
        return _unpackb(self._ws.recv())

    def close(self):
        self._ws.close()


# ---- main ----
def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=13467)
    p.add_argument("--lang", default="fold the cloth")
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--state_history", type=int, default=50)
    p.add_argument("--action_dim", type=int, default=14)
    p.add_argument("--img_size", type=int, default=224)
    return p


def main():
    args = build_argparser().parse_args()

    client = PolicyClient(host=args.host, port=args.port)
    print(f"Server: chunk={client.meta['action_chunk_size']}, "
          f"norm_key={client.meta.get('default_unnorm_key')}")

    H, W = args.img_size, args.img_size
    frames = [np.random.randint(0, 255, (H, W, 3), dtype=np.uint8) for _ in range(args.frames)]
    state = np.random.randn(args.state_history, args.action_dim).astype(np.float32)

    payload = {"examples": [{"image": [frames], "lang": args.lang, "state": state}]}

    print(f"Sending: 1 cam × {args.frames} frames, state={state.shape}")
    result = client.predict(payload)
    actions = np.array(result["data"]["actions"])
    print(f"Actions: {actions.shape}")
    print(f"  Step 0: {np.array2string(actions[0, 0], precision=3, suppress_small=True)}")
    print(f"  Step {actions.shape[1]-1}: {np.array2string(actions[0, -1], precision=3, suppress_small=True)}")

    # Timing
    import time
    N = 10; t0 = time.perf_counter()
    for _ in range(N):
        client.predict(payload)
    e = time.perf_counter() - t0
    print(f"{N} req in {e:.1f}s → {e/N*1000:.0f} ms/req, {N/e:.1f} Hz")

    client.close()
    print("Done.")


if __name__ == "__main__":
    main()
