"""Real deployment inference client for starVLA policy server.

No starVLA imports — only websockets + msgpack + numpy (+ cv2 for cameras).
Wire up `capture_observation()` to your hardware, or run with --mock for testing.

Usage:
  python inference_client.py --host 8.145.57.160 --port 13467           # real
  python inference_client.py --host 127.0.0.1 --port 13467 --mock       # test with random data
"""

import argparse
import functools
import msgpack
import numpy as np
import time
import websockets.sync.client


# =========================================================================
#  msgpack-numpy glue (no starVLA dependency)
# =========================================================================
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


# =========================================================================
#  Policy client (WebSocket → starVLA server)
# =========================================================================
class PolicyClient:
    def __init__(self, host="127.0.0.1", port=13467, timeout=10):
        uri = f"ws://{host}:{port}"
        self._ws = websockets.sync.client.connect(
            uri, open_timeout=timeout, close_timeout=3,
            ping_interval=None, ping_timeout=60)
        self.meta = _unpackb(self._ws.recv())
        print(f"[server] chunk={self.meta['action_chunk_size']}, "
              f"norm={self.meta.get('default_unnorm_key')}, "
              f"action_keys={self.meta.get('action_keys')}")

        self._chunk_size = int(self.meta["action_chunk_size"])
        self._action_cache = None
        self._cache_step = 0

    def _predict(self, payload):
        self._ws.send(_packb(payload))
        return _unpackb(self._ws.recv())

    def get_action(self, images, lang, state):
        """Return a single action step. Handles chunk caching internally."""
        # Need new prediction if cache empty or exhausted
        if self._action_cache is None or self._cache_step >= self._chunk_size:
            payload = {"examples": [{"image": images, "lang": lang, "state": state}]}
            result = self._predict(payload)
            self._action_cache = np.array(result["data"]["actions"])[0]  # [chunk, dim]
            self._cache_step = 0

        action = self._action_cache[self._cache_step]
        self._cache_step += 1
        return action

    def close(self):
        self._ws.close()


# =========================================================================
#  Hardware hooks — replace these with your actual robot interface
# =========================================================================
CAMERAS = ["cam_high"]           # camera names (matches training)
IMG_SIZE = (224, 224)            # must match training
STATE_DIM = 14                   # ALOHA dual-arm joints
STATE_HISTORY = 50
ACTION_DIM = 14
IMAGE_FRAMES = 8                 # Fibonacci: 8 frames per camera


def capture_cameras() -> list:
    """
    Return a list of camera images, each camera as a list of frames.
    Format: [[cam0_frame0, ..., cam0_frame7], [cam1_frame0, ...], ...]

    Replace this with cv2.VideoCapture or your camera SDK.
    """
    try:
        import cv2
        frames_per_cam = []
        for cam_name in CAMERAS:
            cap = cv2.VideoCapture(cam_name)  # or camera index / device path
            cam_frames = []
            for _ in range(IMAGE_FRAMES):
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, IMG_SIZE)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cam_frames.append(frame)
            cap.release()
            frames_per_cam.append(cam_frames)
        return frames_per_cam
    except ImportError:
        raise NotImplementedError(
            "Install opencv-python and wire up your cameras in capture_cameras(). "
            "Run with --mock for testing.")


def capture_state() -> np.ndarray:
    """
    Return current joint state history: [STATE_HISTORY, STATE_DIM].

    Replace this with your robot SDK (e.g. dynamixel read, ROS subscriber).
    """
    raise NotImplementedError(
        "Wire up your robot state reader in capture_state(). "
        "Run with --mock for testing.")


# =========================================================================
#  Mock helpers (for testing without hardware)
# =========================================================================
def mock_cameras():
    return [[np.random.randint(0, 255, (*IMG_SIZE, 3), dtype=np.uint8)
             for _ in range(IMAGE_FRAMES)] for _ in CAMERAS]

def mock_state():
    return np.random.randn(STATE_HISTORY, STATE_DIM).astype(np.float32)


# =========================================================================
#  Main loop
# =========================================================================
def main():
    p = argparse.ArgumentParser(description="starVLA deployment inference client")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=13467)
    p.add_argument("--lang", default="fold the cloth")
    p.add_argument("--mock", action="store_true",
                   help="use random data instead of real hardware")
    p.add_argument("--hz", type=int, default=10,
                   help="target control frequency")
    p.add_argument("--steps", type=int, default=100,
                   help="number of steps to run (mock mode)")
    args = p.parse_args()

    client = PolicyClient(host=args.host, port=args.port)

    print(f"[mode] {'mock' if args.mock else 'REAL'}")
    print(f"[ctrl] {args.hz} Hz, action_chunk={client._chunk_size}")
    print(f"[task] {args.lang}")

    for step in range(args.steps):
        t0 = time.perf_counter()

        # 1. Observe
        if args.mock:
            images = mock_cameras()
            state = mock_state()
        else:
            images = capture_cameras()
            state = capture_state()

        # 2. Infer + cache
        action = client.get_action(images, args.lang, state)

        # 3. Act (replace with robot.execute(action))
        if step % 10 == 0:
            print(f"  step {step:4d}: {np.array2string(action, precision=3, suppress_small=True)}")

        # 4. Maintain control rate
        elapsed = time.perf_counter() - t0
        sleep_time = 1.0 / args.hz - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    client.close()
    print("Done.")


if __name__ == "__main__":
    main()
