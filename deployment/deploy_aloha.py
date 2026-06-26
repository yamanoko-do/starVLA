"""ALOHA deployment client for starVLA QwenZone policy server.

Adapted from test.py — keeps ROS infrastructure (RosOperator, sync, topics),
replaces OpenPI client with starVLA's WebSocket + msgpack protocol.

Key differences from test.py:
  - Multi-frame video input (8 Fibonacci frames per camera)
  - Rolling frame + state history buffers
  - Chunk-cache scheduling (predict 8 steps, execute one at a time)

Dependencies: websockets, msgpack, numpy, cv2, rclpy, sensor_msgs, cv_bridge, message_filters
"""

import argparse
import collections
import functools
import queue
import threading
import time

import cv2
import einops
import msgpack
import numpy as np
import rclpy
import websockets.sync.client
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header


# =========================================================================
#  Config — match your trained QwenZone checkpoint
# =========================================================================
CAMERA_NAMES = ["cam_high"]          # match training config
IMG_SIZE = 224                        # must match training
ACTION_DIM = 14                       # ALOHA dual-arm
STATE_HISTORY = 50                    # match state_history_len in config
IMAGE_FRAMES = 8                      # match MAX_VIDEO_FRAMES in DataConfig
ACTION_CHUNK = 8                      # match action_horizon
CONTROL_HZ = 30


# =========================================================================
#  msgpack-numpy glue
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
#  starVLA policy client (WebSocket)
# =========================================================================
class StarVLAClient:
    def __init__(self, host="127.0.0.1", port=13467, timeout=10):
        uri = f"ws://{host}:{port}"
        self._ws = websockets.sync.client.connect(
            uri, open_timeout=timeout, close_timeout=3,
            ping_interval=None, ping_timeout=60)
        self.meta = _unpackb(self._ws.recv())
        print(f"[starVLA] chunk={self.meta['action_chunk_size']}, "
              f"norm={self.meta.get('default_unnorm_key')}")

        self._action_cache = collections.deque()

    def predict(self, images, lang, state):
        """Send observation, fill cache with predicted actions."""
        payload = {"examples": [{"image": images, "lang": lang, "state": state}]}
        self._ws.send(_packb(payload))
        result = _unpackb(self._ws.recv())
        self._action_cache = collections.deque(
            np.array(result["data"]["actions"])[0]   # [chunk, dim]
        )

    def pop_action(self):
        """Return next cached action, or None if cache is empty."""
        if not self._action_cache:
            return None
        return self._action_cache.popleft()

    @property
    def cache_empty(self):
        return len(self._action_cache) == 0

    def close(self):
        self._ws.close()


# =========================================================================
#  Frame / state history buffers
# =========================================================================
class ObservationBuffer:
    """Ring buffers for multi-frame image history and joint state history."""
    def __init__(self):
        self.frame_buffers = {cam: collections.deque(maxlen=IMAGE_FRAMES)
                              for cam in CAMERA_NAMES}
        self.state_buffer = collections.deque(maxlen=STATE_HISTORY)
        # Seed with zeros so we can send observations immediately
        dummy_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        for cam in CAMERA_NAMES:
            for _ in range(IMAGE_FRAMES):
                self.frame_buffers[cam].append(dummy_img.copy())
        dummy_state = np.zeros(ACTION_DIM, dtype=np.float32)
        for _ in range(STATE_HISTORY):
            self.state_buffer.append(dummy_state.copy())

    def push(self, cam_images: dict, state: np.ndarray):
        """cam_images: {cam_name: np.ndarray (H,W,3) uint8}"""
        for cam, img in cam_images.items():
            if cam in self.frame_buffers:
                self.frame_buffers[cam].append(img)
        self.state_buffer.append(state.astype(np.float32))

    def get_images(self):
        """Return [[cam0_frames], [cam1_frames], ...] for starVLA format."""
        return [list(self.frame_buffers[cam]) for cam in CAMERA_NAMES]

    def get_state(self):
        """Return [STATE_HISTORY, ACTION_DIM] float32."""
        return np.array(list(self.state_buffer), dtype=np.float32)


# =========================================================================
#  ROS operator (from test.py, adapted)
# =========================================================================
class RosOperator:
    def __init__(self, args):
        self.bridge = CvBridge()
        self.sync_queue = queue.Queue(maxsize=10)
        self.sync_slop = 0.04
        self.args = args
        self._init_ros()

    def _init_ros(self):
        rclpy.init()
        self.node = rclpy.create_node('starvla_deploy')
        self.rate = self.node.create_rate(self.args.publish_rate)
        self.left_pub = self.node.create_publisher(JointState, self.args.master_arm_left_topic, 10)
        self.right_pub = self.node.create_publisher(JointState, self.args.master_arm_right_topic, 10)

        self.thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.thread.start()

        sub_front = Subscriber(self.node, Image, self.args.img_front_topic)
        sub_left  = Subscriber(self.node, Image, self.args.img_left_topic)
        sub_right = Subscriber(self.node, Image, self.args.img_right_topic)
        sub_puppet_l = Subscriber(self.node, JointState, self.args.puppet_arm_left_topic)
        sub_puppet_r = Subscriber(self.node, JointState, self.args.puppet_arm_right_topic)

        self.ts = ApproximateTimeSynchronizer(
            [sub_left, sub_right, sub_front, sub_puppet_l, sub_puppet_r],
            queue_size=1000, slop=self.sync_slop)
        self.ts.registerCallback(self._sync_callback)

    def _sync_callback(self, img_left, img_right, img_front, puppet_l, puppet_r):
        try:
            front = self.bridge.imgmsg_to_cv2(img_front, 'passthrough')
            left  = self.bridge.imgmsg_to_cv2(img_left,  'passthrough')
            right = self.bridge.imgmsg_to_cv2(img_right, 'passthrough')
        except Exception:
            return
        frame = (front, left, right, puppet_l, puppet_r)
        if self.sync_queue.full():
            try:
                self.sync_queue.get_nowait()
            except queue.Empty:
                pass
        self.sync_queue.put(frame)

    def sync_get_frame(self, timeout=0.2):
        try:
            return self.sync_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def publish_arms(self, left_action, right_action):
        """Direct publish (no interpolation)."""
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
        msg.position = [float(x) for x in left_action]
        self.left_pub.publish(msg)
        msg.position = [float(x) for x in right_action]
        self.right_pub.publish(msg)

    def publish_arms_continuous(self, left_target, right_target, arm_steps_length=None):
        """Smooth interpolation from current position to target (from aa.py).

        Reads current joint positions, then linearly steps toward target at
        `arm_steps_length` increments per joint until all joints reach target.
        """
        if arm_steps_length is None:
            arm_steps_length = [0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.4]

        # Get current arm positions
        left_cur = None
        right_cur = None
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.005)
            if not self.sync_queue.empty():
                _, _, _, puppet_l, puppet_r = self.sync_queue.get(timeout=0.02)
                left_cur  = [float(x) for x in puppet_l.position]
                right_cur = [float(x) for x in puppet_r.position]
            if left_cur is not None and right_cur is not None:
                break
            self.rate.sleep()

        left_sign = [1 if left_target[i] - left_cur[i] > 0 else -1 for i in range(7)]
        right_sign = [1 if right_target[i] - right_cur[i] > 0 else -1 for i in range(7)]

        step = 0
        moving = True
        while moving and rclpy.ok():
            moving = False
            left_diff = [abs(left_target[i] - left_cur[i]) for i in range(7)]
            right_diff = [abs(right_target[i] - right_cur[i]) for i in range(7)]

            for i in range(7):
                if left_diff[i] < arm_steps_length[i]:
                    left_cur[i] = left_target[i]
                else:
                    left_cur[i] += left_sign[i] * arm_steps_length[i]
                    moving = True
                if right_diff[i] < arm_steps_length[i]:
                    right_cur[i] = right_target[i]
                else:
                    right_cur[i] += right_sign[i] * arm_steps_length[i]
                    moving = True

            msg = JointState()
            msg.header = Header()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
            msg.position = [float(v) for v in left_cur]
            self.left_pub.publish(msg)
            msg.position = [float(v) for v in right_cur]
            self.right_pub.publish(msg)

            step += 1
            self.rate.sleep()


# =========================================================================
#  Main loop
# =========================================================================
def get_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=13467)
    p.add_argument("--lang", default="fold the cloth")
    p.add_argument("--episode_len", type=int, default=1500)
    # ROS topics
    p.add_argument("--img_front_topic", default='/camera_01/color/image_raw')
    p.add_argument("--img_left_topic",  default='/camera_03/color/image_raw')
    p.add_argument("--img_right_topic", default='/camera_02/color/image_raw')
    p.add_argument("--puppet_arm_left_topic",  default='/joint_states_left')
    p.add_argument("--puppet_arm_right_topic", default='/joint_states_right')
    p.add_argument("--master_arm_left_topic",  default='/joint_ctrl_cmd_left')
    p.add_argument("--master_arm_right_topic", default='/joint_ctrl_cmd_right')
    p.add_argument("--publish_rate", type=int, default=30)
    # Mock mode
    p.add_argument("--mock", action="store_true", help="test without ROS hardware")
    return p.parse_args()


def main():
    args = get_arguments()

    # 1. Connect to starVLA server
    client = StarVLAClient(host=args.host, port=args.port)
    obs_buf = ObservationBuffer()

    if not args.mock:
        ros_op = RosOperator(args)

    print(f"[deploy] {args.lang}  |  control={CONTROL_HZ} Hz  |  chunk={ACTION_CHUNK}")
    print(f"[deploy] cameras={CAMERA_NAMES}  frames={IMAGE_FRAMES}  state_history={STATE_HISTORY}")

    # Move to start position (smooth interpolation)
    if not args.mock:
        left0  = [0, 0.32, -0.36, 0, 0.24, 0, 0.07]
        right0 = [0, 0.32, -0.36, 0, 0.24, 0, 0.07]
        print("[deploy] moving to start position...")
        ros_op.publish_arms_continuous(left0, right0)
        print("[deploy] ready.")

    step = 0
    while step < args.episode_len:
        loop_start = time.perf_counter()

        # 2. Observe
        if args.mock:
            # Simulate: random images, random state
            cam_images = {
                cam: np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                for cam in CAMERA_NAMES}
            state = np.random.randn(ACTION_DIM).astype(np.float32)
            time.sleep(1.0 / CONTROL_HZ)
        else:
            result = ros_op.sync_get_frame(0.02)
            if result is None:
                continue
            front_img, left_img, right_img, puppet_l, puppet_r = result
            # Map ROS camera images to config camera names
            cam_images = {}
            # cam_high = front camera
            if "cam_high" in CAMERA_NAMES:
                cam_images["cam_high"] = cv2.resize(front_img, (IMG_SIZE, IMG_SIZE))
            if "cam_left_wrist" in CAMERA_NAMES:
                cam_images["cam_left_wrist"] = cv2.resize(left_img, (IMG_SIZE, IMG_SIZE))
            if "cam_right_wrist" in CAMERA_NAMES:
                cam_images["cam_right_wrist"] = cv2.resize(right_img, (IMG_SIZE, IMG_SIZE))
            state = np.concatenate([
                np.array(puppet_l.position, dtype=np.float32),
                np.array(puppet_r.position, dtype=np.float32),
            ])

        # 3. Push to rolling buffer
        obs_buf.push(cam_images, state)

        # 4. Infer when cache empty
        if client.cache_empty:
            t0 = time.perf_counter()
            client.predict(
                images=obs_buf.get_images(),
                lang=args.lang,
                state=obs_buf.get_state(),
            )
            dt = time.perf_counter() - t0
            if step % 10 == 0:
                print(f"  step {step:4d}: inference {dt*1000:.0f} ms")

        # 5. Execute one action
        action = client.pop_action()
        if action is None:
            continue

        left_action  = [float(x) for x in action[:7]]
        right_action = [float(x) for x in action[7:14]]
        # Gripper thresholding (same as test.py)
        left_action[-1] = 0.0 if left_action[-1] < 0.02 else 0.068
        right_action[-1] = 0.0 if right_action[-1] < 0.02 else 0.068

        if not args.mock:
            ros_op.publish_arms(left_action, right_action)

        # 6. Maintain control rate
        elapsed = time.perf_counter() - loop_start
        sleep_t = 1.0 / CONTROL_HZ - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)
        step += 1

    client.close()
    print(f"[deploy] done — {step} steps")


if __name__ == "__main__":
    main()
