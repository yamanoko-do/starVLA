# Copyright 2025 starVLA community. All rights reserved.

"""HDF5 RoboTwin dataset for starVLA training pipeline.

Reads raw RoboTwin simulation output (HDF5 episodes with JPEG-encoded images)
and returns samples in starVLA's expected format, including stereo pairs for
the QwenZone action head.
"""

import io
import json
import random
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class HDF5RobotwinDataset(Dataset):
    """PyTorch Dataset over RoboTwin HDF5 episodes.

    Expected directory layout::

        {root}/
          demo_randomized/
            data/episode0.hdf5 ... episodeN.hdf5
            instructions/episode0.json ... episodeN.json

    Each HDF5 contains 112 steps of ``joint_action/vector`` (14-dim) and
    JPEG-encoded camera images.  Stereo pairs are taken from
    ``head_camera_left`` / ``head_camera_right``; VLM sees left only.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        T: int = 4,
        H: int = 50,
        img_size: int = 224,
        stereo_size: tuple[int, int] = (512, 256),
        action_min: np.ndarray | None = None,
        action_max: np.ndarray | None = None,
        state_min: np.ndarray | None = None,
        state_max: np.ndarray | None = None,
        shuffle: bool = True,
    ):
        super().__init__()
        self.root = Path(dataset_path)
        self.T = T
        self.H = H
        self.img_size = img_size
        self.stereo_size = stereo_size
        self.shuffle = shuffle

        hdf5_dir = self.root / "demo_randomized" / "data"
        self.episodes = sorted(hdf5_dir.glob("episode*.hdf5"))
        if not self.episodes:
            raise FileNotFoundError(f"No episode*.hdf5 found in {hdf5_dir}")

        # Index all (ep_idx, t0) windows
        self.samples = []
        for ep_idx, ep_path in enumerate(self.episodes):
            with h5py.File(ep_path, "r") as f:
                total = f["joint_action/vector"].shape[0]
            # stride ensures non-overlapping windows (e.g., T=4 → stride 4)
            stride = max(1, T)
            for t0 in range(0, total - T - H + 1, stride):
                self.samples.append((ep_idx, t0))

        # Action/state normalization ranges (Aloha agilex joints + grippers)
        # Default to reasonable ranges; override with data statistics for production.
        if action_min is None:
            action_min = np.array(
                [-3.0] * 6 + [-3.0] * 6 + [0.0, 0.0], dtype=np.float32
            )
        if action_max is None:
            action_max = np.array(
                [3.0] * 6 + [3.0] * 6 + [1.0, 1.0], dtype=np.float32
            )
        if state_min is None:
            state_min = action_min.copy()
        if state_max is None:
            state_max = action_max.copy()

        self.action_min = action_min
        self.action_max = action_max
        self.state_min = state_min
        self.state_max = state_max
        self.tag = "robotwin_qwenzone"     # for LeRobotMixtureDataset

    # --- compatibility with LeRobotMixtureDataset -----------------------
    # These attributes are expected by LeRobotMixtureDataset.__init__ /
    # update_metadata / transforms factories.
    tag: str = ""
    image_size: tuple = (224, 224)

    @property
    def trajectory_ids(self):
        return np.array([ep for ep, _ in self.samples])

    @property
    def trajectory_lengths(self):
        return np.array([self.T + self.H - 1] * len(self.samples))

    @property
    def stats(self):
        return {"action": {"min": self.action_min, "max": self.action_max}}

    @property
    def metadata(self):
        return self.stats

    def save_dataset_statistics(self, path):
        pass

    def set_transforms_metadata(self, metadata):
        pass

    # -------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def _normalize(self, x, lo, hi):
        """min_max to [-1, 1] for joints; threshold for grippers."""
        out = np.empty_like(x, dtype=np.float32)
        # joints: min_max
        out[:, :12] = 2.0 * (x[:, :12] - lo[:12]) / (hi[:12] - lo[:12] + 1e-8) - 1.0
        # grippers: binary threshold at 0.5
        out[:, 12:] = (x[:, 12:] > 0.5).astype(np.float32)
        return out

    @staticmethod
    def _decode(rgb_bytes):
        return Image.open(io.BytesIO(rgb_bytes)).convert("RGB")

    def __getitem__(self, idx):
        ep_idx, t0 = self.samples[idx]

        with h5py.File(self.episodes[ep_idx], "r") as f:
            # --- VLM image (head_camera_left only, 1 cam × T frames) -------
            vlm_frames = []
            for t in range(self.T):
                img = self._decode(f["observation/head_camera_left/rgb"][t0 + t])
                vlm_frames.append(img.resize((self.img_size, self.img_size)))
            vlm_images = [vlm_frames]  # [1 cam][T] — VLM receives single view

            # --- stereo images (left + right) ------------------------------
            stereo_left = []
            stereo_right = []
            for t in range(self.T):
                sl = self._decode(f["observation/head_camera_left/rgb"][t0 + t])
                sr = self._decode(f["observation/head_camera_right/rgb"][t0 + t])
                stereo_left.append(sl.resize(self.stereo_size))
                stereo_right.append(sr.resize(self.stereo_size))

            # --- action: vector layout is [L6, Lg, R6, Rg] → reorder to starVLA [L6, R6, Lg, Rg] → normalize
            raw_vec = f["joint_action/vector"][t0 : t0 + self.T + self.H - 1].astype(np.float32)
            raw_action = np.concatenate(
                [raw_vec[:, 0:6], raw_vec[:, 7:13], raw_vec[:, 6:7], raw_vec[:, 13:14]], axis=1
            )  # [T+H-1, 14] in [L6, R6, Lg, Rg]
            action = self._normalize(raw_action, self.action_min, self.action_max)

            # --- state: endpose(7+7) + joint(6+6) + gripper(1+1) = 28, RAW (un-normalized) ---
            # Action-head LayerNorm handles scale; keeping raw avoids train/deploy mismatch.
            left_ep = f["endpose/left_endpose"][t0 : t0 + self.T].astype(np.float32)      # [T,7] pos+quat
            right_ep = f["endpose/right_endpose"][t0 : t0 + self.T].astype(np.float32)    # [T,7]
            left_arm = f["joint_action/left_arm"][t0 : t0 + self.T].astype(np.float32)    # [T,6]
            right_arm = f["joint_action/right_arm"][t0 : t0 + self.T].astype(np.float32)  # [T,6]
            left_g = f["endpose/left_gripper"][t0 : t0 + self.T].astype(np.float32)[:, None]
            right_g = f["endpose/right_gripper"][t0 : t0 + self.T].astype(np.float32)[:, None]
            state = np.concatenate(
                [left_ep, right_ep, left_arm, right_arm, left_g, right_g], axis=1
            )  # [T, 28] raw proprioception

        # --- language instruction ------------------------------------------
        inst_path = self.root / "demo_randomized" / "instructions" / f"episode{ep_idx}.json"
        with open(inst_path) as fi:
            inst_data = json.load(fi)
        seen = inst_data.get("seen", [])
        lang = random.choice(seen) if seen else "perform the task"

        return {
            "image": vlm_images,            # [1 cam][T] PIL
            "lang": lang,
            "action": action,               # [T+H-1, 14] normalized
            "state": state,                 # [T, 14] normalized
            "stereo_left": stereo_left,      # [T] PIL
            "stereo_right": stereo_right,    # [T] PIL
            "robot_tag": self.tag,
            "fps": 30.0,
        }
