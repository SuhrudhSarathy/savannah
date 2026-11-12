"""Tasks from franka_sim (https://github.com/SuhrudhSarathy/franka_sim), a
MuJoCo Franka Panda benchmark, trained from its LeRobot exports.

Policy input and output share one representation, an end-effector pose plus
a gripper command (8 values):

    [x, y, z, qw, qx, qy, qz, gripper]

    state  (input):  the measured end-effector pose at each obs frame
                     (`observation.ee_pose`) + the gripper command in force
                     at that frame, i.e. the previous action's gripper value
                     (0 = open at the start of an episode, as franka_sim
                     resets it)
    action (output): the commanded end-effector pose + gripper command
                     (`action.ee_pose`), rolled out with franka_sim's
                     `action_mode="ee_pose"` (IK to joint targets in the env)

Positions are metres in the world frame; quaternions are wxyz; the gripper
command is a closure in [0, 1] (0 = open, 1 = closed). q and -q are the same
rotation, so quaternions are canonicalized to qx >= 0: the gripper points
down (home orientation (0, 1, 0, 0)) and its yaw stays within about +-90
degrees, so |qx| stays well away from 0 (>= 0.68 in the exports). The
commanded quaternions already satisfy it; a few measured ones (~0.04% of
kitting frames) don't.

Both are min-max normalized to [-1, 1] with ONE set of ranges, so input and
output live in the same space: x/y/z and the quaternion use the
`action.ee_pose` range from the dataset's `meta/stats.json`, the gripper
uses its full [0, 1] range. Min-max rather than z-scoring because
`run_rollout` clamps policy actions to [-1, 1] before `postprocess_action`.
Dimensions that never change in the commands (qw / qz of a top-down gripper)
are unnormalized back to their constant value, so the policy can't tilt the
gripper by accident; measured poses may sit slightly outside the command
range, which only matters for the (unclamped) input.

Cameras (`config.cameras`): franka_sim's `observation.images.<cam>` videos.
`target_mask` adds `observation.images.target_mask`, a binary (0/255, 3
identical channels) amodal mask of the episode's target entity seen from the
top camera -- single-target tasks only (no `*_long` dataset has it). Eval
renders the same mask live with `FrankaEnv.render_target_mask`. It skips
photometric augmentation (color jitter, speckle noise); crop and camera
dropout still apply.

Other privileged extras (`observation.environment_state`, `cond.*`) aren't
used: the policy sees only proprioception, cameras and (optionally) the
language instruction, matching franka_sim's `vision` benchmark protocol.

Episodes are reset with seeds `EVAL_SEED, EVAL_SEED + 1, ...` (franka_sim's
held-out evaluation seeds; demos are recorded below them), so every policy
sees the same layouts.

franka_sim is an optional, separately installed package (its repo is
private, and it needs mujoco>=4): install it into this venv with e.g.
`uv pip install -e ~/programming/python/franka_sim`. It's only imported when
an env is built, so training works without it.
"""

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torchvision.transforms import v2

from savannah.tasks import BaseRobotTask
from savannah.utils.device import configure_mujoco_gl
from savannah.utils.observation import ObservationKey

STATE_DIM = 8
ACTION_DIM = 8

POSE_KEY = "observation.ee_pose"  # (7) measured x y z qw qx qy qz
ACTION_KEY = "action.ee_pose"  # (8) commanded x y z qw qx qy qz + gripper
QUAT = slice(3, 7)
QX = 4
GRIPPER = 7

# franka_sim.benchmark.EVAL_SEED, duplicated so this module imports without
# franka_sim installed.
EVAL_SEED = 100_000

# The target mask's name in `config.cameras`, its LeRobot feature, and the
# camera franka_sim renders it from (lerobot_export's default mask_camera).
MASK_CAMERA = "target_mask"
MASK_KEY = f"observation.images.{MASK_CAMERA}"
MASK_RENDER_CAMERA = "top"

# Added to franka_sim observations by FrankaSimObservation.
GRIPPER_COMMAND_OBS = "gripper_command"

# Dimensions whose min == max in the dataset would divide by zero: they
# normalize with a unit range and unnormalize back to their constant value.
_MIN_RANGE = 1e-6


def _load_lerobot_meta(repo_id: str):
    """Loads a hub LeRobotDataset's metadata only (no video/parquet download)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    return LeRobotDatasetMetadata(repo_id)


def canonicalize_quat(pose):
    """Flips each wxyz quaternion in `pose[..., 3:7]` to qx >= 0 (q and -q
    are the same rotation). Works on numpy arrays and tensors."""
    sign = (pose[..., QX : QX + 1] < 0) * -2 + 1  # -1 where qx < 0, else 1
    out = pose.clone() if isinstance(pose, torch.Tensor) else pose.copy()
    out[..., QUAT] = pose[..., QUAT] * sign
    return out


class FrankaSimObservation(gym.ObservationWrapper):
    """Adds what the policy needs to franka_sim observations:

    - `gripper_command`: the gripper command in force (the last one sent;
      0 = open after reset), the gripper part of the policy's state.
    - `rgb_target_mask` (with `mask_size`): the live target mask, (H, W, 3)
      uint8 0/255 like the dataset's PNGs.
    """

    def __init__(self, env: gym.Env, mask_size: Optional[Tuple[int, int]] = None):
        super().__init__(env)
        self._mask_size = mask_size
        extra = {
            GRIPPER_COMMAND_OBS: spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
        }
        if mask_size is not None:
            extra[f"rgb_{MASK_CAMERA}"] = spaces.Box(
                0, 255, (*mask_size, 3), dtype=np.uint8
            )
        self.observation_space = spaces.Dict({**env.observation_space.spaces, **extra})

    def observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        franka_env = self.unwrapped.env  # franka_sim.env.FrankaEnv
        obs = {
            **obs,
            GRIPPER_COMMAND_OBS: np.array(
                [franka_env.last_command["gripper"][0]], dtype=np.float32
            ),
        }
        if self._mask_size is not None:
            h, w = self._mask_size
            mask = franka_env.render_target_mask(
                camera=MASK_RENDER_CAMERA, height=h, width=w
            )
            if mask is None:
                raise ValueError(
                    f"franka_sim task {franka_env.task.name!r} has no single "
                    f"target, so it has no target mask; remove {MASK_CAMERA!r} "
                    "from task.config.cameras"
                )
            obs[f"rgb_{MASK_CAMERA}"] = np.repeat(mask[..., None], 3, axis=2)
        return obs


class FrankaSimTask(BaseRobotTask):
    """A franka_sim benchmark task, trained from its LeRobot export.

    `config.env_name` is the franka_sim task name (e.g. "hammer_nails") and
    `config.repo_id` the LeRobot dataset recorded from it.
    """

    action_space = "ee_pose"

    def __init__(
        self,
        config,
        device: torch.device,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
    ):
        super().__init__(config, device)
        # Declared in the task config so model configs can read them; they
        # are fixed by the [ee_pose, gripper] representation.
        if (state_dim, action_dim) != (STATE_DIM, ACTION_DIM):
            raise ValueError(
                f"FrankaSimTask's state and actions are {STATE_DIM}-dim "
                f"[ee_pose, gripper]; got state_dim={state_dim}, "
                f"action_dim={action_dim}"
            )
        if not config.env_name:
            raise ValueError(
                "FrankaSimTask requires config.env_name (a franka_sim task, "
                "e.g. 'kitting_long')"
            )
        if config.action_key != ACTION_KEY:
            raise ValueError(
                f"FrankaSimTask predicts {ACTION_KEY!r} ([ee_pose, gripper]); "
                f"set config.action_key: {ACTION_KEY} (got {config.action_key!r})"
            )

        cameras = list(config.cameras)
        self._rgb_cameras = [c for c in cameras if c != MASK_CAMERA]
        self._use_mask = MASK_CAMERA in cameras
        # Per entry of config.cameras: does it get color jitter / noise?
        self._photometric = [c != MASK_CAMERA for c in cameras]

        self._meta = None
        self._norm_stats: Optional[Dict[str, np.ndarray]] = None
        self._norm_tensors: Optional[Tuple[torch.Tensor, ...]] = None
        self._resize = (
            v2.Resize((config.image_size, config.image_size), antialias=True)
            if config.image_size is not None
            else None
        )
        self._next_seed = EVAL_SEED
        self._instruction: Optional[str] = None
        self._num_subgoals: Optional[int] = None

    # ------------------------------------------------------------------
    # Dataset metadata / normalization
    # ------------------------------------------------------------------
    def _get_meta(self):
        if self._meta is None:
            meta = _load_lerobot_meta(self.config.repo_id)
            missing = [
                key for key in self.delta_timestamps() if key not in meta.features
            ]
            if missing:
                hint = (
                    " (only single-target tasks have a target mask; no *_long "
                    "dataset does)"
                    if MASK_KEY in missing
                    else ""
                )
                raise ValueError(
                    f"{self.config.repo_id} has no {missing}{hint}. "
                    f"It has: {sorted(meta.features)}"
                )
            self._meta = meta
        return self._meta

    def delta_timestamps(self) -> Dict[str, List[float]]:
        fps, oh = self.config.fps, self.config.obs_horizon
        obs_ts = [-(oh - 1 - i) / fps for i in range(oh)]
        deltas = {POSE_KEY: obs_ts}
        for cam in self.config.cameras:
            deltas[f"observation.images.{cam}"] = obs_ts
        # The oh commands *before* the obs frames (their gripper part is the
        # command in force at each frame), then the action horizon itself.
        deltas[ACTION_KEY] = [
            (i - oh) / fps for i in range(oh + self.config.action_horizon)
        ]
        return deltas

    def get_train_loader(self):
        self._get_meta()  # fail on a missing feature before any download
        return super().get_train_loader()

    def _get_norm_stats(self) -> Dict[str, np.ndarray]:
        """One (low, high) range for both state and actions."""
        if self._norm_stats is None:
            stats = self._get_meta().stats[ACTION_KEY]
            low = np.asarray(stats["min"], dtype=np.float32).copy()
            high = np.asarray(stats["max"], dtype=np.float32).copy()
            low[GRIPPER], high[GRIPPER] = 0.0, 1.0
            const = high - low < _MIN_RANGE
            high = np.where(const, low + 1.0, high)
            self._norm_stats = {"low": low, "high": high, "const": const}
        return self._norm_stats

    def _get_norm_tensors(self) -> Tuple[torch.Tensor, ...]:
        if self._norm_tensors is None:
            s = self._get_norm_stats()
            self._norm_tensors = tuple(
                torch.as_tensor(s[k], device=self.device)
                for k in ("low", "high", "const")
            )
        return self._norm_tensors

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        low, high, _ = self._get_norm_tensors()
        return (x.to(self.device) - low) / (high - low) * 2 - 1

    def _unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        low, high, const = self._get_norm_tensors()
        return torch.where(const, low, (x + 1) / 2 * (high - low) + low)

    def _action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Raw (low, high) actions are clipped to before `env.step()`: the
        position range seen in the demos (franka_sim has no workspace limit
        of its own), any unit quaternion, gripper in [0, 1]."""
        s = self._get_norm_stats()
        low = np.concatenate([s["low"][:3], -np.ones(4), [0.0]])
        high = np.concatenate([s["high"][:3], np.ones(4), [1.0]])
        return low.astype(np.float32), high.astype(np.float32)

    def action_limits(self) -> Tuple[List[float], List[float]]:
        low, high = self._action_bounds()
        return low.tolist(), high.tolist()

    def normalization_stats(self) -> Dict[str, Any]:
        s = self._get_norm_stats()
        entry = {
            "type": "min_max",
            "low": s["low"].tolist(),
            "high": s["high"].tolist(),
            "constant_dims": np.flatnonzero(s["const"]).tolist(),
            "quaternion": "wxyz, canonicalized to qx >= 0",
        }
        return {"state": entry, "actions": entry}

    def make_state(
        self, ee_pose: torch.Tensor, gripper_command: torch.Tensor
    ) -> torch.Tensor:
        """Normalized policy state from raw (..., 7) poses and (..., 1)
        gripper commands."""
        raw = torch.cat([ee_pose, gripper_command], dim=-1).to(self.device).float()
        return self._normalize(canonicalize_quat(raw))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def format_batch(
        self, batch: Dict[str, Any], step: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        oh = self.config.obs_horizon
        pose = batch[POSE_KEY].to(self.device)
        if pose.ndim == 2:  # obs_horizon == 1 may drop the time dim
            pose = pose.unsqueeze(1)

        commands = batch[ACTION_KEY].to(self.device)  # (B, oh + H, 8)
        gripper = commands[:, :oh, GRIPPER:].clone()  # command in force per frame
        pad_key = f"{ACTION_KEY}_is_pad"
        if pad_key in batch:
            # Before the episode's first command the gripper is open (0),
            # exactly as franka_sim resets it.
            gripper[batch[pad_key][:, :oh].to(self.device)] = 0.0

        state = self.make_state(pose, gripper)
        actions = self._normalize(canonicalize_quat(commands[:, oh:]))

        images = []
        for cam in self.config.cameras:
            img = batch[f"observation.images.{cam}"].to(self.device)
            if img.ndim == 4:  # obs_horizon == 1 may drop the time dim
                img = img.unsqueeze(1)
            images.append(img)  # (B, obs_horizon, C, H, W)

        if step is not None:
            images = self._augmenter.augment_images(
                images, step, photometric=self._photometric
            )
            state = self._augmenter.augment_state(state, step)

        result = {
            ObservationKey.images: images,
            ObservationKey.state: state,
            ObservationKey.gt_actions: actions,
        }
        if self.config.use_language:
            result[ObservationKey.language] = batch[ObservationKey.language]
        return result

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------
    def _make_env(self, render_mode: str) -> gym.Env:
        configure_mujoco_gl()
        from franka_sim.gym_env import FrankaGymEnv

        # Render at the dataset's own resolution, then resize with the same
        # transform the training loader applies, so eval images go through
        # the same pipeline as the training videos.
        features = self._get_meta().features
        first = (self._rgb_cameras or [MASK_CAMERA])[0]
        h, w = features[f"observation.images.{first}"]["shape"][:2]
        env = FrankaGymEnv(
            self.config.env_name,
            render_mode=render_mode,
            render_camera=self._rgb_cameras[0] if self._rgb_cameras else "front",
            action_mode="ee_pose",
            cameras=tuple(self._rgb_cameras),
            image_size=(h, w),
        )
        self._num_subgoals = env.env.task.num_steps
        mask_size = tuple(features[MASK_KEY]["shape"][:2]) if self._use_mask else None
        return FrankaSimObservation(env, mask_size)

    def reset_env(self, env: gym.Env, seed: Optional[int] = None, **kwargs):
        # Other tasks' reset options (e.g. ManiSkill's pick_color) arrive in
        # kwargs from scripts/eval/eval.py and are ignored here.
        if seed is None:
            seed = self._next_seed
            self._next_seed += 1
        obs, info = env.reset(seed=seed)
        self._instruction = str(info["instruction"])
        return obs, info

    def _obs_to_tensors(self, obs_list: List[Dict[str, Any]]) -> dict:
        pose = np.stack(
            [np.concatenate([o["ee_pos"], o["ee_quat"]]) for o in obs_list]
        ).astype(np.float32)  # (T, 7)
        gripper = np.stack([o[GRIPPER_COMMAND_OBS] for o in obs_list]).astype(
            np.float32
        )  # (T, 1)
        state = self.make_state(
            torch.from_numpy(pose).unsqueeze(0), torch.from_numpy(gripper).unsqueeze(0)
        )  # (1, T, 8)

        images = []
        for cam in self.config.cameras:
            imgs = np.stack([o[f"rgb_{cam}"] for o in obs_list])  # (T, H, W, 3)
            img = torch.from_numpy(imgs).permute(0, 3, 1, 2).float().divide(255.0)
            if self._resize is not None:
                img = self._resize(img)
            images.append(img.unsqueeze(0).to(self.device))  # (1, T, C, H, W)

        result = {ObservationKey.images: images, ObservationKey.state: state}
        if self.config.use_language and self._instruction is not None:
            result[ObservationKey.language] = [self._instruction]
        return result

    def preprocess_observation(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return self._obs_to_tensors([obs])

    def preprocess_observation_history(self, obs_history: deque) -> dict:
        obs_list = list(obs_history)
        if len(obs_list) < self.config.obs_horizon:
            pad = [obs_list[0]] * (self.config.obs_horizon - len(obs_list))
            obs_list = pad + obs_list
        return self._obs_to_tensors(obs_list)

    def postprocess_chunk(self, actions: torch.Tensor) -> None:
        # ActionChunkVisualizer draws 2D pixel coords (pusht-only); these are
        # 3D poses, so there's nothing to draw.
        return None

    def postprocess_action(self, action_tensor: torch.Tensor) -> np.ndarray:
        action = self._unnormalize(action_tensor.to(self.device))
        action = action.detach().cpu().numpy().reshape(-1).astype(np.float32)
        low, high = self._action_bounds()
        return np.clip(action, low, high)

    def is_success(self, info: Dict[str, Any]) -> bool:
        return bool(info.get("success", False))

    def get_metrics(self, info: Dict[str, Any]) -> str:
        progress = int(info.get("progress", 0))
        total = self._num_subgoals if self._num_subgoals is not None else "?"
        return (
            f"progress: {progress}/{total}, "
            f"wrong_target: {bool(info.get('wrong_target', False))}"
        )
