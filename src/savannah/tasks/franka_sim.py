"""Tasks from franka_sim (https://github.com/SuhrudhSarathy/franka_sim), a
MuJoCo Franka Panda benchmark, trained from its LeRobot exports.

The action mode is picked by `config.action_key` (see ACTION_MODES). Either
way, policy input and output share one representation (8 values), the arm
part of which is:

    action.ee_pose (default)   [x, y, z, qw, qx, qy, qz, gripper]
        state  (input):  the measured end-effector pose (`observation.ee_pose`)
        action (output): the commanded pose (`action.ee_pose`), rolled out
                         with franka_sim's `action_mode="ee_pose"` (IK to
                         joint targets in the env)
    action.joint_pos           [q1 .. q7, gripper]
        state  (input):  the measured joint positions (the first 7 values of
                         `observation.state`)
        action (output): the commanded joint targets (`action.joint_pos`),
                         rolled out with `action_mode="joint_pos"`

The state's last value is the gripper command in force at each obs frame, i.e.
the previous action's gripper value (0 = open at the start of an episode, as
franka_sim resets it).

ee_pose details: positions are metres in the world frame; quaternions are wxyz; the gripper
command is a closure in [0, 1] (0 = open, 1 = closed). q and -q are the same
rotation, so quaternions are canonicalized to qx >= 0: the gripper points
down (home orientation (0, 1, 0, 0)) and its yaw stays within about +-90
degrees, so |qx| stays well away from 0 (>= 0.68 in the exports). The
commanded quaternions already satisfy it; a few measured ones (~0.04% of
kitting frames) don't.

Both are min-max normalized to [-1, 1] with ONE set of ranges, so input and
output live in the same space: the arm part uses the range of
`config.action_key` from the dataset's `meta/stats.json`, the gripper uses
its full [0, 1] range. Min-max rather than z-scoring because
`run_rollout` clamps policy actions to [-1, 1] before `postprocess_action`.
Dimensions that never change in the commands (qw / qz of a top-down gripper)
are unnormalized back to their constant value, so the policy can't tilt the
gripper by accident; measured values may sit slightly outside the command
range, which only matters for the (unclamped) input. Before `env.step()`,
ee_pose actions are clipped to the demos' workspace and joint targets to the
Panda joint limits.

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

Eval follows franka_sim's benchmark protocol (`franka_sim.benchmark.evaluate`):
episode k of every rollout is reset with seed `EVAL_SEED + k` (held-out seeds;
demos are recorded below them) and, for single tasks, target `k % num_targets`,
so every policy sees the same layouts and a balanced set of targets. Acting on
a wrong target ends the episode as a failure.

franka_sim is an optional, separately installed package (its repo is
private, and it needs mujoco>=4): install it into this venv with e.g.
`uv pip install -e ~/programming/python/franka_sim`. It's only imported when
an env is built, so training works without it.
"""

from collections import deque
from dataclasses import dataclass
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
JOINT_STATE_KEY = "observation.state"  # (8) 7 joint positions + gripper width
ACTION_KEY = "action.ee_pose"  # (8) commanded x y z qw qx qy qz + gripper
JOINT_ACTION_KEY = "action.joint_pos"  # (8) 7 joint targets + gripper
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

# Panda joint position limits (rad), from franka_sim's mjx_panda.xml actuator
# ctrlranges, plus the gripper closure command in [0, 1].
JOINT_ACTION_LOW = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0],
    dtype=np.float32,
)
JOINT_ACTION_HIGH = np.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1.0],
    dtype=np.float32,
)


@dataclass(frozen=True)
class ActionMode:
    """How one action representation is trained and rolled out."""

    env_action_mode: str  # FrankaEnv action_mode
    action_space: str  # recorded in checkpoint manifests
    state_key: str  # dataset feature whose first 7 values are the arm state
    env_state_keys: Tuple[str, ...]  # env obs keys concatenated into it
    canonicalize_quat: bool


# `config.action_key` -> its action mode.
ACTION_MODES = {
    ACTION_KEY: ActionMode(
        env_action_mode="ee_pose",
        action_space="ee_pose",
        state_key=POSE_KEY,
        env_state_keys=("ee_pos", "ee_quat"),
        canonicalize_quat=True,
    ),
    JOINT_ACTION_KEY: ActionMode(
        env_action_mode="joint_pos",
        action_space="joint_position",
        state_key=JOINT_STATE_KEY,
        env_state_keys=("joint_pos",),
        canonicalize_quat=False,
    ),
}

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


class FrankaSimWrongTarget(gym.Wrapper):
    """Ends an episode once the policy acts on a wrong target, as franka_sim's
    benchmark does: FrankaEnv itself only terminates on success. The flag is
    latched (`info["wrong_target"]` stays True to the end) and the episode
    counts as a failure."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._wrong = False

    def reset(self, **kwargs):
        self._wrong = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._wrong |= bool(info.get("wrong_target", False))
        if self._wrong:
            info = {**info, "wrong_target": True, "success": False}
            terminated = True
        return obs, reward, terminated, truncated, info


class FrankaSimTask(BaseRobotTask):
    """A franka_sim benchmark task, trained from its LeRobot export.

    `config.env_name` is the franka_sim task name (e.g. "hammer_nails"),
    `config.repo_id` the LeRobot dataset recorded from it, and
    `config.action_key` the action mode (see ACTION_MODES).
    """

    def __init__(
        self,
        config,
        device: torch.device,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
    ):
        super().__init__(config, device)
        # Declared in the task config so model configs can read them; they
        # are fixed by the [arm (7), gripper] representation of every mode.
        if (state_dim, action_dim) != (STATE_DIM, ACTION_DIM):
            raise ValueError(
                f"FrankaSimTask's state and actions are {STATE_DIM}-dim "
                f"[arm (7), gripper]; got state_dim={state_dim}, "
                f"action_dim={action_dim}"
            )
        if not config.env_name:
            raise ValueError(
                "FrankaSimTask requires config.env_name (a franka_sim task, "
                "e.g. 'kitting_long')"
            )
        if config.action_key not in ACTION_MODES:
            raise ValueError(
                f"FrankaSimTask needs config.action_key in {sorted(ACTION_MODES)}, "
                f"got {config.action_key!r}"
            )
        self.mode = ACTION_MODES[config.action_key]
        self.action_space = self.mode.action_space

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
        deltas = {self.mode.state_key: obs_ts}
        for cam in self.config.cameras:
            deltas[f"observation.images.{cam}"] = obs_ts
        # The oh commands *before* the obs frames (their gripper part is the
        # command in force at each frame), then the action horizon itself.
        deltas[self.config.action_key] = [
            (i - oh) / fps for i in range(oh + self.config.action_horizon)
        ]
        return deltas

    def get_train_loader(self):
        self._get_meta()  # fail on a missing feature before any download
        return super().get_train_loader()

    def _get_norm_stats(self) -> Dict[str, np.ndarray]:
        """One (low, high) range for both state and actions."""
        if self._norm_stats is None:
            stats = self._get_meta().stats[self.config.action_key]
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
        """Raw (low, high) actions are clipped to before `env.step()`.
        joint_pos: the Panda joint limits, gripper in [0, 1]. ee_pose: the
        position range seen in the demos (franka_sim has no workspace limit
        of its own), any unit quaternion, gripper in [0, 1]."""
        if not self.mode.canonicalize_quat:
            return JOINT_ACTION_LOW, JOINT_ACTION_HIGH
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
        }
        if self.mode.canonicalize_quat:
            entry["quaternion"] = "wxyz, canonicalized to qx >= 0"
        return {"state": entry, "actions": entry}

    def _canonical(self, x: torch.Tensor) -> torch.Tensor:
        return canonicalize_quat(x) if self.mode.canonicalize_quat else x

    def make_state(
        self, arm: torch.Tensor, gripper_command: torch.Tensor
    ) -> torch.Tensor:
        """Normalized policy state from the raw (..., 7) arm state (ee pose or
        joint positions, per the action mode) and (..., 1) gripper commands."""
        raw = torch.cat([arm, gripper_command], dim=-1).to(self.device).float()
        return self._normalize(self._canonical(raw))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def format_batch(
        self, batch: Dict[str, Any], step: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        oh = self.config.obs_horizon
        arm = batch[self.mode.state_key].to(self.device)[..., :GRIPPER]
        if arm.ndim == 2:  # obs_horizon == 1 may drop the time dim
            arm = arm.unsqueeze(1)

        action_key = self.config.action_key
        commands = batch[action_key].to(self.device)  # (B, oh + H, 8)
        gripper = commands[:, :oh, GRIPPER:].clone()  # command in force per frame
        pad_key = f"{action_key}_is_pad"
        if pad_key in batch:
            # Before the episode's first command the gripper is open (0),
            # exactly as franka_sim resets it.
            gripper[batch[pad_key][:, :oh].to(self.device)] = 0.0

        state = self.make_state(arm, gripper)
        actions = self._normalize(self._canonical(commands[:, oh:]))

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
            action_mode=self.mode.env_action_mode,
            cameras=tuple(self._rgb_cameras),
            image_size=(h, w),
        )
        self._num_subgoals = env.env.task.num_steps
        mask_size = tuple(features[MASK_KEY]["shape"][:2]) if self._use_mask else None
        return FrankaSimWrongTarget(FrankaSimObservation(env, mask_size))

    def begin_eval(self) -> None:
        # Every rollout starts again at episode 0, so periodic evals during
        # training are scored on the same layouts.
        self._next_seed = EVAL_SEED

    def reset_env(self, env: gym.Env, seed: Optional[int] = None, **kwargs):
        # Other tasks' reset options (e.g. ManiSkill's pick_color) arrive in
        # kwargs from scripts/eval/eval.py and are ignored here.
        from franka_sim.benchmark import episode_options

        if seed is None:
            seed = self._next_seed
            self._next_seed += 1
        # Episode k = seed - EVAL_SEED; single tasks get target k % num_targets.
        options = episode_options(self.config.env_name, max(seed - EVAL_SEED, 0))
        obs, info = env.reset(seed=seed, options=options)
        self._instruction = str(info["instruction"])
        return obs, info

    def _obs_to_tensors(self, obs_list: List[Dict[str, Any]]) -> dict:
        arm = np.stack(
            [np.concatenate([o[k] for k in self.mode.env_state_keys]) for o in obs_list]
        ).astype(np.float32)  # (T, 7)
        gripper = np.stack([o[GRIPPER_COMMAND_OBS] for o in obs_list]).astype(
            np.float32
        )  # (T, 1)
        state = self.make_state(
            torch.from_numpy(arm).unsqueeze(0), torch.from_numpy(gripper).unsqueeze(0)
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
        # 3D poses / joint targets, so there's nothing to draw.
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
