from collections import deque

import cv2
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from gymnasium.spaces import Dict as GymDict

from savannah.tasks import BaseRobotTask
from savannah.utils.device import configure_mujoco_gl
from savannah.utils.observation import ObservationKey

configure_mujoco_gl()

import metaworld  # noqa: E402,F401  (registers Meta-World/MT1 with gymnasium)
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

# We only need the current ee position and the gripper state
STATE_KEEP_INDICES = [0, 1, 2, 3]
IMG_SIZE = (224, 224)

# Bounds from SawyerXYZEnv._HAND_SPACE (indices 0-2) and gripper_low/high (index 3).
# These are class-level constants shared across all MT50 tasks — no env creation needed.
_STATE_LOW = np.array([-0.525, 0.348, -0.0525, -1.0], dtype=np.float32)
_STATE_HIGH = np.array([0.525, 1.025, 0.7, 1.0], dtype=np.float32)
# Action space is Box([-1,-1,-1,-1], [1,1,1,1]) for all MetaWorld envs.
_ACTION_LOW = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)
_ACTION_HIGH = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)


class MultiCameraObsWrapper(ObservationWrapper):
    """Renders extra named camera views on top of MetaWorld's proprioceptive obs."""

    def __init__(
        self,
        env: gym.Env,
        camera_names: list[str],
        img_size: tuple[int, int] = IMG_SIZE,
    ):
        super().__init__(env)
        model = env.unwrapped.model
        self.camera_names = camera_names
        h, w = img_size
        self._renderer = mujoco.Renderer(model, height=h, width=w)
        self.observation_space = GymDict(
            {
                "state": env.observation_space,
                **{cam: Box(0, 255, (h, w, 3), dtype=np.uint8) for cam in camera_names},
            }
        )

    def observation(self, obs: np.ndarray) -> dict:
        data = self.env.unwrapped.data
        frames = {"state": obs}
        for cam in self.camera_names:
            self._renderer.update_scene(data, camera=cam)
            frames[cam] = self._renderer.render()
        return frames

    def render(self):
        main_frame = self.env.render()
        if main_frame is None:
            return None
        main_h, main_w = main_frame.shape[:2]

        data = self.env.unwrapped.data
        thumbnails = []
        for cam in self.camera_names:
            self._renderer.update_scene(data, camera=cam)
            frame = self._renderer.render().copy()
            frame[:18] = (frame[:18] * 0.3).astype(np.uint8)
            cv2.putText(
                frame,
                cam,
                (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            thumbnails.append(frame)

        strip = np.concatenate(thumbnails, axis=1)
        strip_h, strip_w = strip.shape[:2]
        new_h = int(strip_h * main_w / strip_w)
        strip = cv2.resize(strip, (main_w, new_h))

        return np.concatenate([main_frame, strip], axis=0)

    def close(self):
        self._renderer.close()
        super().close()


class MetaworldTask(BaseRobotTask):
    def __init__(self, config, device: torch.device):
        super().__init__(config, device)

        self._state_low = torch.tensor(_STATE_LOW, device=device)
        self._state_high = torch.tensor(_STATE_HIGH, device=device)
        self._action_low = torch.tensor(_ACTION_LOW, device=device)
        self._action_high = torch.tensor(_ACTION_HIGH, device=device)

    def _make_env(self, render_mode: str) -> gym.Env:
        env = gym.make(
            "Meta-World/MT1", env_name=self.config.env_name, render_mode=render_mode
        )
        img_size = (
            (self.config.image_size, self.config.image_size)
            if self.config.image_size is not None
            else IMG_SIZE
        )
        return MultiCameraObsWrapper(
            env, camera_names=self.config.cameras, img_size=img_size
        )

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return (state - self._state_low) / (self._state_high - self._state_low) * 2 - 1

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self._action_low) / (
            self._action_high - self._action_low
        ) * 2 - 1

    def _unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action + 1) / 2 * (
            self._action_high - self._action_low
        ) + self._action_low

    def format_batch(self, batch: dict, step: int | None = None) -> dict:
        state = batch["observation.state"].to(self.device)[..., STATE_KEEP_INDICES]
        actions = batch["action"].to(self.device)

        images = []
        for cam in self.config.cameras:
            img = batch[f"observation.images.{cam}"].to(self.device)
            if img.ndim == 4:
                img = img.unsqueeze(1)
            images.append(img)

        state_norm = self._normalize_state(state)
        if step is not None:
            images = self._augmenter.augment_images(images, step)
            state_norm = self._augmenter.augment_state(state_norm, step)

        return {
            ObservationKey.images: images,
            ObservationKey.state: state_norm,
            ObservationKey.gt_actions: self._normalize_action(actions),
        }

    def _obs_to_tensors(self, obs_list: list[dict]) -> dict:
        """Stacks a list of raw env obs dicts (oldest first) into normalized tensors."""
        states = np.stack([o["state"][STATE_KEEP_INDICES] for o in obs_list])  # (T, 25)
        state_tensor = self._normalize_state(
            torch.from_numpy(states).float().unsqueeze(0).to(self.device)  # (1, T, 25)
        )

        images = []
        for cam in self.config.cameras:
            imgs = np.stack([o[cam] for o in obs_list])  # (T, H, W, 3) uint8
            img_tensor = (
                torch.from_numpy(imgs)
                .permute(0, 3, 1, 2)  # (T, C, H, W)
                .float()
                .divide(255.0)
                .unsqueeze(0)  # (1, T, C, H, W)
                .to(self.device)
            )
            images.append(img_tensor)

        return {ObservationKey.images: images, ObservationKey.state: state_tensor}

    def preprocess_observation(self, obs: dict) -> dict:
        return self._obs_to_tensors([obs])

    def preprocess_observation_history(self, obs_history: deque) -> dict:
        obs_horizon = self.config.obs_horizon
        obs_list = list(obs_history)

        if len(obs_list) < obs_horizon:
            pad = [obs_list[0]] * (obs_horizon - len(obs_list))
            obs_list = pad + obs_list

        return self._obs_to_tensors(obs_list)

    def postprocess_chunk(self, actions: torch.Tensor) -> None:
        return None

    def postprocess_action(self, action_tensor: torch.Tensor) -> np.ndarray:
        action = self._unnormalize_action(action_tensor.to(self.device))
        action = torch.clamp(action, -1.0, 1.0)
        return action.detach().cpu().numpy().reshape(-1).astype(np.float32)

    def is_success(self, info: dict) -> bool:
        return info["success"]
