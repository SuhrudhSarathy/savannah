from collections import deque

import gymnasium as gym
import metaworld  # noqa: F401  (registers Meta-World/MT1 with gymnasium)
import mujoco
import numpy as np
import torch
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from gymnasium.spaces import Dict as GymDict
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from savannah.tasks import BaseRobotTask
from savannah.utils.observation import ObservationKey

# We only need the current ee position and the gripper state
STATE_KEEP_INDICES = [0, 1, 2, 3]
IMG_SIZE = (224, 224)


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

    def close(self):
        self._renderer.close()
        super().close()


class MetaworldAssemblyTask(BaseRobotTask):
    ENV_NAME = "assembly-v3"

    def __init__(self, config, device: torch.device):
        super().__init__(config, device)

        stats = LeRobotDatasetMetadata(config.repo_id).stats
        self._state_mean = torch.tensor(
            stats["observation.state"]["mean"][STATE_KEEP_INDICES],
            dtype=torch.float32,
            device=device,
        )
        self._state_std = torch.tensor(
            stats["observation.state"]["std"][STATE_KEEP_INDICES],
            dtype=torch.float32,
            device=device,
        ).clamp_min(1e-4)
        self._action_mean = torch.tensor(
            stats["action"]["mean"], dtype=torch.float32, device=device
        )
        self._action_std = torch.tensor(
            stats["action"]["std"], dtype=torch.float32, device=device
        ).clamp_min(1e-4)

    def _make_env(self, render_mode: str) -> gym.Env:
        env = gym.make(
            "Meta-World/MT1", env_name=self.ENV_NAME, render_mode=render_mode
        )
        return MultiCameraObsWrapper(
            env, camera_names=self.config.cameras, img_size=IMG_SIZE
        )

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        return (state - self._state_mean) / self._state_std

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self._action_mean) / self._action_std

    def _unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self._action_std + self._action_mean

    def format_batch(self, batch: dict) -> dict:
        state = batch["observation.state"].to(self.device)[..., STATE_KEEP_INDICES]
        actions = batch["action"].to(self.device)

        images = []
        for cam in self.config.cameras:
            img = batch[f"observation.images.{cam}"].to(self.device)
            if img.ndim == 4:
                img = img.unsqueeze(1)
            images.append(img)

        return {
            ObservationKey.images: images,
            ObservationKey.state: self._normalize_state(state),
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

    def postprocess_action(self, action_tensor: torch.Tensor) -> np.ndarray:
        action = self._unnormalize_action(action_tensor.to(self.device))
        action = torch.clamp(action, -1.0, 1.0)
        return action.detach().cpu().numpy().reshape(-1).astype(np.float32)
