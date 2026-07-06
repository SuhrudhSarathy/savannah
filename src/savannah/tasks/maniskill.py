from collections import deque

import gymnasium as gym
import numpy as np
import torch

from savannah.tasks import BaseRobotTask
from savannah.utils.observation import ObservationKey

IMG_SIZE = (128, 128)

_ACTION_LOW = np.full(7, -1.0, dtype=np.float32)
_ACTION_HIGH = np.full(7, 1.0, dtype=np.float32)


class ManiSkillColorMatchingTask(BaseRobotTask):
    def __init__(self, config, device: torch.device):
        super().__init__(config, device)
        self._action_low = torch.tensor(_ACTION_LOW, device=device)
        self._action_high = torch.tensor(_ACTION_HIGH, device=device)

    def _make_env(self, render_mode: str) -> gym.Env:
        import savannah.envs  # noqa: F401 — triggers @register_env

        return gym.make(
            "ColorMatching-v1",
            num_envs=1,
            obs_mode="rgbd",
            control_mode="pd_ee_delta_pose",
            render_mode=render_mode,
        )

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self._action_low) / (
            self._action_high - self._action_low
        ) * 2 - 1

    def _unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action + 1) / 2 * (
            self._action_high - self._action_low
        ) + self._action_low

    def _extract_obs(self, obs: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        tcp_pose = obs["extra"]["tcp_pose"]
        if isinstance(tcp_pose, torch.Tensor):
            tcp_pose = tcp_pose.cpu().numpy().flatten()
        state = tcp_pose[:3]

        images = {}
        for cam in self.config.cameras:
            rgb = obs["sensor_data"][cam]["rgb"]
            if isinstance(rgb, torch.Tensor):
                rgb = rgb.cpu().numpy()
            images[cam] = rgb[..., :3].astype(np.uint8)

        return state, images

    def _obs_to_tensors(self, obs_list: list[dict]) -> dict:
        extracted = [self._extract_obs(o) for o in obs_list]

        states = np.stack([e[0] for e in extracted])
        state_tensor = torch.from_numpy(states).float().unsqueeze(0).to(self.device)

        img_list = []
        for cam in self.config.cameras:
            imgs = np.stack([e[1][cam] for e in extracted])
            img_tensor = (
                torch.from_numpy(imgs)
                .permute(0, 3, 1, 2)
                .float()
                .divide(255.0)
                .unsqueeze(0)
                .to(self.device)
            )
            img_list.append(img_tensor)

        return {ObservationKey.images: img_list, ObservationKey.state: state_tensor}

    def format_batch(self, batch: dict) -> dict:
        state = batch["observation.state"].to(self.device)
        actions = batch["action"].to(self.device)

        images = []
        for cam in self.config.cameras:
            img = batch[f"observation.images.{cam}"].to(self.device)
            if img.ndim == 4:
                img = img.unsqueeze(1)
            images.append(img)

        return {
            ObservationKey.images: images,
            ObservationKey.state: state,
            ObservationKey.gt_actions: self._normalize_action(actions),
        }

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

    def is_success(self, info: dict) -> bool:
        success = info.get("success", False)
        if isinstance(success, torch.Tensor):
            return success.any().item()
        return bool(success)
