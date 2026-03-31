import gym_pusht  # Registers the env
import gymnasium as gym
import numpy as np
import torch

from savannah.tasks import BaseRobotTask
from savannah.utils.observation import ObservationKey
from collections import deque


class PushTTask(BaseRobotTask):
    def _make_env(self, render_mode: str) -> gym.Env:
        return gym.make(
            "gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode=render_mode
        )

    def format_batch(self, batch: dict) -> dict:
        images = batch["observation.image"].to(self.device)
        state = batch["observation.state"].to(self.device)
        actions = batch["action"].to(self.device)

        if images.ndim == 4:
            images = images.unsqueeze(1)

        return {
            ObservationKey.images: [images],
            ObservationKey.state: (state - 256.0) / 512.0,
            ObservationKey.gt_actions: (actions - 256.0) / 512.0,
        }

    def preprocess_observation(self, obs: dict) -> dict:
        # Prepares a single Gym observation for inference
        img_tensor = (
            torch.from_numpy(obs["pixels"])
            .permute(2, 0, 1)
            .float()
            .divide(255.0)
            .unsqueeze(0)  # Add Batch Dim
            .unsqueeze(0)  # Add Time/History Dim (N=1 for simple eval)
            .to(self.device)
        )

        raw_state = obs["agent_pos"]
        state_tensor = (
            torch.from_numpy((raw_state - 256.0) / 512.0)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )

        return {ObservationKey.images: [img_tensor], ObservationKey.state: state_tensor}

    def preprocess_observation_history(self, obs_history: deque) -> dict:
        """
        Prepares a history of Gym observations for inference.
        obs_history: deque of raw obs dicts, oldest first, newest last.
                     If shorter than obs_horizon, the earliest observation
                     is repeated to pad up to the required length.
        """
        obs_horizon = self.config.obs_horizon
        obs_list = list(obs_history)

        # Pad on the left by repeating the earliest available observation
        if len(obs_list) < obs_horizon:
            pad = [obs_list[0]] * (obs_horizon - len(obs_list))
            obs_list = pad + obs_list

        imgs = np.stack([o["pixels"] for o in obs_list])  # (T, H, W, C)
        img_tensor = (
            torch.from_numpy(imgs)
            .permute(0, 3, 1, 2)  # (T, C, H, W)
            .float()
            .divide(255.0)
            .unsqueeze(0)  # (1, T, C, H, W)
            .to(self.device)
        )

        states = np.stack([o["agent_pos"] for o in obs_list])  # (T, 2)
        state_tensor = (
            torch.from_numpy((states - 256.0) / 512.0)
            .float()
            .unsqueeze(0)  # (1, T, 2)
            .to(self.device)
        )

        return {ObservationKey.images: [img_tensor], ObservationKey.state: state_tensor}

    def postprocess_action(self, action_tensor: torch.Tensor) -> np.ndarray:
        # Takes the chosen single action (or chunk) and un-normalizes
        action_np = action_tensor.cpu().numpy().reshape(-1)
        return (action_np * 512.0) + 256.0


if __name__ == "__main__":
    from savannah.data.dataset import DataSetConfig
    from savannah.utils.device import get_device

    ds_config = DataSetConfig(
        repo_id="lerobot/pusht",
        fps=10,
        obs_horizon=1,
        action_horizon=8,
        cameras=["image"],
    )
    task = PushTTask(config=ds_config, device=get_device())
    env = task.get_env()
    print(env)
