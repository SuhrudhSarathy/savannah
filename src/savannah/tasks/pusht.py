import gym_pusht  # Registers the env
import gymnasium as gym
import numpy as np
import torch

from savannah.tasks import BaseRobotTask
from savannah.utils.observation import ObservationKey


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
