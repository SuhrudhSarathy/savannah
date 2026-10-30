"""Download-free stand-ins shared across the test suite.

Not a test module itself (no test_ prefix). `tests/` is on the pytest
pythonpath (see pyproject.toml), so test modules import this directly, and
the smoke tests put it on PYTHONPATH so Hydra can `_target_` SyntheticTask
from inside scripts/train.py.

For a vision backbone stand-in, use savannah.models.backbones.DummyVisionBackbone
(it lives in src/ so Hydra configs can target it too).
"""

from collections import deque
from typing import ClassVar

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from savannah.data.dataset import DataSetConfig
from savannah.tasks import BaseRobotTask
from savannah.utils.observation import ObservationKey


class StubLanguageEncoder(nn.Module):
    """Lightweight stand-in for LanguageEncoder (avoids downloading CLIP weights)."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Linear(1, embed_dim)

    def forward(self, text_inputs: list[str]) -> torch.Tensor:
        batch = len(text_inputs)
        dummy = torch.ones(batch, 1, 1, device=self.proj.weight.device)
        return self.proj(dummy)


def _camera_key(cam: str) -> str:
    return f"observation.images.{cam}"


class _SyntheticDataset(Dataset):
    """Fixed random samples laid out like a LeRobot dataset item."""

    def __init__(
        self,
        config: DataSetConfig,
        state_dim: int,
        action_dim: int,
        num_samples: int,
        seed: int,
    ):
        g = torch.Generator().manual_seed(seed)
        n, h, size = num_samples, config.obs_horizon, config.image_size
        self.images = {
            cam: torch.rand(n, h, 3, size, size, generator=g) for cam in config.cameras
        }
        self.state = torch.randn(n, h, state_dim, generator=g)
        self.action = torch.randn(n, config.action_horizon, action_dim, generator=g)

    def __len__(self):
        return len(self.state)

    def __getitem__(self, idx):
        item = {_camera_key(cam): imgs[idx] for cam, imgs in self.images.items()}
        item["observation.state"] = self.state[idx]
        item["action"] = self.action[idx]
        return item


class SyntheticEnv(gym.Env):
    """Random-pixel env that terminates after `max_steps`. Never succeeds."""

    render_mode = "rgb_array"
    metadata: ClassVar[dict] = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self, cameras: list[str], image_size: int, state_dim: int, max_steps: int
    ):
        self.cameras = cameras
        self.image_size = image_size
        self.state_dim = state_dim
        self.max_steps = max_steps
        self._t = 0

    def _obs(self) -> dict:
        size = self.image_size
        return {
            "pixels": {
                cam: self.np_random.integers(0, 256, (size, size, 3), dtype=np.uint8)
                for cam in self.cameras
            },
            "agent_pos": self.np_random.standard_normal(self.state_dim).astype(
                np.float32
            ),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        self._last_obs = self._obs()
        return self._last_obs, {}

    def step(self, action):
        self._t += 1
        self._last_obs = self._obs()
        terminated = self._t >= self.max_steps
        return self._last_obs, 0.0, terminated, False, {}

    def render(self):
        return self._last_obs["pixels"][self.cameras[0]]


class SyntheticTask(BaseRobotTask):
    """A BaseRobotTask backed by in-memory random data and SyntheticEnv.

    Exercises the trainer's full loop (data loading, format_batch +
    augmentation, validation, sim eval, checkpointing) without touching the
    Hub or a simulator.
    """

    def __init__(
        self,
        config: DataSetConfig,
        device: torch.device,
        state_dim: int,
        action_dim: int,
        num_train_samples: int = 8,
        num_val_samples: int = 4,
        env_max_steps: int = 6,
    ):
        super().__init__(config, device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.env_max_steps = env_max_steps

    def _loader(self, num_samples: int, seed: int, shuffle: bool) -> DataLoader:
        dataset = _SyntheticDataset(
            self.config, self.state_dim, self.action_dim, num_samples, seed
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
        )

    def get_train_loader(self):
        if self._train_loader is None:
            self._train_loader = self._loader(self.num_train_samples, 0, True)
            self._val_loader = self._loader(self.num_val_samples, 1, False)
        return self._train_loader

    def get_episode_loader(self, episode_index: int, frame_index: int | None = 0):
        dataset = _SyntheticDataset(
            self.config, self.state_dim, self.action_dim, 1, seed=episode_index
        )
        return DataLoader(dataset, batch_size=1)

    def format_batch(self, batch: dict, step: int | None = None) -> dict:
        images = [
            batch[_camera_key(cam)].to(self.device) for cam in self.config.cameras
        ]
        state = batch["observation.state"].to(self.device)
        if step is not None:
            images = self._augmenter.augment_images(images, step)
            state = self._augmenter.augment_state(state, step)

        return {
            ObservationKey.images: images,
            ObservationKey.state: state,
            ObservationKey.gt_actions: batch["action"].to(self.device),
        }

    def _make_env(self, render_mode: str) -> gym.Env:
        return SyntheticEnv(
            cameras=list(self.config.cameras),
            image_size=self.config.image_size,
            state_dim=self.state_dim,
            max_steps=self.env_max_steps,
        )

    def preprocess_observation(self, obs: dict) -> dict:
        return self.preprocess_observation_history(deque([obs]))

    def preprocess_observation_history(self, obs_history: deque) -> dict:
        obs_list = list(obs_history)
        images = []
        for cam in self.config.cameras:
            frames = np.stack([o["pixels"][cam] for o in obs_list])  # (T, H, W, C)
            images.append(
                torch.from_numpy(frames)
                .permute(0, 3, 1, 2)
                .float()
                .divide(255.0)
                .unsqueeze(0)  # (1, T, C, H, W)
                .to(self.device)
            )
        states = np.stack([o["agent_pos"] for o in obs_list])  # (T, state_dim)
        state = torch.from_numpy(states).float().unsqueeze(0).to(self.device)
        return {ObservationKey.images: images, ObservationKey.state: state}

    def postprocess_action(self, action_tensor: torch.Tensor) -> np.ndarray:
        return action_tensor.float().cpu().numpy().reshape(-1)

    def postprocess_chunk(self, actions: torch.Tensor) -> None:
        return None
