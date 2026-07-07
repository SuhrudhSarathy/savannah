from abc import ABC, abstractmethod
import torch
import numpy as np
import gymnasium as gym
from typing import Dict, Any

from savannah.utils.observation import ObservationKey
from savannah.data.dataset import DataSetConfig, LerobotDatasetWrapper
from collections import deque


class BaseRobotTask(ABC):
    def __init__(self, config: "DataSetConfig", device: torch.device):
        self.config = config
        self.device = device

        # Will be populated lazily
        self._train_loader = None
        self._val_loader = None
        self._env = None

    def get_train_loader(self):
        if self._train_loader is None:
            self._train_loader, self._val_loader = LerobotDatasetWrapper.create_loaders(
                self.config, self.device
            )
        return self._train_loader

    def get_val_loader(self):
        if self._val_loader is None:
            self.get_train_loader()  # Forces creation of both
        return self._val_loader

    @abstractmethod
    def format_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Translates raw LeRobot string keys to your ObservationKey Enums,
        and applies normalization for training.
        """
        ...

    def get_env(self, render_mode: str = "rgb_array") -> gym.Env:
        if self._env is None:
            self._env = self._make_env(render_mode)
        return self._env

    def reset_env(self, env: gym.Env, **kwargs) -> tuple:
        """Resets `env` for a new episode.

        Base implementation ignores kwargs and does a plain reset. Override
        for tasks that need extra reset options (e.g. object colors) and want
        to cache per-episode context (e.g. a language instruction) for
        preprocess_observation/preprocess_observation_history to pick up.
        """
        return env.reset()

    @abstractmethod
    def _make_env(self, render_mode: str) -> gym.Env:
        """Instantiates the specific Gym environment."""
        ...

    @abstractmethod
    def preprocess_observation(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Converts a raw Gym observation dict into the exact normalized tensor
        dict expected by `policy.compute_action()`.
        """
        ...

    @abstractmethod
    def preprocess_observation_history(self, obs_history: deque) -> dict:
        """
        Prepares a history of Gym observations for inference.
        obs_history: deque of raw obs dicts, oldest first, newest last.
                     Length should equal obs_horizon used during training.
        """
        ...

    @abstractmethod
    def postprocess_action(self, action_tensor: torch.Tensor) -> np.ndarray:
        """
        Converts the policy's normalized action output [-1, 1] back into
        the raw physics values [0, 512] expected by `env.step()`.
        """
        ...

    def is_success(self, info: Dict[str, Any]) -> bool:
        return info.get("score", 0) > 0.95
