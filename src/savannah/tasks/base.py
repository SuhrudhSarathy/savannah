from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import torch

from savannah.data.augmentation import ObservationAugmenter
from savannah.data.dataset import DataSetConfig, LerobotDatasetWrapper


class BaseRobotTask(ABC):
    # What the env's actions mean, recorded in checkpoint manifests.
    action_space: str = "unknown"

    def __init__(self, config: "DataSetConfig", device: torch.device):
        self.config = config
        self.device = device
        self._augmenter = ObservationAugmenter(config.augmentation)

        # Will be populated lazily
        self._train_loader = None
        self._val_loader = None
        self._env = None

    def delta_timestamps(self) -> Dict[str, list[float]]:
        """LeRobot features to load, each at its time offsets (s) from the
        current frame. Default: state + cameras over the obs horizon and
        `config.action_key` over the action horizon. Override to load other
        features (e.g. past actions as an input)."""
        return LerobotDatasetWrapper._build_timestamps(self.config)

    def get_train_loader(self):
        if self._train_loader is None:
            self._train_loader, self._val_loader = LerobotDatasetWrapper.create_loaders(
                self.config, self.device, self.delta_timestamps()
            )
        return self._train_loader

    def get_val_loader(self):
        if self._val_loader is None:
            self.get_train_loader()  # Forces creation of both
        return self._val_loader

    def get_episode_loader(self, episode_index: int, frame_index: int | None = 0):
        """One fixed frame (frame_index=int) or the whole episode
        (frame_index=None) as a single batch — see
        LerobotDatasetWrapper.get_episode_loader."""
        return LerobotDatasetWrapper.get_episode_loader(
            self.config,
            self.device,
            episode_index,
            frame_index,
            delta_timestamps=self.delta_timestamps(),
        )

    @abstractmethod
    def format_batch(
        self, batch: Dict[str, Any], step: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Translates raw LeRobot string keys to your ObservationKey Enums,
        and applies normalization for training.

        Pass `step` (the current training step) to enable augmentation —
        it's used to compute the decay schedule in `self._augmenter`. Leave
        it None for validation/eval batches, which should stay unaugmented.
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

    def action_limits(self) -> Optional[tuple[list[float], list[float]]]:
        """Per-dimension (low, high) of the raw actions sent to `env.step()`,
        for checkpoint manifests. None if the task doesn't define any."""
        return None

    def normalization_stats(self) -> Dict[str, Any]:
        """JSON-serializable description of how `format_batch` normalizes
        state and actions, for checkpoint manifests. Keyed by "state"/"actions",
        each with a "type" and that type's parameters."""
        return {}

    def is_success(self, info: Dict[str, Any]) -> bool:
        return info.get("score", 0) > 0.95

    def get_metrics(self, info: Dict[str, Any]) -> str:
        return ""
