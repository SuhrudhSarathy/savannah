import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from torch.utils.data import DataLoader, Subset
import numpy as np

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

from savannah.utils import ObservationKey


@dataclass
class DataSetConfig:
    repo_id: str
    fps: int
    root: Optional[str] = None
    cameras: List[str] = field(default_factory=list)

    # Temporal horizons
    obs_horizon: int = 1
    action_horizon: int = 10

    # Dataloader specifics
    batch_size: int = 8
    num_workers: int = 4
    train_fraction: float = 0.95


class LerobotDatasetWrapper:
    """A wrapper/factory for handling LeRobot datasets scalably."""

    @staticmethod
    def _build_timestamps(config: DataSetConfig) -> dict:
        obs_timestamps = [
            -(config.obs_horizon - 1 - i) / config.fps
            for i in range(config.obs_horizon)
        ]
        act_timestamps = [i / config.fps for i in range(config.action_horizon)]

        deltas = {
            ObservationKey.state: obs_timestamps,
            ObservationKey.actions: act_timestamps,
        }

        for cam in config.cameras:
            # Handle standard LeRobot naming convention for cameras
            cam_key = (
                f"observation.images.{cam}" if cam != "image" else "observation.image"
            )
            deltas[cam_key] = obs_timestamps

        return deltas

    @classmethod
    def create_loaders(cls, config: DataSetConfig, device: torch.device):
        deltas = cls._build_timestamps(config)
        full_dataset = LeRobotDataset(config.repo_id, delta_timestamps=deltas)

        num_episodes = full_dataset.num_episodes
        all_episodes = np.arange(num_episodes)
        np.random.shuffle(all_episodes)
        split_idx = int(num_episodes * config.train_fraction)

        train_episodes = set(all_episodes[:split_idx].tolist())

        # Vectorized frame-to-episode mapping — no Python loop needed
        episode_indices = np.array(full_dataset.hf_dataset["episode_index"])
        all_frame_indices = np.arange(len(full_dataset))

        train_mask = np.isin(episode_indices, list(train_episodes))
        train_indices = all_frame_indices[train_mask].tolist()
        val_indices = all_frame_indices[~train_mask].tolist()

        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
        )
        return train_loader, val_loader


if __name__ == "__main__":
    # 1. Setup Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ==========================================
    # TEST 1: Standard Single Observation (N=1)
    # ==========================================
    print("\n" + "=" * 50)
    print("TESTING: obs_horizon = 1, action_horizon = 4")
    print("=" * 50)

    config_single = DataSetConfig(
        repo_id="lerobot/pusht",
        fps=10,
        cameras=["image"],  # PushT's default camera name
        obs_horizon=1,
        action_horizon=4,
        batch_size=2,
        num_workers=0,  # Keep at 0 for clean error stack traces!
        train_fraction=0.9,
    )

    try:
        train_loader_single, _ = LerobotDatasetWrapper.create_loaders(
            config_single, device
        )
        batch_single = next(iter(train_loader_single))

        print("\nBatch Keys and Tensor Shapes (obs_horizon=1):")
        for key, val in batch_single.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key:<25}: {val.shape}")
    except Exception as e:
        print(f"❌ Failed on obs_horizon=1. Error: {e}")

    # ==========================================
    # TEST 2: Multi-Frame Observation (N=3)
    # ==========================================
    print("\n" + "=" * 50)
    print("TESTING: obs_horizon = 3, action_horizon = 4")
    print("=" * 50)

    config_multi = DataSetConfig(
        repo_id="lerobot/pusht",
        fps=10,
        cameras=["image"],
        obs_horizon=3,
        action_horizon=4,
        batch_size=2,
        num_workers=0,
        train_fraction=0.9,
    )

    try:
        train_loader_multi, _ = LerobotDatasetWrapper.create_loaders(
            config_multi, device
        )
        batch_multi = next(iter(train_loader_multi))

        print("\nBatch Keys and Tensor Shapes (obs_horizon=3):")
        for key, val in batch_multi.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key:<25}: {val.shape}")
    except Exception as e:
        print(f"❌ Failed on obs_horizon=3. Error: {e}")
