from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import v2

from savannah.data.augmentation import AugmentationConfig
from savannah.utils import ObservationKey


@dataclass
class DataSetConfig:
    repo_id: str
    fps: int
    root: Optional[str] = None
    env_name: Optional[str] = None
    cameras: List[str] = field(default_factory=list)

    # Temporal horizons
    obs_horizon: int = 1
    action_horizon: int = 10

    # Image preprocessing
    image_size: Optional[int] = None  # resize camera images to (image_size, image_size)

    # Dataloader specifics
    batch_size: int = 8
    num_workers: int = 4
    train_fraction: float = 0.95

    # Cache decoded (and resized) samples in memory after first access,
    # so each sample is only decoded once across all epochs.
    cache_in_memory: bool = False

    # Expose LeRobot's per-frame natural-language task description under
    # ObservationKey.language, for language-conditioned policies.
    use_language: bool = False

    # Train-time image/state augmentation (crop, color jitter, speckle noise,
    # state noise), disabled by default. See savannah.data.augmentation.
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


cs = ConfigStore.instance()
cs.store(name="dataset_config", node=DataSetConfig)


class _CachedDataset(Dataset):
    """Wraps a dataset and caches the result of __getitem__ in memory.

    Each worker process holds its own cache, but with persistent_workers=True
    the cache survives across epochs, so repeated decodes of the same sample
    are avoided after the first pass.

    Image tensors (float32 in [0, 1]) are cached as uint8 to cut cache memory
    4x, and converted back to float32 on retrieval.
    """

    def __init__(self, dataset: Dataset, image_keys: set[str]):
        self.dataset = dataset
        self.image_keys = image_keys
        self._cache: dict = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx not in self._cache:
            item = self.dataset[idx]
            cached = dict(item)
            for key in self.image_keys:
                if key in cached:
                    cached[key] = (cached[key] * 255).round().to(torch.uint8)
            self._cache[idx] = cached

        item = dict(self._cache[idx])
        for key in self.image_keys:
            if key in item:
                item[key] = item[key].to(torch.float32) / 255.0
        return item


class _LanguageKeyDataset(Dataset):
    """Exposes LeRobot's per-frame `task` string under ObservationKey.language.

    LeRobotDataset always populates `item["task"]` (the natural-language task
    description for the frame's episode) regardless of delta_timestamps, so
    this is just a rename — no extra decoding cost.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = dict(self.dataset[idx])
        item[ObservationKey.language] = item["task"]
        return item


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

        image_transforms = None
        if config.image_size is not None:
            image_transforms = v2.Resize(
                (config.image_size, config.image_size), antialias=True
            )

        full_dataset = LeRobotDataset(
            config.repo_id,
            delta_timestamps=deltas,
            image_transforms=image_transforms,
            video_backend="torchcodec",
        )

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

        if config.use_language:
            train_dataset = _LanguageKeyDataset(train_dataset)
            val_dataset = _LanguageKeyDataset(val_dataset)

        if config.cache_in_memory:
            image_keys = {
                k
                for k in deltas
                if k != ObservationKey.state and k != ObservationKey.actions
            }
            train_dataset = _CachedDataset(train_dataset, image_keys)
            val_dataset = _CachedDataset(val_dataset, image_keys)

        loader_kwargs = {}
        if config.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 4

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            **loader_kwargs,
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

    # ==========================================
    # TEST 3: Language-Conditioned Batch
    # ==========================================
    print("\n" + "=" * 50)
    print("TESTING: use_language = True")
    print("=" * 50)

    config_language = DataSetConfig(
        repo_id="lerobot/pusht",
        fps=10,
        cameras=["image"],
        obs_horizon=1,
        action_horizon=4,
        batch_size=2,
        num_workers=0,
        train_fraction=0.9,
        use_language=True,
    )

    try:
        train_loader_language, _ = LerobotDatasetWrapper.create_loaders(
            config_language, device
        )
        batch_language = next(iter(train_loader_language))

        print("\nBatch Keys (use_language=True):")
        for key, val in batch_language.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key:<25}: {val.shape}")
            else:
                print(f"  {key:<25}: {val}")
    except Exception as e:
        print(f"❌ Failed on use_language=True. Error: {e}")
