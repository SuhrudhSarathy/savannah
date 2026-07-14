import pytest

from savannah.data.dataset import DataSetConfig, LerobotDatasetWrapper
from savannah.utils.device import get_device

OBS_HORIZON = 3
ACTION_HORIZON = 8


@pytest.fixture(scope="module")
def loaders():
    """Shared across tests in this module — avoids re-downloading."""
    config = DataSetConfig(
        repo_id="lerobot/pusht",
        fps=10,
        cameras=["image"],
        obs_horizon=OBS_HORIZON,
        action_horizon=ACTION_HORIZON,
    )
    device = get_device()
    return LerobotDatasetWrapper.create_loaders(config, device)


def test_loaders_created(loaders):
    train_loader, val_loader = loaders
    assert train_loader is not None
    assert val_loader is not None


def test_batch_shape(loaders):
    train_loader, _ = loaders
    batch = next(iter(train_loader))

    B = train_loader.batch_size

    assert "observation.state" in batch
    assert "action" in batch
    assert "observation.image" in batch

    assert batch["action"].shape == (B, ACTION_HORIZON, 2)
    assert batch["observation.state"].shape == (B, OBS_HORIZON, 2)
    # Images must have the temporal dimension matching obs_horizon so train == eval
    assert batch["observation.image"].shape[1] == OBS_HORIZON


def test_no_episode_leakage(loaders):
    """Ensure train and val episodes are disjoint."""
    train_loader, val_loader = loaders

    train_episodes = set(
        train_loader.dataset.dataset.hf_dataset["episode_index"][
            list(train_loader.dataset.indices)
        ]
    )
    val_episodes = set(
        val_loader.dataset.dataset.hf_dataset["episode_index"][
            list(val_loader.dataset.indices)
        ]
    )

    assert train_episodes.isdisjoint(val_episodes), (
        "Episode leak between train and val!"
    )


@pytest.fixture(scope="module")
def custom_dataset_loader():
    """Shared across tests in this module — avoids re-downloading."""
    config = DataSetConfig(
        repo_id="suhrudhsarathy/metaworld-assembly-v3",
        fps=20,
        cameras=[
            "observation.images.topview",
            "observation.images.gripperPOV",
            "observation.images.behindGripper",
        ],
    )
    device = get_device()
    return LerobotDatasetWrapper.create_loaders(config, device)


def test_loaders_created_for_custom_dataset(custom_dataset_loader):
    train_loader, val_loader = custom_dataset_loader
    assert train_loader is not None
    assert val_loader is not None
