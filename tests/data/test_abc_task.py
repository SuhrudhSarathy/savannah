import pytest
import torch

from savannah.data.dataset import DataSetConfig
from savannah.tasks.abc import ABCPutBottlesTask
from savannah.utils.device import get_device
from savannah.utils.observation import ObservationKey

REPO_ID = "suhrudhsarathy/abc-put-bottle"
CAMERAS = ["top", "left", "right"]
OBS_HORIZON = 2
ACTION_HORIZON = 8


@pytest.fixture(scope="module")
def task():
    """Shared across tests in this module — avoids re-downloading."""
    config = DataSetConfig(
        data_source="lerobot",
        repo_id=REPO_ID,
        fps=30,
        cameras=CAMERAS,
        obs_horizon=OBS_HORIZON,
        action_horizon=ACTION_HORIZON,
        batch_size=4,
        num_workers=0,
        use_language=True,
    )
    return ABCPutBottlesTask(config, get_device())


def test_loaders_created(task):
    train_loader = task.get_train_loader()
    val_loader = task.get_val_loader()
    assert train_loader is not None
    assert val_loader is not None


def test_prompt_from_hub_metadata(task):
    assert task._get_prompt() == "sim_put the plastic bottles in the bin"


def test_norm_stats_from_hub_metadata(task):
    stats = task._get_norm_stats()
    assert stats["state"]["mean"].shape == (14,)
    assert stats["state"]["std"].shape == (14,)
    assert stats["actions"]["mean"].shape == (14,)
    assert stats["actions"]["std"].shape == (14,)


def test_format_batch_shapes_and_normalization(task):
    train_loader = task.get_train_loader()
    batch = next(iter(train_loader))
    formatted = task.format_batch(batch, step=None)

    B = train_loader.batch_size
    assert len(formatted[ObservationKey.images]) == len(CAMERAS)
    for img in formatted[ObservationKey.images]:
        assert img.shape[:2] == (B, OBS_HORIZON)

    assert formatted[ObservationKey.state].shape == (B, OBS_HORIZON, 14)
    assert formatted[ObservationKey.gt_actions].shape == (B, ACTION_HORIZON, 14)
    assert ObservationKey.language in formatted
    assert (
        formatted[ObservationKey.language][0]
        == "sim_put the plastic bottles in the bin"
    )

    # Raw hub values are unnormalized; format_batch must normalize them with
    # the hub's own stats (mean/std from meta/stats.json).
    stats = task._get_norm_stats()
    mean = torch.as_tensor(stats["state"]["mean"])
    std = torch.as_tensor(stats["state"]["std"])
    expected_state = (batch["observation.state"] - mean) / (std + 1e-6)
    torch.testing.assert_close(
        formatted[ObservationKey.state].cpu(), expected_state, atol=1e-4, rtol=1e-4
    )


def test_postprocess_action_unnormalizes(task):
    stats = task._get_norm_stats()
    zero_action = torch.zeros(1, 1, 14)
    action = task.postprocess_action(zero_action)
    # unnormalize(0) == mean, then clipped to the env's action bounds.
    from abc_bottle_gym.env import ACTION_HIGH, ACTION_LOW
    import numpy as np

    expected = np.clip(stats["actions"]["mean"], ACTION_LOW, ACTION_HIGH)
    np.testing.assert_allclose(action, expected, atol=1e-4)
