from collections import deque

import numpy as np
import pytest
import torch

from savannah.data.dataset import DataSetConfig
from savannah.tasks.franka_sim import ACTION_KEY, MASK_CAMERA, FrankaSimTask
from savannah.utils.device import get_device
from savannah.utils.observation import ObservationKey

REPO_ID = "suhrudhsarathy/franka_sim_kitting_long"
CAMERAS = ["front", "wrist"]
INSTRUCTION = "put all the parts in the kit tray"
OBS_HORIZON = 2
ACTION_HORIZON = 8
IMAGE_SIZE = 128

# Smallest single-target export, i.e. one with observation.images.target_mask.
MASK_REPO_ID = "suhrudhsarathy/franka_sim_buttons"


def make_config(**overrides) -> DataSetConfig:
    kwargs = {
        "repo_id": REPO_ID,
        "env_name": "kitting_long",
        "fps": 20,
        "cameras": CAMERAS,
        "action_key": ACTION_KEY,
        "image_size": IMAGE_SIZE,
        "obs_horizon": OBS_HORIZON,
        "action_horizon": ACTION_HORIZON,
        "batch_size": 4,
        "num_workers": 0,
        "use_language": True,
    }
    return DataSetConfig(**{**kwargs, **overrides})


@pytest.fixture(scope="module")
def task():
    """Shared across tests in this module — avoids re-downloading."""
    return FrankaSimTask(make_config(), get_device())


@pytest.fixture(scope="module")
def mask_task():
    config = make_config(
        repo_id=MASK_REPO_ID,
        env_name="buttons",
        cameras=[*CAMERAS, MASK_CAMERA],
        use_language=False,
    )
    return FrankaSimTask(config, get_device())


def test_requires_env_name():
    config = DataSetConfig(repo_id=REPO_ID, fps=20, action_key=ACTION_KEY)
    with pytest.raises(ValueError, match="env_name"):
        FrankaSimTask(config, torch.device("cpu"))


def test_norm_stats_from_hub_metadata(task):
    stats = task._get_norm_stats()
    assert stats["low"].shape == (8,)
    assert np.all(stats["high"] > stats["low"])
    assert (stats["low"][7], stats["high"][7]) == (0.0, 1.0)  # gripper
    # Top-down gripper: the commanded qw / qz never change.
    assert np.flatnonzero(stats["const"]).tolist() == [3, 6]


def test_format_batch_shapes_and_normalization(task):
    loader = task.get_train_loader()
    batch = next(iter(loader))
    formatted = task.format_batch(batch, step=None)

    B = loader.batch_size
    assert len(formatted[ObservationKey.images]) == len(CAMERAS)
    for img in formatted[ObservationKey.images]:
        assert img.shape == (B, OBS_HORIZON, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert formatted[ObservationKey.state].shape == (B, OBS_HORIZON, 8)
    assert formatted[ObservationKey.gt_actions].shape == (B, ACTION_HORIZON, 8)
    assert formatted[ObservationKey.language][0] == INSTRUCTION

    # Actions are normalized within [-1, 1]; the measured-pose state may
    # stray slightly outside the command range.
    assert formatted[ObservationKey.gt_actions].abs().max() <= 1.0 + 1e-5
    assert formatted[ObservationKey.state].abs().max() <= 1.5

    # The first target is the command at offset 0 (after the OBS_HORIZON
    # past commands), and postprocess_action inverts its normalization.
    raw = batch[ACTION_KEY][0, OBS_HORIZON].numpy()
    recovered = task.postprocess_action(formatted[ObservationKey.gt_actions][0, 0])
    np.testing.assert_allclose(recovered, raw, atol=1e-4)


def test_episode_start_state_has_open_gripper(mask_task):
    """At an episode's first frame LeRobot pads the previous commands
    (`*_is_pad`); the state's gripper must read 0 (open), as after reset."""
    batch = next(iter(mask_task.get_episode_loader(episode_index=0, frame_index=0)))
    state = mask_task.format_batch(batch)[ObservationKey.state]
    open_gripper = mask_task._normalize(torch.zeros(8))[7]
    torch.testing.assert_close(state[0, :, 7], open_gripper.expand(OBS_HORIZON))


def test_missing_mask_is_reported():
    task = FrankaSimTask(
        make_config(cameras=[*CAMERAS, MASK_CAMERA]), torch.device("cpu")
    )
    with pytest.raises(ValueError, match="single-target"):
        task.get_train_loader()


def test_format_batch_includes_target_mask_stream(mask_task):
    batch = next(iter(mask_task.get_train_loader()))
    images = mask_task.format_batch(batch)[ObservationKey.images]

    assert len(images) == 3
    mask = images[-1]
    assert mask.shape[2:] == (3, IMAGE_SIZE, IMAGE_SIZE)
    # Binary mask, softened only at the edges by the resize.
    assert mask.min() >= 0.0 and mask.max() <= 1.0
    assert (mask > 0.5).any(), "the target should be visible from the top camera"


def test_env_rollout_step(task):
    pytest.importorskip("franka_sim")
    env = task.get_env()
    obs, _ = task.reset_env(env, pick_color=None)  # foreign kwargs are ignored
    assert task._instruction == INSTRUCTION
    assert obs["gripper_command"].tolist() == [0.0]

    history = deque([obs] * OBS_HORIZON, maxlen=OBS_HORIZON)
    processed = task.preprocess_observation_history(history)
    assert processed[ObservationKey.state].shape == (1, OBS_HORIZON, 8)
    for img in processed[ObservationKey.images]:
        assert img.shape == (1, OBS_HORIZON, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert processed[ObservationKey.language] == [INSTRUCTION]

    # Commanding the current end-effector pose (the state, unnormalized)
    # holds the arm in place. The command is exactly top-down (qw = qz = 0,
    # like every demo command) while the arm starts slightly tilted, so it
    # straightens over the first steps (a few mm of transient) and settles.
    hold = task.postprocess_action(processed[ObservationKey.state][0, -1])
    hold[7] = 0.5
    for _ in range(10):
        obs, _, _, _, info = env.step(hold)
    np.testing.assert_allclose(obs["ee_pos"], hold[:3], atol=2e-3)
    assert obs["gripper_command"].tolist() == [0.5]
    assert not task.is_success(info)
    assert "progress: 0/4" in task.get_metrics(info)


def test_env_adds_target_mask(mask_task):
    pytest.importorskip("franka_sim")
    env = mask_task.get_env()
    obs, _ = mask_task.reset_env(env)
    mask = obs[f"rgb_{MASK_CAMERA}"]
    assert mask.dtype == np.uint8 and set(np.unique(mask)) <= {0, 255}
    assert mask.max() == 255

    processed = mask_task.preprocess_observation_history(deque([obs] * OBS_HORIZON))
    assert len(processed[ObservationKey.images]) == 3
