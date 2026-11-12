"""FrankaSimTask without the Hub or franka_sim: fake dataset metadata and a
fake env. tests/data/test_franka_sim_task.py covers the real dataset/sim."""

from collections import deque
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces

import savannah.tasks.franka_sim as franka_sim
from savannah.data.augmentation import AugmentationConfig, ObservationAugmenter
from savannah.data.dataset import DataSetConfig
from savannah.tasks.franka_sim import (
    ACTION_KEY,
    MASK_CAMERA,
    MASK_KEY,
    POSE_KEY,
    FrankaSimObservation,
    FrankaSimTask,
    canonicalize_quat,
)
from savannah.utils.observation import ObservationKey

# Shaped like franka_sim_kitting's meta/stats.json: qw/qz of the commanded
# ee pose are constant (top-down gripper); gripper commands span [0.2, 1].
EE_LOW = [0.22, -0.36, 0.01, 0.0, 0.70, -0.70, 0.0, 0.2]
EE_HIGH = [0.70, 0.36, 0.58, 0.0, 1.00, 0.70, 0.0, 1.0]
OBS_HORIZON, ACTION_HORIZON = 2, 4


def fake_meta(with_mask: bool):
    features = {
        POSE_KEY: {"shape": [7]},
        ACTION_KEY: {"shape": [8]},
        "observation.images.front": {"shape": [256, 256, 3]},
        "observation.images.wrist": {"shape": [256, 256, 3]},
    }
    if with_mask:
        features[MASK_KEY] = {"shape": [256, 256, 3]}
    stats = {ACTION_KEY: {"min": EE_LOW, "max": EE_HIGH}}
    return SimpleNamespace(features=features, stats=stats)


def make_task(monkeypatch, cameras=("front", "wrist"), with_mask=True, **overrides):
    monkeypatch.setattr(
        franka_sim, "_load_lerobot_meta", lambda repo_id: fake_meta(with_mask)
    )
    config = DataSetConfig(
        **{
            "repo_id": "someone/franka_sim_kitting",
            "env_name": "kitting",
            "fps": 20,
            "cameras": list(cameras),
            "action_key": ACTION_KEY,
            "image_size": 32,
            "obs_horizon": OBS_HORIZON,
            "action_horizon": ACTION_HORIZON,
            **overrides,
        }
    )
    return FrankaSimTask(config, torch.device("cpu"))


def pose(x=0.4, qx=0.9, qy=0.2):
    """A top-down ee pose (qw = qz = 0), quaternion normalized."""
    q = np.array([0.0, qx, qy, 0.0])
    q /= np.linalg.norm(q)
    return np.array([x, 0.1, 0.3, *q], dtype=np.float32)


def test_requires_ee_pose_action_key(monkeypatch):
    with pytest.raises(ValueError, match=r"action\.ee_pose"):
        make_task(monkeypatch, action_key="action.joint_pos")


def test_canonicalize_quat_flips_to_qx_nonnegative():
    p = pose()
    flipped = p.copy()
    flipped[3:7] *= -1
    np.testing.assert_allclose(canonicalize_quat(flipped), p)
    np.testing.assert_allclose(canonicalize_quat(p), p)
    t = torch.from_numpy(np.stack([p, flipped]))
    torch.testing.assert_close(canonicalize_quat(t), torch.from_numpy(np.stack([p, p])))


def test_delta_timestamps_load_previous_commands_before_the_horizon(monkeypatch):
    task = make_task(monkeypatch, cameras=("front", MASK_CAMERA))
    deltas = task.delta_timestamps()
    assert deltas[POSE_KEY] == [-0.05, 0.0]
    assert deltas[MASK_KEY] == [-0.05, 0.0]
    # 2 commands before the obs frames ... then the 4-step action horizon.
    assert deltas[ACTION_KEY] == pytest.approx([-0.1, -0.05, 0.0, 0.05, 0.1, 0.15])
    assert "observation.state" not in deltas


def test_state_and_actions_share_one_normalization(monkeypatch):
    task = make_task(monkeypatch)
    stats = task.normalization_stats()
    assert stats["state"] == stats["actions"]
    # Gripper normalizes over its full [0, 1] range, not the demos' [0.2, 1].
    assert stats["actions"]["low"][7] == 0.0
    assert stats["actions"]["constant_dims"] == [3, 6]

    raw = torch.from_numpy(np.concatenate([pose(), [0.7]]).astype(np.float32))
    state = task.make_state(raw[:7], raw[7:])
    action = task._normalize(raw)
    torch.testing.assert_close(state, action)
    np.testing.assert_allclose(task.postprocess_action(action), raw.numpy(), atol=1e-6)


def test_constant_dims_unnormalize_to_their_value(monkeypatch):
    task = make_task(monkeypatch)
    action = task.postprocess_action(torch.full((8,), 0.3))
    assert action[3] == 0.0 and action[6] == 0.0


def test_actions_are_clipped_to_demo_workspace(monkeypatch):
    task = make_task(monkeypatch)
    high = task.postprocess_action(torch.full((8,), 50.0))
    np.testing.assert_allclose(high[:3], EE_HIGH[:3], rtol=1e-6)
    assert high[7] == 1.0
    low, _ = task.action_limits()
    np.testing.assert_allclose(low[:3], EE_LOW[:3], rtol=1e-6)


def test_missing_mask_feature_is_reported(monkeypatch):
    task = make_task(monkeypatch, cameras=("front", MASK_CAMERA), with_mask=False)
    with pytest.raises(ValueError, match="single-target"):
        task.get_train_loader()


def fake_batch(task, B=2, pad_first=0):
    """Commands with gripper = 0.1 * offset index, so the test can tell which
    command ended up where."""
    T, H = OBS_HORIZON, ACTION_HORIZON
    commands = torch.from_numpy(np.tile(pose(), (B, T + H, 1)))
    gripper = torch.arange(T + H, dtype=torch.float32).mul(0.1).expand(B, -1)
    commands = torch.cat([commands, gripper[..., None]], dim=-1)
    is_pad = torch.zeros(B, T + H, dtype=torch.bool)
    is_pad[:, :pad_first] = True

    measured = torch.from_numpy(np.tile(pose(x=0.5), (B, T, 1)))
    measured[1, 0, 3:7] *= -1  # a flipped measured quaternion
    batch = {POSE_KEY: measured, ACTION_KEY: commands, f"{ACTION_KEY}_is_pad": is_pad}
    for cam in task.config.cameras:
        img = torch.rand(B, T, 3, 32, 32)
        if cam == MASK_CAMERA:
            img = (img > 0.5).float()
        batch[f"observation.images.{cam}"] = img
    return batch


def test_format_batch_builds_state_from_pose_and_previous_gripper_command(
    monkeypatch,
):
    task = make_task(monkeypatch, cameras=("front", "wrist", MASK_CAMERA))
    out = task.format_batch(fake_batch(task, pad_first=1))

    state = out[ObservationKey.state]
    assert state.shape == (2, OBS_HORIZON, 8)
    # Frame 1: the measured pose + the command in force (the previous
    # command, offset index 1 -> gripper 0.1).
    expected = np.concatenate([pose(x=0.5), [0.1]]).astype(np.float32)
    torch.testing.assert_close(
        state[:, 1:], task._normalize(torch.from_numpy(expected).expand(2, 1, 8))
    )
    # Frame 0 precedes the episode's first command: gripper open (0).
    torch.testing.assert_close(
        state[:, 0, 7], task._normalize(torch.zeros(8))[7].expand(2)
    )
    # The flipped measured quaternion is canonicalized.
    torch.testing.assert_close(state[1, 0, :7], state[0, 0, :7])

    actions = out[ObservationKey.gt_actions]
    assert actions.shape == (2, ACTION_HORIZON, 8)
    # The first target is the command at offset 0 (index 2 -> gripper 0.2).
    assert task.postprocess_action(actions[0, 0])[7] == pytest.approx(0.2)
    assert len(out[ObservationKey.images]) == 3


def test_mask_stream_skips_photometric_augmentation(monkeypatch):
    task = make_task(monkeypatch, cameras=("front", MASK_CAMERA))
    task._augmenter = ObservationAugmenter(
        AugmentationConfig(
            enabled=True,
            crop_scale_min_initial=1.0,  # no crop: compare pixels directly
            crop_scale_min_final=1.0,
            color_jitter_initial=0.5,
            color_jitter_final=0.5,
        )
    )
    batch = fake_batch(task)
    rgb, mask = task.format_batch(batch, step=0)[ObservationKey.images]

    assert not torch.allclose(rgb, batch["observation.images.front"])
    torch.testing.assert_close(mask, batch[f"observation.images.{MASK_CAMERA}"])


class FakeFrankaGymEnv(gym.Env):
    """The bits of franka_sim's FrankaGymEnv the observation wrapper touches."""

    def __init__(self, mask):
        self.observation_space = spaces.Dict(
            {"rgb_front": spaces.Box(0, 255, (8, 8, 3), dtype=np.uint8)}
        )
        self.action_space = spaces.Box(-1, 1, (8,), dtype=np.float32)
        self.env = SimpleNamespace(
            render_target_mask=self._render,
            last_command={"gripper": np.zeros(1)},
            task=SimpleNamespace(name="kitting_long", num_steps=4),
        )
        self._mask = mask
        self.render_calls = []

    def _render(self, camera, height, width):
        self.render_calls.append((camera, height, width))
        return self._mask

    def _obs(self):
        p = pose()
        return {
            "rgb_front": np.zeros((8, 8, 3), np.uint8),
            "ee_pos": p[:3],
            "ee_quat": -p[3:7],  # measured quaternions may come flipped
        }

    def reset(self, *, seed=None, options=None):
        self.env.last_command["gripper"][:] = 0.0
        return self._obs(), {"instruction": "put the part in the kit tray"}

    def step(self, action):
        self.env.last_command["gripper"][:] = action[-1]
        return self._obs(), 0.0, False, False, {}


def test_observation_wrapper_adds_gripper_command_and_mask(monkeypatch):
    mask = np.zeros((8, 8), np.uint8)
    mask[2:4, 2:4] = 255
    base = FakeFrankaGymEnv(mask)
    env = FrankaSimObservation(base, mask_size=(8, 8))

    obs0, _ = env.reset(seed=0)
    assert obs0["gripper_command"].tolist() == [0.0]
    assert obs0[f"rgb_{MASK_CAMERA}"].shape == (8, 8, 3)
    assert obs0[f"rgb_{MASK_CAMERA}"][2, 2].tolist() == [255, 255, 255]
    assert obs0[f"rgb_{MASK_CAMERA}"][0, 0].tolist() == [0, 0, 0]

    obs1, *_ = env.step(np.array([*pose(), 0.75], np.float32))
    assert obs1["gripper_command"].tolist() == [0.75]
    assert obs0["gripper_command"].tolist() == [0.0]  # a copy, not a view
    assert base.render_calls == [("top", 8, 8)] * 2

    # The task builds the same state/images the training batch has.
    task = make_task(monkeypatch, cameras=("front", MASK_CAMERA))
    processed = task.preprocess_observation_history(deque([obs0, obs1], maxlen=2))
    raw = np.stack([np.concatenate([pose(), [g]]) for g in (0.0, 0.75)])
    torch.testing.assert_close(
        processed[ObservationKey.state][0],
        task._normalize(torch.from_numpy(raw.astype(np.float32))),
    )
    _, mask_img = processed[ObservationKey.images]
    assert mask_img.shape == (1, 2, 3, 32, 32)
    assert mask_img.max() == pytest.approx(1.0) and mask_img.min() == 0.0


def test_observation_wrapper_without_mask():
    env = FrankaSimObservation(FakeFrankaGymEnv(None))
    obs, _ = env.reset()
    assert f"rgb_{MASK_CAMERA}" not in obs and "gripper_command" in obs


def test_observation_wrapper_rejects_mask_for_tasks_without_a_target():
    env = FrankaSimObservation(FakeFrankaGymEnv(None), mask_size=(8, 8))
    with pytest.raises(ValueError, match="no target mask"):
        env.reset()
