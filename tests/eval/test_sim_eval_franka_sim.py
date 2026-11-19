"""Smoke test for evaluate_and_log (the live sim-eval loop the trainer runs
periodically during training) on franka_sim tasks: a single-target task with
the target-mask stream, and a long-horizon task without one, in both action
modes (ee_pose, joint_pos). Not testing model
quality -- only that the rollout loop runs to completion without crashing,
through FrankaSimTask's whole eval stack (FrankaSimObservation,
FrankaSimWrongTarget, seeded reset_env, both action modes).

franka_sim is installed separately (see the README); skipped without it. The
env reads the dataset's Hub metadata (camera resolution, normalization), but
no episode data.
"""

import gymnasium as gym
import pytest
from policy_helpers import build_tiny_policy

from savannah.data.dataset import DataSetConfig
from savannah.tasks.franka_sim import (
    ACTION_DIM,
    ACTION_KEY,
    JOINT_ACTION_KEY,
    STATE_DIM,
    FrankaSimTask,
)
from savannah.utils.device import get_device
from savannah.utils.eval_and_log import evaluate_and_log
from savannah.utils.log import setup_logging
from savannah.utils.logger import ExperimentLogger

pytest.importorskip("franka_sim")

setup_logging(level="WARNING")

ACTION_HORIZON = 4

# Cap episode length so the smoke test stays fast -- franka_sim episodes run
# for hundreds of steps. We only care that the loop doesn't crash.
MAX_EPISODE_STEPS = 12


@pytest.mark.parametrize(
    "action_key", [ACTION_KEY, JOINT_ACTION_KEY], ids=["ee_pose", "joint_pos"]
)
@pytest.mark.parametrize(
    ("env_name", "cameras"),
    [
        ("hammer_nails", ["front", "wrist", "target_mask"]),
        ("hammer_nails_long", ["front", "wrist"]),
    ],
    ids=["hammer_nails_mask", "hammer_nails_long"],
)
def test_sim_eval_loop_does_not_crash(env_name, cameras, action_key):
    device = get_device()

    config = DataSetConfig(
        repo_id=f"suhrudhsarathy/franka_sim_{env_name}",  # metadata only
        env_name=env_name,
        fps=20,
        cameras=cameras,
        action_key=action_key,
        obs_horizon=1,  # build_tiny_policy is single-frame
        action_horizon=ACTION_HORIZON,
        image_size=64,
    )
    task = FrankaSimTask(config=config, device=device)
    assert task.mode.env_action_mode in ("ee_pose", "joint_pos")

    base_env = task.get_env(render_mode="rgb_array")
    task._env = gym.wrappers.TimeLimit(base_env, max_episode_steps=MAX_EPISODE_STEPS)

    policy = build_tiny_policy(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        num_cameras=len(cameras),
        device=device,
    )
    logger = ExperimentLogger(project_name="test", config={}, enable_logging=False)

    success_rate, mean_reward = evaluate_and_log(
        policy=policy,
        task=task,
        logger=logger,
        step=0,
        num_episodes=1,
        execute_steps=4,
    )

    assert 0.0 <= success_rate <= 1.0
    assert mean_reward == mean_reward  # not NaN

    task.get_env().close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
