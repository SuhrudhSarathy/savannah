"""Smoke test for evaluate_and_log (the live sim-eval loop the trainer runs
periodically during training) on the PushT task. Not testing model quality —
only that the rollout loop runs to completion without crashing.
"""

import gymnasium as gym
import pytest
from policy_helpers import build_tiny_policy

from savannah.data.dataset import DataSetConfig
from savannah.tasks.pusht import PushTTask
from savannah.utils.device import get_device
from savannah.utils.eval_and_log import evaluate_and_log
from savannah.utils.log import setup_logging
from savannah.utils.logger import ExperimentLogger

setup_logging(level="WARNING")

ACTION_HORIZON = 4
STATE_DIM = 2
ACTION_DIM = 2
NUM_CAMERAS = 1

# Cap episode length so the smoke test stays fast regardless of PushT's
# default 300-step limit — we only care that the loop doesn't crash.
MAX_EPISODE_STEPS = 12


def test_sim_eval_loop_does_not_crash():
    device = get_device()

    config = DataSetConfig(
        repo_id="lerobot/pusht",  # unused here — get_env() doesn't touch the dataset
        fps=10,
        cameras=["image"],
        obs_horizon=1,
        action_horizon=ACTION_HORIZON,
    )
    task = PushTTask(config=config, device=device)

    # get_env() lazily creates + caches task._env; wrap that cached instance
    # so evaluate_and_log's own task.get_env() call reuses our short-episode env.
    base_env = task.get_env(render_mode="rgb_array")
    task._env = gym.wrappers.TimeLimit(base_env, max_episode_steps=MAX_EPISODE_STEPS)

    policy = build_tiny_policy(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        num_cameras=NUM_CAMERAS,
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
