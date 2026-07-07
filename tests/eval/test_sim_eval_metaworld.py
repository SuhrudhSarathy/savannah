"""Smoke test for evaluate_and_log (the live sim-eval loop the trainer runs
periodically during training) on the MetaWorld task, for both env variants
that have a task config (assembly, button-press). Not testing model quality —
only that the rollout loop runs to completion without crashing.
"""

import gymnasium as gym
import pytest

from savannah.data.dataset import DataSetConfig
from savannah.tasks.metaworld import MetaworldTask
from savannah.utils.device import get_device
from savannah.utils.eval_and_log import evaluate_and_log
from savannah.utils.log import setup_logging
from savannah.utils.logger import ExperimentLogger

from policy_helpers import build_tiny_policy

setup_logging(level="WARNING")

ACTION_HORIZON = 4
STATE_DIM = 4
ACTION_DIM = 4
NUM_CAMERAS = 1

# Cap episode length so the smoke test stays fast — Meta-World episodes can
# otherwise run for hundreds of steps. We only care that the loop doesn't crash.
MAX_EPISODE_STEPS = 12


@pytest.mark.parametrize(
    "env_name", ["assembly-v3", "button-press-v3"], ids=["assembly", "button_press"]
)
def test_sim_eval_loop_does_not_crash(env_name):
    device = get_device()

    config = DataSetConfig(
        repo_id=f"suhrudhsarathy/metaworld-{env_name}",  # unused — get_env() doesn't touch the dataset
        env_name=env_name,
        fps=20,
        cameras=["topview"],
        obs_horizon=1,
        action_horizon=ACTION_HORIZON,
        image_size=64,
    )
    task = MetaworldTask(config=config, device=device)

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
