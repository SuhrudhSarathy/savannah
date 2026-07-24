from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import torch

from savannah.models import Policy
from savannah.tasks import BaseRobotTask
from savannah.utils.action_buffer import ActionBuffer


def _to_hwc_frame(frame) -> np.ndarray:
    """Normalizes env.render() output to a plain (H, W, C) numpy array.

    Most gym envs (gym_pusht, MetaWorld) already return this. ManiSkill's
    vectorized envs instead return a batched torch.Tensor of shape
    (num_envs, H, W, C) — squeeze the env dim and move it off-device.
    """
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    return frame


@dataclass
class EpisodeResult:
    reward: float
    success: bool
    info: dict[str, Any]
    frames: Optional[np.ndarray] = None  # (T, C, H, W), only if captured


@dataclass
class EvalResult:
    success_rate: float
    mean_reward: float
    episodes: list[EpisodeResult] = field(default_factory=list)


def run_rollout(
    policy: Policy,
    task: BaseRobotTask,
    env,
    num_episodes: int = 3,
    execute_steps: int = 8,
    obs_horizon: int | None = None,
    capture_frames: bool = True,
    reset_kwargs: Optional[dict] = None,
    on_episode_start: Optional[Callable[[int], None]] = None,
    on_chunk: Optional[Callable[[torch.Tensor], None]] = None,
    on_step: Optional[
        Callable[[int, np.ndarray, np.ndarray, dict, bool, bool], None]
    ] = None,
    on_episode_end: Optional[Callable[[int, EpisodeResult], None]] = None,
) -> EvalResult:
    """
    Rolls the policy out on `env` for `num_episodes` using receding horizon
    control (action chunking). This is the single source of truth for the
    eval loop — the trainer's periodic sim-eval and the standalone eval
    script both call this and then log the resulting `EvalResult` however
    fits their setup (W&B vs. console, video-in-memory vs. video-to-disk).

    `on_chunk`, `on_step` and `on_episode_end` are optional hooks for callers
    that want to observe the rollout (e.g. driving a video-recording env
    wrapper, or emitting console logs) without duplicating the loop itself.
    """
    if obs_horizon is None:
        obs_horizon = task.config.obs_horizon
    reset_kwargs = reset_kwargs or {}

    was_training = policy.training
    policy.eval()

    episodes: list[EpisodeResult] = []
    for ep in range(num_episodes):
        if on_episode_start is not None:
            on_episode_start(ep)

        raw_obs, info = task.reset_env(env, **reset_kwargs)
        frames = [] if capture_frames else None
        done = False
        cum_reward = 0.0
        success = False

        # CRITICAL: Reset the buffer for every new episode!
        action_buffer = ActionBuffer(execute_steps=execute_steps)
        obs_history = deque([raw_obs] * obs_horizon, maxlen=obs_horizon)

        while not done:
            if capture_frames:
                frame = _to_hwc_frame(env.render())
                frames.append(frame.transpose(2, 0, 1))

            # Only query the heavy neural network if we are out of actions!
            if action_buffer.is_empty():
                # Task handles all normalizations and tensor casting
                obs_dict = task.preprocess_observation_history(obs_history)

                with torch.no_grad():
                    policy_out = policy.compute_action(obs_dict)

                if on_chunk is not None:
                    on_chunk(policy_out.actions)

                # policy_out.actions shape: (B, T, action_dim).
                # Take Batch 0 and push it to the buffer.
                action_buffer.push(policy_out.actions[0])

            raw_action = torch.clamp(action_buffer.pop(), -1, 1)
            action_to_apply = task.postprocess_action(raw_action)

            raw_obs, reward, terminated, truncated, info = env.step(action_to_apply)
            obs_history.append(raw_obs)

            cum_reward += reward
            success = task.is_success(info)
            done = terminated or truncated or success

            if on_step is not None:
                on_step(ep, raw_action, action_to_apply, info, terminated, truncated)

        result = EpisodeResult(
            reward=cum_reward,
            success=success,
            info=info,
            frames=np.stack(frames) if capture_frames else None,
        )
        episodes.append(result)
        if on_episode_end is not None:
            on_episode_end(ep, result)

    if was_training:
        policy.train()

    success_rate = sum(e.success for e in episodes) / num_episodes
    mean_reward = float(np.mean([e.reward for e in episodes]))
    return EvalResult(
        success_rate=success_rate, mean_reward=mean_reward, episodes=episodes
    )
