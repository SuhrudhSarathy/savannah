import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import cv2
import numpy as np
import torch

from .base import BaseRobotTask
from savannah.utils.log import logger


class ActionChunkVisualizer(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_chunk = None

    def set_action_chunk(self, chunk: np.ndarray):
        """Expects chunk of shape (N, 2) for x, y coordinates."""
        self.action_chunk = chunk

    def render(self):
        frame = self.env.render()
        if frame is None:
            return frame

        if isinstance(frame, torch.Tensor):
            # ManiSkill's vectorized env returns a (num_envs, H, W, C) tensor
            # even for num_envs=1 — RecordVideo needs a plain (H, W, C) array.
            frame = frame[0].detach().cpu().numpy().astype(np.uint8)

        if self.action_chunk is None:
            return frame

        frame = frame.copy()
        H, W = frame.shape[:2]  # Height, Width

        # Ensure action_chunk is (horizon, 2)
        actions = self.action_chunk
        logger.debug("render: action_chunk={}", actions)

        # Actions are in raw env units [0, 512] (same as gym_pusht's
        # action_space and agent_pos). Scale to the actual frame size,
        # matching gym_pusht's own action marker: coord = action/512*[H, W].
        scaled_actions = (actions / 512.0) * np.array([H, W])
        # Clamp to frame bounds: an undertrained policy can emit huge/NaN
        # values that overflow the C int cv2.circle expects for 'center'.
        scaled_actions = np.nan_to_num(scaled_actions, nan=0.0, posinf=0.0, neginf=0.0)
        scaled_actions = np.clip(scaled_actions, [0, 0], [H - 1, W - 1])
        scaled_actions = np.round(scaled_actions).astype(int)

        num_steps = len(scaled_actions)

        # We iterate backwards so the nearest-future action is drawn last (on top).
        for k, action in enumerate(scaled_actions[::-1]):
            # action[0] is X (column), action[1] is Y (row) — same convention
            # gym_pusht uses for its own action marker, no swap needed.
            center = (int(action[0]), int(action[1]))

            logger.debug("render: action=({}, {})", action[0], action[1])

            # Color logic matching your source
            color = (
                int(255 * k / num_steps),
                0,
                int(255 * (num_steps - k) / num_steps),
            )

            cv2.circle(frame, center, radius=2, color=color, thickness=-1)

        return frame


class RecordedEnv:
    def __init__(self, env, video_folder="./videos", name_prefix="eval"):
        # Initialize the wrapper
        self.viz_wrapper = ActionChunkVisualizer(env)

        # Record the visualized output
        self.env = RecordVideo(
            self.viz_wrapper,
            video_folder=video_folder,
            name_prefix=name_prefix,
            episode_trigger=lambda _: True,
        )

    def set_chunk(self, chunk):
        # Access the viz_wrapper directly to update the points
        self.viz_wrapper.set_action_chunk(chunk)

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def close(self):
        self.env.close()
