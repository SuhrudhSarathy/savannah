import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import cv2
import numpy as np

from .base import BaseRobotTask


class ActionChunkVisualizer(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_chunk = None

    def set_action_chunk(self, chunk: np.ndarray):
        """Expects chunk of shape (N, 2) for x, y coordinates."""
        self.action_chunk = chunk

    def render(self):
        frame = self.env.render()
        if frame is None or self.action_chunk is None:
            return frame

        frame = frame.copy()
        H, W = frame.shape[:2]  # Height, Width

        # Ensure action_chunk is (horizon, 2)
        actions = self.action_chunk
        print("[RENDER]: Actions: ", actions)

        # # Match the scaling from your source:
        # # (action / 512) * [H, W]
        # # We use np.array([H, W]) to ensure element-wise multiplication matches the source
        # scaled_actions = (actions / 512.0) * np.array([H, W])
        # # Clamp to frame bounds: an undertrained policy can emit huge/NaN
        # # values that overflow the C int cv2.circle expects for 'center'.
        # scaled_actions = np.nan_to_num(scaled_actions, nan=0.0, posinf=0.0, neginf=0.0)
        # scaled_actions = np.clip(scaled_actions, [0, 0], [H - 1, W - 1])
        # scaled_actions = np.round(scaled_actions).astype(int)

        num_steps = len(actions)

        # We iterate backwards just like your source: actions[::-1]
        for k, action in enumerate(actions[::-1]):
            # IMPORTANT: The source logic implies action[0] is Y and action[1] is X.
            # OpenCV needs (column, row) which is (X, Y).
            # So we pass (action[1], action[0])
            center = (int(action[0]), int(action[1]))

            print("[RENDER] Action: ", action[0], action[1])

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
