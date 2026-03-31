from .base import BaseRobotTask

from gymnasium.wrappers import RecordVideo
import gymnasium as gym


class RecordedEnv:
    """
    A thin wrapper around a Gym env that records episodes to video.
    Works with rgb_array render mode — no display required.
    """

    def __init__(
        self,
        env: gym.Env,
        video_folder: str = "./videos",
        name_prefix: str = "eval",
        episode_trigger=None,  # defaults to recording every episode
    ):
        self.env = RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=episode_trigger or (lambda _: True),
            name_prefix=name_prefix,
        )

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def __getattr__(self, name):
        # Fall through to the wrapped env for anything else
        return getattr(self.env, name)
