import gymnasium as gym
import torch

import savannah.envs  # noqa: F401 — triggers @register_env
from savannah.envs import VectorizedScriptedPolicy

if __name__ == "__main__":
    num_environments = 1
    env = gym.make(
        "ColorMatching-v1",
        num_envs=num_environments,
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode="human",
    )

    choice = ["red", "blue", "green"]

    for i in range(5):
        pick_color = choice[torch.randint(0, 2, (1,))]
        place_color = choice[torch.randint(0, 2, (1,))]

        obs, info = env.reset(options={"pick_color": pick_color, "place_color": place_color})
        raw_env = env.unwrapped
        policy = VectorizedScriptedPolicy(
            num_envs=num_environments,
            device=raw_env.device,
            pick_color=pick_color,
            place_color=place_color,
        )

        for step in range(500):
            action = policy.act(raw_env)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            if info["success"].any():
                print(f"Success accomplished at step {step}!")
                break

    env.close()
