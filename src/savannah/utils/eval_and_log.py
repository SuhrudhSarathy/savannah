import torch
from savannah.utils.action_buffer import ActionBuffer
from savannah.utils.logger import ExperimentLogger
from savannah.models import Policy
from savannah.tasks import BaseRobotTask

import numpy as np


def evaluate_and_log(
    policy: Policy,
    task: BaseRobotTask,
    logger: ExperimentLogger,
    step: int,
    num_episodes: int = 3,
    execute_steps: int = 8,
):
    """
    Evaluates the policy using receding horizon control (action chunking).
    """
    env = task.get_env(render_mode="rgb_array")

    # Put model in inference mode
    policy.eval()

    all_episode_videos = []
    success_count = 0
    rewards = []

    for i in range(num_episodes):
        raw_obs, info = env.reset()
        frames = []
        done = False
        cum_reward = 0.0

        # CRITICAL: Reset the buffer for every new episode!
        action_buffer = ActionBuffer(execute_steps=execute_steps)

        while not done:
            # 1. Render and collect frame for logging (W&B expects C, H, W)
            frame = env.render()
            frames.append(frame.transpose(2, 0, 1))

            # 2. Only query the heavy neural network if we are out of actions!
            if action_buffer.is_empty():
                # Task handles all normalizations and tensor casting
                obs_dict = task.preprocess_observation(raw_obs)

                with torch.no_grad():
                    policy_out = policy.compute_action(obs_dict)

                # policy_out.actions shape: (B, T, action_dim).
                # Take Batch 0 and push it to the buffer.
                action_buffer.push(policy_out.actions[0])

            # 3. Pop the next immediate action from the queue
            raw_action = action_buffer.pop()

            # 4. Task handles un-normalization back to physics values
            action_to_apply = task.postprocess_action(raw_action)

            # 5. Step the environment
            raw_obs, reward, terminated, truncated, info = env.step(action_to_apply)

            cum_reward += reward
            done = terminated or truncated

        # Tally metrics for this episode
        rewards.append(cum_reward)
        if info.get("success", False) or info.get("score", 0.0) > 0.95:
            success_count += 1

        all_episode_videos.append(np.stack(frames))

    # ------- Logging
    #
    # 1. Log the metrics
    logger.log_scalars(
        {
            "eval/success_rate": success_count / num_episodes,
            "eval/mean_reward": np.mean(rewards).item(),
        },
        step=step,
    )

    # 2. Log Video (Just log the first episode to save W&B bandwidth)
    if len(all_episode_videos) > 0:
        logger.log_video(
            video_array=all_episode_videos[0],
            metric_name="eval/video",
            step=step,
            fps=10,
        )

    # Switch back to training mode
    policy.train()

    return success_count / num_episodes, np.mean(rewards).item()
