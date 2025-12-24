import torch
from model import ACTPolicy, ACTConfig
from temporal_ensemble import ACTTemporalEnsembler
import os

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import gym_pusht
import wandb
import numpy as np


dtype = torch.float32
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
    torch.mps.empty_cache()
    torch.mps.synchronize()
print("Currently using device: ", device)


def get_model() -> ACTPolicy:
    if "checkpoints" in os.listdir(os.getcwd()):
        checkpoint_path = os.path.join(os.getcwd(), "checkpoints", "best_model.ckpt")
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        config = checkpoint["config"]
        model = ACTPolicy(config)

        model.load_state_dict(checkpoint["model_state_dict"])

        return model

    else:
        raise AssertionError("Checkpoint model does not exist")


def normalize_action(action, center=256, scale=512):
    """Normalizes raw [0, 512] coordinates to [-1, 1]"""
    return (action - center) / scale


def unnormalize_action(action, center=256, scale=512):
    """Unnormalizes model output [-1, 1] back to [0, 512]"""
    return (action * scale) + center


def evaluate_and_log(model, device, idx, num_episodes=3):
    # Use rgb_array so we can grab the frames for W&B
    env = gym.make(
        "gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array"
    )

    model.eval()
    all_episode_videos = []
    success_count = 0
    rewards = []

    temporal_ensembler = ACTTemporalEnsembler(0.001, chunk_size=10)

    for i in range(num_episodes):
        obs, info = env.reset()
        frames = []
        done = False

        cum_reward = 0.0
        while not done:
            # Render and collect frame
            frame = env.render()
            frames.append(frame.transpose(2, 0, 1))  # W&B expects (C, H, W)

            img_tensor = (
                torch.from_numpy(obs["pixels"])
                .permute(2, 0, 1)
                .float()
                .divide(255.0)
                .unsqueeze(0)
                .to(device)
            )

            raw_state = obs["agent_pos"]
            norm_state = normalize_action(raw_state)
            state_tensor = (
                torch.from_numpy(norm_state)
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )

            with torch.no_grad():
                action_chunk, _ = model(
                    torch.zeros_like(img_tensor), torch.zeros_like(state_tensor), None
                )

            ensemble_action = temporal_ensembler.update(action_chunk)
            pred_action_norm = (
                ensemble_action.cpu()
                .numpy()
                .reshape(
                    -1,
                )
            )
            action_to_apply = unnormalize_action(pred_action_norm)

            # 7. Step the environment
            obs, reward, terminated, truncated, info = env.step(action_to_apply)

            cum_reward += reward
            done = terminated or truncated

        rewards.append(cum_reward)

        if info.get("score", 0) > 0.95:
            success_count += 1

        all_episode_videos.append(np.stack(frames))

    # 2. Log to WandB
    # We log the first video and the success rate
    wandb.log(
        {
            "eval/success_rate": success_count / num_episodes,
            "eval/video": wandb.Video(all_episode_videos[0], fps=10, format="mp4"),
            "eval/reward": np.mean(np.array(cum_reward)).item(),
            "idx": idx,
        }
    )

    env.close()
    model.train()  # Switch back to training mode!


def eval_one_epoch():
    # 1. Initialize Env (pixels_agent_pos gives you the image AND (x,y) state)
    base_env = gym.make(
        "gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array"
    )
    env = RecordVideo(
        base_env,
        video_folder="./videos",
        episode_trigger=lambda x: True,
        name_prefix="act_eval",
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = get_model().to(device).eval()

    obs, info = env.reset()
    done = False

    temporal_ensembler = ACTTemporalEnsembler(0.001, chunk_size=10)

    while not done:
        # 2. Extract and Preprocess Image (96x96x3 -> 1x3x96x96)
        # We permute to get Channels-First and normalize pixels to [0, 1]
        img_tensor = (
            torch.from_numpy(obs["pixels"])
            .permute(2, 0, 1)
            .float()
            .divide(255.0)
            .unsqueeze(0)
            .to(device)
        )

        raw_state = obs["agent_pos"]
        norm_state = normalize_action(raw_state)
        state_tensor = (
            torch.from_numpy(norm_state).float().unsqueeze(0).unsqueeze(0).to(device)
        )

        # 4. Inference
        with torch.no_grad():
            action_chunk, _ = model(
                img_tensor, state_tensor, None
            )

        ensemble_action = temporal_ensembler.update(action_chunk)
        # ensemble_action = action_chunk[0, 0, :]

        # 5. Process the first action in the chunk
        pred_action_norm = (
            ensemble_action.cpu()
            .numpy()
            .reshape(
                -1,
            )
        )

        # 6. Unnormalize back to [0, 512] for the simulator
        action_to_apply = unnormalize_action(pred_action_norm)

        # 7. Step the environment
        obs, reward, terminated, truncated, info = env.step(action_to_apply)
        done = terminated or truncated

    env.close()

    print("Episode Finished. Success:", info.get("is_success", False))

if __name__ == "__main__":
    eval_one_epoch()