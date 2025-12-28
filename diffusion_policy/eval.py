import torch
import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import gym_pusht
import wandb
import numpy as np
from tqdm import tqdm
from collections import deque

from diffusion import DiffusionPolicy, DiffusionPolicyConfig
from unet import UNetConfig

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("Using Device: ", device)

def normalize_state(state, center=256, scale=512):
    return (state - center) / scale

def unnormalize_action(action, center=256, scale=512):
    return (action * scale) + center

def eval_epochs(num_episodes: int = 5, action_steps: int = 8, history_len: int = 2):
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")

    # --- Model Loading ---
    checkpoint_path = os.path.join(os.getcwd(), "checkpoints", "best_model.ckpt")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DiffusionPolicy(model_config=UNetConfig(), config=DiffusionPolicyConfig()).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False

        # Initialize buffers for history
        # We fill them with the first observation initially
        img_history = deque([obs["pixels"]] * history_len, maxlen=history_len)
        state_history = deque([obs["agent_pos"]] * history_len, maxlen=history_len)

        while not done:
            # 1. Stack history and Preprocess (B, T, C, H, W) for images
            # and (B, T, D) for state
            imgs = np.stack(img_history) # (2, 96, 96, 3)
            img_tensor = (
                torch.from_numpy(imgs)
                .permute(0, 3, 1, 2) # (2, 3, 96, 96)
                .float().divide(255.0)
                .unsqueeze(0).to(device) # (1, 2, 3, 96, 96)
            )

            states = np.stack(state_history) # (2, 2)
            norm_states = normalize_state(states)
            state_tensor = (
                torch.from_numpy(norm_states)
                .float()
                .unsqueeze(0).to(device) # (1, 2, 2)
            )

            # 2. Diffusion Inference
            with torch.no_grad():
                # Assuming the model predicts a chunk of 16 steps
                noise = torch.randn((1, 16, 2), device=device)
                action_chunk = model.get_action(noise, img_tensor, state_tensor, 20)
                action_chunk = action_chunk.cpu().numpy()[0]

            # 3. Execute 'action_steps'
            for i in range(action_steps):
                if done: break

                action_to_apply = unnormalize_action(action_chunk[i])
                obs, reward, terminated, truncated, info = env.step(action_to_apply)

                # Update history with the new observation
                img_history.append(obs["pixels"])
                state_history.append(obs["agent_pos"])

                done = terminated or truncated

    env.close()

if __name__ == "__main__":
    # action_steps=8 is a common choice for Diffusion Policy to ensure temporal smoothness
    eval_epochs(num_episodes=5, action_steps=8)
