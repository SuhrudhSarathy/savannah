from savannah.tasks import RecordedEnv
import os
import sys
from collections import deque
from dataclasses import asdict

import gym_pusht
import gymnasium as gym
import torch
import numpy as np

import wandb
from savannah.utils.device import get_device
from savannah.utils.observation import ObservationKey

# --- Python 3.12 Compatibility Patch ---
try:
    import imp
except ImportError:
    import importlib.util
    from types import ModuleType

    imp = ModuleType("imp")
    sys.modules["imp"] = imp

from savannah.data.dataset import DataSetConfig
from savannah.models.backbones.resnet_backbone import ResNetBackbone
from savannah.models.flow_matching import FlowMatchingPolicy
from savannah.models.vision_encoder import VisionEncoder
from savannah.tasks.pusht import PushTTask

from tqdm import tqdm


def load_policy(checkpoint_path, device) -> FlowMatchingPolicy:
    embed_dim = 128
    num_attn_heads = 4
    num_blocks = 6
    feedforward_dim = 4 * embed_dim
    num_cameras = 3
    state_dim = 2
    action_dim = 2
    action_horizon = 8
    obs_horizon = 3

    resnet_backbone = ResNetBackbone(out_channels=96).to(device)
    vision_encoder = VisionEncoder(backbone=resnet_backbone, embed_dim=embed_dim)
    policy = FlowMatchingPolicy(
        embed_dim,
        num_attn_heads,
        feedforward_dim,
        num_blocks,
        state_dim,
        action_dim,
        action_horizon,
        vision_encoder,
        num_cameras,
        flowmatching_inference_steps=50,
    ).to(device)

    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    return policy


def normalize_state(state, center=256, scale=512):
    return (state - center) / scale


def unnormalize_action(action, center=256, scale=512):
    return (action * scale) + center


import cv2


def run_single_episode(checkpoint_path):
    device = get_device()

    obs_horizon = 3
    action_horizon = 8

    policy = load_policy(checkpoint_path, device)

    base_env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        max_episode_steps=300,
    )

    env = RecordedEnv(base_env)

    policy.eval()
    all_episode_videos = []
    success_count = 0
    rewards = []

    max_episodes = 3

    for ep in tqdm(range(max_episodes)):
        obs, info = env.reset()
        img_history = deque([obs["pixels"]] * obs_horizon, maxlen=obs_horizon)
        state_history = deque([obs["agent_pos"]] * obs_horizon, maxlen=obs_horizon)

        done = False
        cum_reward = 0.0

        while not done:
            frame = env.render()

            imgs = np.stack(img_history)
            img_tensor = (
                torch.from_numpy(imgs)
                .permute(0, 3, 1, 2)  # (3, 3, 96, 96)
                .float()
                .divide(255.0)
                .unsqueeze(0)
                .to(device)  # (1, 3, 3, 96, 96)
            )

            states = np.stack(state_history)  # (2, 2)
            norm_states = normalize_state(states)
            state_tensor = (
                torch.from_numpy(norm_states)
                .float()
                .unsqueeze(0)
                .to(device)  # (1, 2, 2)
            )

            obs = {}
            obs[ObservationKey.state] = state_tensor
            obs[ObservationKey.images] = [img_tensor]

            action_chunk = policy.compute_action(obs).actions[0].cpu().numpy()

            for i in range(action_horizon):
                if done:
                    break

                action_to_apply = unnormalize_action(action_chunk[i])
                print(f"Action To Apply: {action_to_apply}")
                obs, reward, terminated, truncated, info = env.step(action_to_apply)

                # Update history with the new observation
                img_history.append(obs["pixels"])
                state_history.append(obs["agent_pos"])

                cum_reward += reward

                done = terminated or truncated

        rewards.append(cum_reward)
        if info.get("score", 0) > 0.95:
            success_count += 1

    env.close()

    print(f"Rewards: {rewards}, Success Count: {success_count}")


if __name__ == "__main__":
    ckpt_path = "/Users/suhrudh/savannah/scripts/artifacts/flow_matching_pusht:v0/best_model.ckpt"

    run_single_episode(ckpt_path)
