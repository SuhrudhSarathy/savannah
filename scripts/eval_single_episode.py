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


def load_policy(checkpoint_path, device):
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
        flowmatching_inference_steps=100,
    ).to(device)

    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    return policy


from lerobot.datasets.lerobot_dataset import LeRobotDataset
from savannah.utils import ObservationKey


def run_dataset_episode_eval(checkpoint_path):
    device = get_device()
    obs_horizon = 3
    action_horizon = 8

    dataset_config = DataSetConfig(
        repo_id="lerobot/pusht",
        fps=10,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        batch_size=1,
        cameras=["image"],
    )
    task = PushTTask(config=dataset_config, device=device)
    policy = load_policy(checkpoint_path, device)

    # Build delta_timestamps the same way your LerobotDatasetWrapper does
    obs_timestamps = [
        -(obs_horizon - 1 - i) / dataset_config.fps for i in range(obs_horizon)
    ]
    act_timestamps = [i / dataset_config.fps for i in range(action_horizon)]

    delta_timestamps = {
        "observation.state": obs_timestamps,
        "observation.image": obs_timestamps,
        "action": act_timestamps,
    }

    full_dataset = LeRobotDataset("lerobot/pusht", delta_timestamps=delta_timestamps)

    # Pick an episode
    target_episode = 0
    episode_indices = full_dataset.hf_dataset["episode_index"]
    frame_indices = [i for i, ep in enumerate(episode_indices) if ep == target_episode]
    print(f"Episode {target_episode}: {len(frame_indices)} steps")

    print(
        f"\n{'Step':>5} | {'GT Action (raw)':>30} | {'Pred Action (raw)':>30} | {'MSE':>10}"
    )
    print("-" * 85)

    all_mse = []

    for step_idx, frame_idx in enumerate(frame_indices):
        sample = full_dataset[frame_idx]

        # Add batch dim to all tensors
        batch = {
            k: v.unsqueeze(0) for k, v in sample.items() if isinstance(v, torch.Tensor)
        }

        formatted = task.format_batch(batch)

        with torch.no_grad():
            policy_out = policy.compute_action(formatted)

        pred_norm = policy_out.actions  # (1, action_horizon, 2)
        gt_norm = formatted[ObservationKey.gt_actions]  # (1, action_horizon, 2)

        pred_raw = (pred_norm[0, 0].cpu().numpy() * 512.0) + 256.0
        gt_raw = (gt_norm[0, 0].cpu().numpy() * 512.0) + 256.0

        mse = torch.nn.functional.mse_loss(pred_norm, gt_norm).item()
        all_mse.append(mse)

        print(
            f"{step_idx:>5} | "
            f"GT: [{gt_raw[0]:>8.2f}, {gt_raw[1]:>8.2f}]     | "
            f"Pred: [{pred_raw[0]:>8.2f}, {pred_raw[1]:>8.2f}]   | "
            f"{mse:>10.6f}"
        )

    print("\n" + "=" * 85)
    print(f"Episode {target_episode}: {len(all_mse)} steps")
    print(
        f"  Avg MSE: {np.mean(all_mse):.6f}  |  Min: {min(all_mse):.6f}  |  Max: {max(all_mse):.6f}"
    )


if __name__ == "__main__":
    ckpt_path = "/Users/suhrudh/savannah/scripts/artifacts/flow_matching_pusht:v0/best_model.ckpt"
    run_dataset_episode_eval(ckpt_path)
