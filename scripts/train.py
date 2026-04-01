import os
import sys
from collections import deque
from dataclasses import asdict

import gym_pusht
import gymnasium as gym
import torch
import torch.nn.functional as F

import wandb
from savannah.utils.device import get_device
from savannah.utils.observation import ObservationKey

# --- 1. Python 3.12 Compatibility Patch ---
try:
    import imp
except ImportError:
    import importlib.util
    from types import ModuleType

    imp = ModuleType("imp")
    sys.modules["imp"] = imp

# --- 2. Savannah Imports ---
from savannah.data.dataset import DataSetConfig
from savannah.models.backbones.resnet_backbone import ResNetBackbone
from savannah.models.flow_matching import FlowMatchingPolicy
from savannah.models.vision_encoder import VisionEncoder
from savannah.tasks import RecordedEnv
from savannah.tasks.pusht import PushTTask
from savannah.utils.eval_and_log import ActionBuffer


def overfit_mlp_baseline(task, device):
    """
    Replace the transformer with a tiny MLP.
    If THIS can't overfit, the bug is in compute_loss itself.
    If it CAN, the bug is in the transformer.
    """
    import torch.nn as nn
    import torch.nn.functional as F

    mlp = nn.Sequential(
        nn.Linear(2, 256),
        nn.GELU(),
        nn.Linear(256, 256),
        nn.GELU(),
        nn.Linear(256, 2),
    ).to(device)

    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)

    dataloader = task.get_train_loader()
    batch = next(iter(dataloader))
    batch = task.format_batch(batch)

    x_1 = batch[ObservationKey.gt_actions]  # (B, action_horizon, 2)
    B = x_1.shape[0]

    x_0 = torch.randn_like(x_1)
    t = torch.rand((B,), device=device)
    x_t = (1 - t.view(B, 1, 1)) * x_0 + t.view(B, 1, 1) * x_1
    target = x_1 - x_0

    for step in range(1000):
        optimizer.zero_grad()

        # flatten action horizon into batch
        pred = mlp(x_t.view(-1, 2)).view(B, -1, 2)
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step}: loss = {loss.item():.6f}")


def overfit_one_batch(checkpoint_path=None):
    device = get_device()

    embed_dim = 128
    num_attn_heads = 4
    num_blocks = 6
    feedforward_dim = 4 * embed_dim
    num_cameras = 3

    state_dim = 2
    action_dim = 2
    action_horizon = 8
    obs_horizon = 3

    # 1. Setup Task (The Source of Truth for Shapes/Norms)
    dataset_config = DataSetConfig(
        repo_id="lerobot/pusht",
        fps=10,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        batch_size=8,
    )

    # Setup
    task = PushTTask(config=dataset_config, device=device)
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
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    dataloader = task.get_train_loader()
    batch = next(iter(dataloader))
    batch = task.format_batch(batch)
    B = batch[ObservationKey.state].shape[0]
    device = batch[ObservationKey.state].device

    x_1 = batch[ObservationKey.gt_actions]
    x_0 = torch.randn_like(x_1)
    t = torch.rand((B,), device=device)
    x_t = (1 - t.view(B, 1, 1)) * x_0 + t.view(B, 1, 1) * x_1
    batch[ObservationKey.time] = t
    target = x_1 - x_0

    print("Overfitting single batch...")
    for step in range(5000):
        optimizer.zero_grad()

        # Resample t and x_0 every step so model learns velocity at ALL timesteps
        x_0 = torch.randn_like(x_1)
        t = torch.rand((B,), device=device)
        x_t = (1 - t.view(B, 1, 1)) * x_0 + t.view(B, 1, 1) * x_1
        batch[ObservationKey.time] = t
        target = x_1 - x_0

        out = policy.forward(batch, noisy_actions=x_t).actions
        loss = F.mse_loss(out, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step}: loss = {loss.item():.6f}")

    # ── Test 1: can the model reproduce the velocity it was trained on? ──
    # This tests if the model memorised the fixed (x_t, target) pair
    policy.eval()
    with torch.no_grad():
        pred_vel = policy.forward(batch, noisy_actions=x_t).actions
        vel_mse = F.mse_loss(pred_vel, target)
        print(f"\nVelocity MSE (should be ~0.0): {vel_mse.item():.6f}")
        print("Pred vel:", pred_vel[0])
        print("Target:  ", target[0])

    # ── Test 2: manually integrate from the SAME x_0 used in training ──
    # This tests if the ODE recovers x_1 when starting from known x_0
    with torch.no_grad():
        x = x_0.clone()  # start from the exact x_0 used above
        dt = 1.0 / policy.flowmatching_inference_steps

        for i in range(policy.flowmatching_inference_steps):
            t_val = i * dt
            t_tensor = torch.full((B,), t_val, device=device)
            batch[ObservationKey.time] = t_tensor
            v = policy.forward(batch, noisy_actions=x).actions
            x = x + v * dt

        mse = F.mse_loss(x, x_1)
        print(f"\nODE MSE from fixed x_0 (should be low): {mse.item():.6f}")
        print("Pred:", x[0])
        print("GT:  ", x_1[0])

    # ── Test 3: compute_action with fresh noise ──
    # This is the hardest test — model must generalise to unseen noise
    with torch.no_grad():
        policy_out = policy.compute_action(batch)
        pred = policy_out.actions
        gt = batch[ObservationKey.gt_actions]
        mse = F.mse_loss(pred, gt)
        print(f"\ncompute_action MSE (fresh noise): {mse.item():.6f}")
        print("Pred:", pred[0])
        print("GT:  ", gt[0])


if __name__ == "__main__":
    overfit_one_batch()
