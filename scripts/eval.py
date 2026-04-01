import os
import sys
from collections import deque
from dataclasses import asdict

import gym_pusht
import gymnasium as gym
import torch

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


def download_artifact(entity, project, run_id, alias="best"):
    """Downloads the model checkpoint from W&B."""

    api = wandb.Api()
    artifact_path = f"{entity}/{project}/flow_matching_pusht:{alias}"
    print(f"Downloading artifact: {artifact_path}")
    artifact = api.artifact(artifact_path)
    artifact_dir = artifact.download()
    return os.path.join(artifact_dir, "best_model.ckpt")


def run_live_eval(checkpoint_path, repo_id="lerobot/pusht", video_folder="./videos"):
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
        batch_size=1,
    )
    task = PushTTask(config=dataset_config, device=device)

    # 2. Reconstruct Model
    # (Values here must match your training hyperparameters)
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

    # 3. Load EMA Weights
    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint["ema_state_dict"])
    policy.eval()

    # 4. Use Task to get the specific Eval Env
    base_env = task.get_env(render_mode="rgb_array")
    env = RecordedEnv(
        base_env,
        video_folder=video_folder,
        name_prefix="flow-matching-eval",
    )

    dataloader = task.get_train_loader()
    batch = next(iter(dataloader))
    batch = task.format_batch(batch)

    with torch.no_grad():
        policy_out = policy.compute_action(batch)

    pred_actions = policy_out.actions  # (B, action_horizon, 2)
    gt_actions = batch[ObservationKey.gt_actions]  # (B, action_horizon, 2)

    print("Pred:", pred_actions[0])
    print("GT:  ", gt_actions[0])
    print("MSE: ", torch.nn.functional.mse_loss(pred_actions, gt_actions).item())

    # # From the dataset
    # batch = next(iter(dataloader))
    # print(
    #     "Dataset image range:",
    #     batch["observation.image"].min(),
    #     batch["observation.image"].max(),
    # )
    # print(
    #     "Dataset state range:",
    #     batch["observation.state"].min(),
    #     batch["observation.state"].max(),
    # )
    # print("Dataset action range:", batch["action"].min(), batch["action"].max())

    # After format_batch
    # formatted = task.format_batch(batch)
    # print(
    #     "Formatted image range:",
    #     formatted[ObservationKey.images][0].min(),
    #     formatted[ObservationKey.images][0].max(),
    # )
    # print(
    #     "Formatted state range:",
    #     formatted[ObservationKey.state].min(),
    #     formatted[ObservationKey.state].max(),
    # )
    # print(
    #     "Formatted actions range:",
    #     formatted[ObservationKey.gt_actions].min(),
    #     formatted[ObservationKey.gt_actions].max(),
    # )

    # From the gym env raw obs
    # raw_obs, _ = env.reset()
    # print("Gym image range:", raw_obs["pixels"].min(), raw_obs["pixels"].max())
    # print("Gym state range:", raw_obs["agent_pos"].min(), raw_obs["agent_pos"].max())

    # # After preprocess_observation_history
    # obs_history = deque([raw_obs] * obs_horizon, maxlen=obs_horizon)
    # obs_dict = task.preprocess_observation_history(obs_history)
    # print(
    #     "Preprocessed image range:",
    #     obs_dict[ObservationKey.images][0].min(),
    #     obs_dict[ObservationKey.images][0].max(),
    # )
    # print(
    #     "Preprocessed state range:",
    #     obs_dict[ObservationKey.state].min(),
    #     obs_dict[ObservationKey.state].max(),
    # )

    # for ep in range(1):
    #     raw_obs, _ = env.reset()
    #     done = False

    #     # Use our generic ActionBuffer for smooth Receding Horizon Control
    #     action_buffer = ActionBuffer(execute_steps=4)

    #     obs_history = deque(
    #                 [raw_obs] * obs_horizon,
    #                 maxlen=obs_horizon,
    #             )

    #     print(f"--- Episode {ep+1} ---")
    #     while not done:
    #         # Render for the human eye (handled by gym)
    #         env.render()

    #         if action_buffer.is_empty():
    #             # USE THE TASK DIRECTLY: This handles the 255.0 div,
    #             # the (pos - 256)/512, and the (1, 1, C, H, W) unsqueezing.
    #             obs_dict = task.preprocess_observation_history(obs_history)
    #             print("Len of Observation Key: ", obs_dict[ObservationKey.images][0].shape, obs_dict[ObservationKey.state])

    #             with torch.no_grad():
    #                 policy_out = policy.compute_action(obs_dict)

    #             action_buffer.push(policy_out.actions[0][1: 5])

    #         # Pop and execute
    #         raw_action = action_buffer.pop()

    #         # USE THE TASK DIRECTLY: Un-normalizes [-1, 1] -> [0, 512]
    #         action_to_apply = task.postprocess_action(raw_action)
    #         print(action_to_apply)

    #         raw_obs, reward, terminated, truncated, info = env.step(action_to_apply)

    #         obs_history.append(raw_obs)
    #         done = terminated or truncated
    #     print(f"Result: Terminated: {terminated}; Truncated: {truncated}")

    env.close()


if __name__ == "__main__":
    # Update these with your W&B details
    WANDB_ENTITY = "suhrudhsarathy"
    WANDB_PROJECT = "flow-matching-robotics"

    ckpt_path = download_artifact(
        WANDB_ENTITY, WANDB_PROJECT, "flow_matching_pusht", alias="v0"
    )
    # ckpt_path = "/Users/suhrudh/savannah/scripts/artifacts/flow_matching_pusht:v0/best_model.ckpt"
    run_live_eval(ckpt_path)
