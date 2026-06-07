import sys

try:
    import imp
except ImportError:
    import importlib.util
    from types import ModuleType

    imp = ModuleType("imp")
    sys.modules["imp"] = imp

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from savannah.models.factory import build_policy, build_task
from savannah.utils.checkpoint import resolve_checkpoint_path
from savannah.utils.device import get_device
from savannah.utils.observation import ObservationKey


def run_dataset_episode_eval(cfg: DictConfig) -> None:
    device = get_device()
    print("Using device:", device)

    task = build_task(cfg, device=device)
    policy = build_policy(cfg).to(device)

    checkpoint_path = resolve_checkpoint_path(cfg)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_key = "ema_state_dict" if cfg.use_ema else "model_state_dict"
    policy.load_state_dict(checkpoint[state_key])
    policy.eval()

    dataset_config = task.config
    obs_horizon = dataset_config.obs_horizon
    action_horizon = dataset_config.action_horizon

    # Build delta_timestamps the same way LerobotDatasetWrapper does
    obs_timestamps = [
        -(obs_horizon - 1 - i) / dataset_config.fps for i in range(obs_horizon)
    ]
    act_timestamps = [i / dataset_config.fps for i in range(action_horizon)]

    delta_timestamps = {
        "observation.state": obs_timestamps,
        "observation.image": obs_timestamps,
        "action": act_timestamps,
    }

    full_dataset = LeRobotDataset(
        dataset_config.repo_id, delta_timestamps=delta_timestamps
    )

    target_episode = cfg.episode_index
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
            print(policy_out)

        pred_norm = policy_out.actions  # (1, action_horizon, action_dim)
        gt_norm = formatted[
            ObservationKey.gt_actions
        ]  # (1, action_horizon, action_dim)

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


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    run_dataset_episode_eval(cfg)


if __name__ == "__main__":
    main()
