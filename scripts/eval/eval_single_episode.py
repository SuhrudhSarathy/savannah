import sys

try:
    import imp
except ImportError:
    from types import ModuleType

    imp = ModuleType("imp")
    sys.modules["imp"] = imp

import hydra
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from omegaconf import DictConfig

from savannah.factory import build_policy, build_task
from savannah.utils.checkpoint import resolve_checkpoint_path
from savannah.utils.device import get_device
from savannah.utils.log import setup_logging
from savannah.utils.observation import ObservationKey


def _lerobot_episode_samples(cfg: DictConfig, task):
    """Fetches (fetch_fn, num_steps) for tasks backed by a real LeRobot repo."""
    dataset_config = task.config
    obs_horizon = dataset_config.obs_horizon
    action_horizon = dataset_config.action_horizon

    # Build delta_timestamps the same way LerobotDatasetWrapper does
    obs_timestamps = [
        -(obs_horizon - 1 - i) / dataset_config.fps for i in range(obs_horizon)
    ]
    act_timestamps = [i / dataset_config.fps for i in range(action_horizon)]

    camera_keys = [f"observation.images.{cam}" for cam in dataset_config.cameras]
    delta_timestamps = {
        "observation.state": obs_timestamps,
        **{cam_key: obs_timestamps for cam_key in camera_keys},
        "action": act_timestamps,
    }

    full_dataset = LeRobotDataset(
        dataset_config.repo_id,
        delta_timestamps=delta_timestamps,
        video_backend="torchcodec",
    )

    target_episode = cfg.episode_index
    episode_indices = full_dataset.hf_dataset["episode_index"]
    frame_indices = [i for i, ep in enumerate(episode_indices) if ep == target_episode]
    print(f"Episode {target_episode}: {len(frame_indices)} steps")

    return (lambda i: full_dataset[frame_indices[i]]), len(frame_indices)


def _abc_episode_samples(cfg: DictConfig, task):
    """Fetches (fetch_fn, num_steps) for ABCPutBottlesTask, which reads flat
    per-episode directories directly instead of a LeRobotDataset (see
    savannah.tasks.abc's module docstring)."""
    from savannah.tasks.abc import ABCEpisodeDataset

    dataset_config = task.config
    if not dataset_config.root:
        raise ValueError("ABCPutBottlesTask requires config.root to locate episodes")

    dataset = ABCEpisodeDataset(
        dataset_config.root,
        dataset_config.cameras,
        dataset_config.obs_horizon,
        dataset_config.action_horizon,
    )

    target_episode = cfg.episode_index
    if not 0 <= target_episode < len(dataset.episodes):
        raise ValueError(
            f"episode_index {target_episode} out of range: found "
            f"{len(dataset.episodes)} episodes under {dataset_config.root}"
        )

    ep_dir, _, usable, _, _ = dataset.episodes[target_episode]
    offset = int(dataset.cum[target_episode - 1]) if target_episode > 0 else 0
    print(f"Episode {target_episode} ({ep_dir.name}): {usable} steps")

    return (lambda i: dataset[offset + i]), usable


def _run_episode_eval(task, policy, fetch_fn, num_steps: int) -> None:
    print(
        f"\n{'Step':>5} | {'GT Action (raw)':>30} | {'Pred Action (raw)':>30} | {'MSE':>10}"
    )
    print("-" * 85)

    all_mse = []

    for step_idx in range(num_steps):
        sample = fetch_fn(step_idx)

        # Add batch dim to all tensors
        batch = {
            k: v.unsqueeze(0) for k, v in sample.items() if isinstance(v, torch.Tensor)
        }
        if task.config.use_language:
            # LeRobotDataset always populates "task" regardless of
            # delta_timestamps; mirror _LanguageKeyDataset's rename here since
            # this script builds the dataset directly instead of going
            # through LerobotDatasetWrapper.
            batch[ObservationKey.language] = [sample["task"]]

        formatted = task.format_batch(batch)

        with torch.no_grad():
            policy_out = policy.compute_action(formatted)

        pred_norm = policy_out.actions  # (1, action_horizon, action_dim)
        gt_norm = formatted[
            ObservationKey.gt_actions
        ]  # (1, action_horizon, action_dim)

        pred_raw = task.postprocess_action(pred_norm[0, 0])
        gt_raw = task.postprocess_action(gt_norm[0, 0])

        mse = torch.nn.functional.mse_loss(pred_norm, gt_norm).item()
        all_mse.append(mse)

        print(
            f"{step_idx:>5} | "
            f"GT: {np.array2string(gt_raw, precision=3, separator=', ')} | "
            f"Pred: {np.array2string(pred_raw, precision=3, separator=', ')} | "
            f"{mse:>10.6f}"
        )

    print("\n" + "=" * 85)
    print(f"{len(all_mse)} steps")
    print(
        f"  Avg MSE: {np.mean(all_mse):.6f}  |  Min: {min(all_mse):.6f}  |  Max: {max(all_mse):.6f}"
    )


def run_dataset_episode_eval(cfg: DictConfig) -> None:
    setup_logging(log_dir="logs", level=cfg.log_level)
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

    # ABCPutBottlesTask has no LeRobotDataset backing it (its episodes live
    # in flat per-episode directories, see savannah/tasks/abc.py's module
    # docstring), so it needs its own sample source. Checked by module name
    # rather than an unconditional top-level import, since abc_bottle_gym
    # isn't installed in every environment that runs this script.
    is_abc_task = type(task).__module__ == "savannah.tasks.abc"
    fetch_fn, num_steps = (
        _abc_episode_samples(cfg, task)
        if is_abc_task
        else _lerobot_episode_samples(cfg, task)
    )

    _run_episode_eval(task, policy, fetch_fn, num_steps)


@hydra.main(version_base=None, config_path="../../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    run_dataset_episode_eval(cfg)


if __name__ == "__main__":
    main()
