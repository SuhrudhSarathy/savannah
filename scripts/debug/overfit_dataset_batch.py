import os
import sys

try:
    import imp
except ImportError:
    from types import ModuleType

    imp = ModuleType("imp")
    sys.modules["imp"] = imp

import hydra
from omegaconf import DictConfig, OmegaConf

from savannah.factory import build_policy, build_task
from savannah.trainer.overfit import overfit_on_batch, validate_action_reconstruction
from savannah.utils.device import get_device
from savannah.utils.log import setup_logging


@hydra.main(version_base=None, config_path="../../configs", config_name="overfit")
def main(cfg: DictConfig) -> None:
    """
    Sanity check #2: can the model overfit a single *real* batch from the
    dataset — one frame's obs/action window (overfit.frame_index=<int>), or
    every frame of one episode cycled through one at a time, batch_size=1
    (overfit.frame_index=null)?

    This exercises the full data pipeline (task.format_batch, normalization,
    image/state shapes) on top of the model — the natural next step once
    `overfit_random.py` passes, and the last check before a full training run.
    """
    log_path = setup_logging(log_dir="logs", level=cfg.log_level)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    device = get_device()
    print("Using device:", device)

    task = build_task(cfg, device=device)
    policy = build_policy(cfg).to(device)
    loader = task.get_episode_loader(cfg.overfit.episode_index, cfg.overfit.frame_index)
    batches = [task.format_batch(raw_batch) for raw_batch in loader]
    batch = batches if len(batches) > 1 else batches[0]

    cfg_t = task.config
    env_info = f"  env={cfg_t.env_name}" if cfg_t.env_name else ""
    print(f"Policy : {policy.__class__.__name__}  ({policy.num_params():.2f}M params)")
    print(f"Task   : {task.__class__.__name__}{env_info}  repo={cfg_t.repo_id}")
    print(
        f"         obs_horizon={cfg_t.obs_horizon}  action_horizon={cfg_t.action_horizon}  image_size={cfg_t.image_size}  cameras={cfg_t.cameras}"
    )

    checkpoint_path = os.path.join(cfg.overfit.checkpoint_dir, "overfit_final.ckpt")
    frame_desc = (
        f"cycling through {len(batch)} frames, batch_size=1 each"
        if isinstance(batch, list)
        else f"frame={cfg.overfit.frame_index}"
    )
    print(
        f"Overfitting on episode={cfg.overfit.episode_index} ({frame_desc}) "
        f"for {cfg.overfit.steps} steps..."
    )
    losses = overfit_on_batch(
        policy,
        batch,
        steps=cfg.overfit.steps,
        lr=cfg.overfit.lr,
        log_every=cfg.overfit.log_every,
        checkpoint_path=checkpoint_path,
        ema_decay=cfg.overfit.ema_decay,
        episode_index=cfg.overfit.episode_index,
        frame_index=cfg.overfit.frame_index,
    )

    print(f"\nLoss: {losses[0]:.6f} -> {losses[-1]:.6f}")
    if losses[-1] < cfg.overfit.loss_threshold:
        print(
            "PASS — model can overfit a real batch. Data pipeline + model wiring looks sound."
        )
    else:
        print(
            "FAIL — loss did not drop below threshold. Inspect the data pipeline/model before training."
        )

    print(
        "\nVal loop: sampling actions via compute_action and comparing to ground truth..."
    )
    if isinstance(batch, list):
        action_mse = sum(
            validate_action_reconstruction(policy, b) for b in batch
        ) / len(batch)
    else:
        action_mse = validate_action_reconstruction(policy, batch)
    print(f"Action reconstruction MSE: {action_mse:.6f}")
    if action_mse < cfg.overfit.action_mse_threshold:
        print(
            "PASS — sampler reconstructs the overfit batch's actions. Objective's compute_action loop looks sound."
        )
    else:
        print(
            "FAIL — compute_action did not reconstruct ground-truth actions. "
            "Inspect the sampler (scheduler config, prediction_type, step loop)."
        )


if __name__ == "__main__":
    main()
