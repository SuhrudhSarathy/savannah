import sys

try:
    import imp
except ImportError:
    from types import ModuleType

    imp = ModuleType("imp")
    sys.modules["imp"] = imp

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from savannah.factory import build_policy
from savannah.models.policy import Policy
from savannah.trainer.overfit import overfit_on_batch
from savannah.utils.device import get_device
from savannah.utils.log import setup_logging
from savannah.utils.observation import ObservationKey


def make_random_batch(
    policy: Policy,
    obs_horizon: int,
    batch_size: int,
    image_size: int,
    device: torch.device,
) -> dict:
    """Builds a single random batch matching the shapes `policy` expects."""
    images = [
        torch.randn(batch_size, obs_horizon, 3, image_size, image_size, device=device)
        for _ in range(policy.num_cameras)
    ]
    state = torch.randn(batch_size, obs_horizon, policy.state_dim, device=device)
    gt_actions = torch.randn(
        batch_size, policy.action_horizon, policy.action_dim, device=device
    )
    # OverfitObjective feeds `actions` in as the model's conditioning input AND
    # checks the output against it (an identity-mapping test) — it must be the
    # same tensor as gt_actions, not independent noise, or there's nothing to learn.
    actions = gt_actions.clone()
    time = torch.rand((batch_size,), device=device)

    return {
        ObservationKey.images: images,
        ObservationKey.state: state,
        ObservationKey.gt_actions: gt_actions,
        ObservationKey.actions: actions,
        ObservationKey.time: time,
    }


@hydra.main(version_base=None, config_path="../../configs", config_name="overfit")
def main(cfg: DictConfig) -> None:
    """
    Sanity check #1: can the model overfit a single batch of *random* data?

    This exercises the architecture and objective in isolation — no dataset,
    no sim — to catch wiring bugs (shapes, gradients, loss) before spending
    time on the data pipeline or a full training run.
    """
    setup_logging(log_dir="logs", level=cfg.log_level)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    device = get_device()
    print("Using device:", device)

    policy = build_policy(cfg).to(device)
    print(f"Policy params: {policy.num_params():.2f}M")

    obs_horizon = cfg.task.config.obs_horizon
    batch = make_random_batch(
        policy,
        obs_horizon=obs_horizon,
        batch_size=cfg.overfit.batch_size,
        image_size=cfg.overfit.image_size,
        device=device,
    )

    print(f"Overfitting on a single random batch for {cfg.overfit.steps} steps...")
    losses = overfit_on_batch(
        policy,
        batch,
        steps=cfg.overfit.steps,
        lr=cfg.overfit.lr,
        log_every=cfg.overfit.log_every,
    )

    print(f"\nLoss: {losses[0]:.6f} -> {losses[-1]:.6f}")
    if losses[-1] < cfg.overfit.loss_threshold:
        print(
            "PASS — model can overfit random data. Architecture/objective wiring looks sound."
        )
    else:
        print(
            "FAIL — loss did not drop below threshold. Inspect the model/objective before training."
        )


if __name__ == "__main__":
    main()
