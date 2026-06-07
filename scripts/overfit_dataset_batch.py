import sys

try:
    import imp
except ImportError:
    import importlib.util
    from types import ModuleType

    imp = ModuleType("imp")
    sys.modules["imp"] = imp

import hydra
from omegaconf import DictConfig

from savannah.models.factory import build_policy, build_task
from savannah.trainer.overfit import overfit_on_batch
from savannah.utils.device import get_device


@hydra.main(version_base=None, config_path="../configs", config_name="overfit")
def main(cfg: DictConfig) -> None:
    """
    Sanity check #2: can the model overfit a single *real* batch from the
    dataset?

    This exercises the full data pipeline (task.format_batch, normalization,
    image/state shapes) on top of the model — the natural next step once
    `overfit_random.py` passes, and the last check before a full training run.
    """
    device = get_device()
    print("Using device:", device)

    task = build_task(cfg, device=device)
    policy = build_policy(cfg).to(device)
    print(f"Policy params: {policy.num_params():.2f}M")

    train_loader = task.get_train_loader()
    raw_batch = next(iter(train_loader))
    batch = task.format_batch(raw_batch)

    print(f"Overfitting on a single real batch for {cfg.overfit.steps} steps...")
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
            "PASS — model can overfit a real batch. Data pipeline + model wiring looks sound."
        )
    else:
        print(
            "FAIL — loss did not drop below threshold. Inspect the data pipeline/model before training."
        )


if __name__ == "__main__":
    main()
