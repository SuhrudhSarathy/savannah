import sys

try:
    import imp
except ImportError:
    import importlib.util
    from types import ModuleType

    imp = ModuleType("imp")
    sys.modules["imp"] = imp

import hydra
from omegaconf import DictConfig, OmegaConf

from savannah.models.factory import build_policy, build_task
from savannah.trainer.trainer import PolicyTrainer, TrainConfig
from savannah.utils.device import get_device
from savannah.utils.logger import ExperimentLogger


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    device = get_device()
    print("Using device:", device)

    task = build_task(cfg, device=device)
    policy = build_policy(cfg).to(device)

    print(f"Policy params: {policy.num_params():,}")

    train_config = TrainConfig(**cfg.trainer)
    logger = ExperimentLogger(
        project_name=cfg.wandb.project,
        config=OmegaConf.to_container(cfg, resolve=True),
        use_wandb=cfg.wandb.use_wandb,
    )

    trainer = PolicyTrainer(policy, task, logger, train_config, device)
    trainer.train()


if __name__ == "__main__":
    main()
