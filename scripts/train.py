import sys

try:
    import imp
except ImportError:
    from types import ModuleType

    imp = ModuleType("imp")
    sys.modules["imp"] = imp

import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from savannah.factory import build_policy, build_task
from savannah.trainer.trainer import PolicyTrainer, TrainConfig
from savannah.utils.device import get_device
from savannah.utils.log import logger, setup_logging
from savannah.utils.logger import ExperimentLogger


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging(log_dir="logs", level=cfg.log_level)
    logger.info("Config:\n{}", OmegaConf.to_yaml(cfg, resolve=True))

    fraction = float(os.environ.get("CUDA_MEMORY_FRACTION", "1.0"))
    if fraction < 1.0 and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)
        logger.info("VRAM cap: {:.0%} of device memory", fraction)

    device = get_device()

    task = build_task(cfg, device=device)
    policy = build_policy(cfg).to(device)

    logger.info("Policy params: {:,}", policy.num_params())

    train_config = TrainConfig(**cfg.trainer)
    exp_logger = ExperimentLogger(
        project_name=cfg.wandb.project,
        config=OmegaConf.to_container(cfg, resolve=True),
        enable_logging=cfg.wandb.enable_logging,
        enable_model_checkpoint=cfg.wandb.enable_model_checkpoint,
        run_name=cfg.wandb.run_name,
    )

    trainer = PolicyTrainer(policy, task, exp_logger, train_config, device)
    trainer.train()

    exp_logger.finish()


if __name__ == "__main__":
    main()
