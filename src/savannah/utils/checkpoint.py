import glob
import os

from omegaconf import DictConfig

import wandb
from savannah.utils.log import logger


def resolve_checkpoint_path(cfg: DictConfig) -> str:
    """Resolves the checkpoint to evaluate, downloading it from W&B if requested."""
    if cfg.wandb_artifact:
        logger.info("Fetching checkpoint from W&B artifact: {}", cfg.wandb_artifact)
        artifact = wandb.Api().artifact(cfg.wandb_artifact, type="model")
        artifact_dir = artifact.download()
        ckpts = glob.glob(os.path.join(artifact_dir, "*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(
                f"No .ckpt file found in artifact dir: {artifact_dir}"
            )
        return ckpts[0]

    if cfg.checkpoint_path:
        return cfg.checkpoint_path

    raise ValueError("Either checkpoint_path or wandb_artifact must be set")
