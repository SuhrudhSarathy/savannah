import numpy as np
import wandb
from typing import Dict, Any, Optional

from savannah.utils.log import logger


class ExperimentLogger:
    def __init__(
        self,
        project_name: str,
        config: dict,
        use_wandb: bool = True,
        run_name: Optional[str] = None,
    ):
        self.use_wandb = use_wandb

        if self.use_wandb:
            wandb.init(project=project_name, name=run_name, config=config)
        else:
            logger.info("W&B disabled — running in local-only mode.")

    def log_scalars(self, metrics: Dict[str, float], step: int):
        if self.use_wandb:
            metrics["global_step"] = step
            wandb.log(metrics)
        else:
            loss = metrics.get("train/loss", metrics.get("train/total_loss", "N/A"))
            logger.info("[step {}] loss={}", step, loss)

    def log_video(
        self, video_array: np.ndarray, metric_name: str, step: int, fps: int = 10
    ):
        if self.use_wandb:
            try:
                wandb.log(
                    {
                        metric_name: wandb.Video(video_array, fps=fps, format="mp4"),
                        "global_step": step,
                    }
                )
            except Exception as e:
                logger.warning("Failed to log video '{}' to W&B: {}", metric_name, e)
        else:
            logger.debug(
                "[step {}] video generated for {} (shape={})",
                step,
                metric_name,
                video_array.shape,
            )

    def finish(self):
        if self.use_wandb:
            wandb.finish()
