import numpy as np
import wandb
from typing import Dict, Any, Optional

from savannah.utils.log import logger


class ExperimentLogger:
    def __init__(
        self,
        project_name: str,
        config: dict,
        enable_logging: bool = True,
        enable_model_checkpoint: bool = False,
        run_name: Optional[str] = None,
    ):
        self.enable_logging = enable_logging
        self.enable_model_checkpoint = enable_model_checkpoint

        self.run_id: Optional[str] = None
        self.run_url: Optional[str] = None

        if self.enable_logging:
            wandb.init(project=project_name, name=run_name, config=config)
            self.run_id = wandb.run.id
            self.run_url = wandb.run.url
        else:
            logger.info("W&B disabled — running in local-only mode.")

    def log_scalars(self, metrics: Dict[str, float], step: int):
        if self.enable_logging:
            metrics["global_step"] = step
            wandb.log(metrics)
        else:
            loss = metrics.get("train/loss", metrics.get("train/total_loss", "N/A"))
            logger.info("[step {}] loss={}", step, loss)

    def log_video(
        self, video_array: np.ndarray, metric_name: str, step: int, fps: int = 10
    ):
        if self.enable_logging:
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

    def log_model_artifact(
        self, model_path: str, artifact_name: str, aliases: list[str] | None = None
    ):
        if not self.enable_logging or not self.enable_model_checkpoint:
            logger.debug(
                "Skipping W&B model artifact upload for {} (enable_logging={}, enable_model_checkpoint={})",
                model_path,
                self.enable_logging,
                self.enable_model_checkpoint,
            )
            return

        logger.info("Uploading {} to W&B artifacts as '{}'…", model_path, artifact_name)
        try:
            artifact = wandb.Artifact(name=artifact_name, type="model")
            artifact.add_file(model_path)
            wandb.log_artifact(artifact, aliases=aliases)
            logger.success("Artifact '{}' uploaded successfully.", artifact_name)
        except Exception as e:
            logger.error("Failed to upload artifact '{}': {}", artifact_name, e)

    def log_hf_push(self, repo_id: str, commit_hash: str):
        if self.enable_logging:
            wandb.run.summary["hf_repo"] = repo_id
            wandb.run.summary["hf_commit"] = commit_hash

    def finish(self):
        if self.enable_logging:
            wandb.finish()
