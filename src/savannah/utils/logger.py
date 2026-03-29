import wandb
import numpy as np
from typing import Dict, Any, Optional


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
            print("Running without W&B logging (Debug Mode).")

    def log_scalars(self, metrics: Dict[str, float], step: int):
        """Logs standard numbers (loss, lr, success rate)."""
        if self.use_wandb:
            # Inject the step into the dictionary so W&B aligns everything
            metrics["global_step"] = step
            wandb.log(metrics)
        else:
            # In debug mode, just print the most important metrics
            loss = metrics.get("train/total_loss", "N/A")
            print(f"[Step {step}] Loss: {loss}")

    def log_video(
        self, video_array: np.ndarray, metric_name: str, step: int, fps: int = 10
    ):
        """
        Safely formats and logs a video.
        Expected video_array shape: (T, C, H, W)
        """
        if self.use_wandb:
            try:
                wandb.log(
                    {
                        metric_name: wandb.Video(video_array, fps=fps, format="mp4"),
                        "global_step": step,
                    }
                )
            except Exception as e:
                print(f"Failed to log video to W&B: {e}")
        else:
            print(
                f"[Step {step}] 🎬 Video generated for {metric_name} (Shape: {video_array.shape})"
            )

    def finish(self):
        """Cleanly closes the logger."""
        if self.use_wandb:
            wandb.finish()
