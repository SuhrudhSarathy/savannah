import copy
import math
import os
import signal
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import torch
from hydra.core.config_store import ConfigStore
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from savannah.models.policy import Policy
from savannah.tasks import BaseRobotTask
from savannah.trainer.ema import EMA
from savannah.trainer.scheduler import get_custom_scheduler
from savannah.utils.device import get_device
from savannah.utils.eval_and_log import evaluate_and_log
from savannah.utils.gcs import upload_file_to_gcs
from savannah.utils.log import logger
from savannah.utils.logger import ExperimentLogger
from savannah.utils.observation import ObservationKey


@dataclass
class TrainConfig:
    lr: float = 2e-4
    training_steps: int = 100_000
    warmup_steps: int = 1000
    eval_freq: int = 5000
    eval_using_sim: bool = False
    sim_eval_freq: int = 5000
    eval_episodes: int = 5
    eval_execute_steps: int = 8
    ema_decay: float = 0.9999
    checkpoint_dir: str = "checkpoints"
    artifact_name: str = "model"


@dataclass
class GCPConfig:
    bucket: Optional[str] = None
    enable_model_checkpoint: bool = False


cs = ConfigStore.instance()
cs.store(name="train_config", node=TrainConfig)
cs.store(name="gcp_config", node=GCPConfig)


class PolicyTrainer:
    def __init__(
        self,
        policy: Policy,  # e.g., FlowMatchingPolicy
        task: BaseRobotTask,  # e.g., PushTTask
        logger: ExperimentLogger,
        config: TrainConfig,
        device: torch.device,
        gcp_config: GCPConfig = GCPConfig(),
    ):
        self.policy = policy.to(device)
        self.task = task
        self.logger = logger
        self.config = config
        self.gcp_config = gcp_config
        self.device = device

        # Setup Optimizer
        self.optimizer = AdamW(
            self.policy.parameters(), lr=self.config.lr, weight_decay=1e-4
        )

        # Setup Scheduler
        self.scheduler = get_custom_scheduler(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.training_steps,
        )

        # Setup EMA
        self.ema = EMA(self.policy, decay=self.config.ema_decay)

        # Tracking
        self.global_step = 0
        self.best_success_rate = -1.0
        self.best_val_loss = 10000.0

        # Captured once so all checkpoints uploaded for this run share a folder.
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, name: str, success_rate: float):
        """Saves both the training weights and the EMA shadow weights."""
        checkpoint = {
            "model_state_dict": self.policy.state_dict(),
            "ema_state_dict": self.ema.shadow.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "success_rate": success_rate,
        }
        path = os.path.join(self.config.checkpoint_dir, f"{name}.ckpt")
        torch.save(checkpoint, path)

    def train(self):
        logger.info("Starting training for {} steps…", self.config.training_steps)

        train_loader = self.task.get_train_loader()
        train_iter = iter(train_loader)

        val_loader = self.task.get_val_loader()

        def _sigint_handler(sig, frame):
            logger.warning("Interrupted at step {} — saving latest.ckpt…", self.global_step)
            self.save_checkpoint("latest", self.best_val_loss)
            raise SystemExit(0)

        signal.signal(signal.SIGINT, _sigint_handler)

        # Use a flat tqdm loop based on steps, rather than epochs
        pbar = tqdm(total=self.config.training_steps, desc="Training")

        while self.global_step < self.config.training_steps:
            self.policy.train()

            # 1. Fetch next batch (cycle loader if empty)
            try:
                raw_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                raw_batch = next(train_iter)

            # 2. Format batch using the Task
            batch = self.task.format_batch(raw_batch)

            # 3. Forward Pass & Loss (Policy handles all ODE matching internally!)
            loss = self.policy.compute_loss(batch)

            # 4. Backward Pass
            self.optimizer.zero_grad()
            loss.backward()

            # Optional: Gradient clipping is highly recommended for Flow Matching
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)

            self.optimizer.step()
            self.scheduler.step()

            # 5. Update EMA shadow weights
            self.ema.update()

            # 6. Logging
            if self.global_step % 10 == 0:
                self.logger.log_scalars(
                    {
                        "train/loss": loss.item(),
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                    },
                    step=self.global_step,
                )

            # 7. Evaluation & Checkpointing
            if self.global_step > 0 and self.global_step % self.config.eval_freq == 0:
                logger.info("── Validating at step {} ──", self.global_step)
                val_losses = []
                with torch.no_grad():
                    for batch in tqdm(val_loader):
                        batch = self.task.format_batch(batch)
                        loss = self.policy.compute_loss(batch)

                        val_losses.append(loss.item())

                avg_val_loss = torch.as_tensor(val_losses).mean().item()

                self.logger.log_scalars(
                    {"val/loss": avg_val_loss}, step=self.global_step
                )

                self.save_checkpoint("latest", avg_val_loss)
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    self.save_checkpoint("best_model", avg_val_loss)
                    logger.success("New best val loss: {:.6f}", self.best_val_loss)

            # 8. Live sim eval & video upload (own cadence — typically much rarer
            # than the val-loss check above, since it rolls out full episodes)
            if (
                self.config.eval_using_sim
                and self.global_step > 0
                and self.global_step % self.config.sim_eval_freq == 0
            ):
                logger.info("── Live sim eval at step {} ──", self.global_step)
                self.ema.shadow.eval()
                success_rate, mean_reward = evaluate_and_log(
                    policy=self.ema.shadow,
                    task=self.task,
                    logger=self.logger,
                    step=self.global_step,
                    num_episodes=self.config.eval_episodes,
                    execute_steps=self.config.eval_execute_steps,
                )

                self.logger.log_scalars(
                    {
                        "val/success_rate": success_rate,
                        "val/mean_reward": mean_reward,
                    },
                    step=self.global_step,
                )

                if success_rate > self.best_success_rate:
                    self.best_success_rate = success_rate
                    self.save_checkpoint("best_model_success", success_rate)
                    logger.success(
                        "New best success rate: {:.3f}", self.best_success_rate
                    )

            self.global_step += 1
            pbar.update(1)

        pbar.close()

        checkpoint_aliases = {
            "best_model.ckpt": "best",
            "best_model_success.ckpt": "best_success",
            "latest.ckpt": "latest",
        }

        if self.logger.enable_model_checkpoint:
            for ckpt_name, alias in checkpoint_aliases.items():
                local_path = os.path.join(self.config.checkpoint_dir, ckpt_name)
                if os.path.exists(local_path):
                    self.logger.log_model_artifact(
                        model_path=local_path,
                        artifact_name=self.config.artifact_name,
                        aliases=[alias],
                    )
        else:
            logger.debug(
                "wandb.enable_model_checkpoint is false — skipping W&B artifact upload."
            )

        if self.gcp_config.enable_model_checkpoint and self.gcp_config.bucket:
            gcs_prefix = f"{self.config.artifact_name}_{self.run_timestamp}"
            for ckpt_name in checkpoint_aliases:
                local_path = os.path.join(self.config.checkpoint_dir, ckpt_name)
                if os.path.exists(local_path):
                    try:
                        uri = upload_file_to_gcs(
                            local_path,
                            self.gcp_config.bucket,
                            f"{gcs_prefix}/{ckpt_name}",
                        )
                        logger.success("Uploaded {} to {}", local_path, uri)
                    except Exception as e:
                        logger.error("Failed to upload {} to GCS: {}", local_path, e)
        elif self.gcp_config.enable_model_checkpoint:
            logger.debug(
                "gcp.enable_model_checkpoint is true but gcp.bucket is not set — skipping GCS upload."
            )
        else:
            logger.debug(
                "gcp.enable_model_checkpoint is false — skipping checkpoint upload to GCS."
            )

        self.logger.finish()
        print("Training Complete!")
