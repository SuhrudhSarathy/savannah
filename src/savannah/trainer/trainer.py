import os
import signal
from dataclasses import dataclass, field
from pathlib import Path

import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from tqdm import tqdm

from savannah.hub import HubCheckpointUploader
from savannah.models.policy import Policy
from savannah.pretrained import read_checkpoint, save_pretrained
from savannah.tasks import BaseRobotTask
from savannah.trainer.ema import EMA
from savannah.trainer.scheduler import get_custom_scheduler
from savannah.utils.eval_and_log import evaluate_and_log
from savannah.utils.log import logger
from savannah.utils.logger import ExperimentLogger
from savannah.utils.observation import ObservationKey


@dataclass
class TrainParams:
    lr: float = 2e-4
    weight_decay: float = 1e-6
    vision_lr: float | None = None
    training_steps: int = 100_000
    warmup_steps: int = 1000
    schedule: str = "cosine"
    gradient_accumulation_steps: int = 4
    ema_decay: float = 0.9999


@dataclass
class EvalParams:
    eval_freq: int = 5000
    eval_using_sim: bool = False
    sim_eval_freq: int = 5000
    eval_episodes: int = 5
    eval_execute_steps: int = 8


@dataclass
class CheckpointParams:
    checkpoint_dir: str = "checkpoints"
    # Push every checkpoint to the Hugging Face Hub repo `hub_repo_id`, one
    # subfolder per checkpoint kind. training_state.pt is never pushed.
    push_to_hub: bool = False
    hub_repo_id: str | None = None
    hub_private: bool = True
    # A local checkpoint dir (e.g. checkpoints/<run>/latest) or run dir.
    resume_from: str | None = None


@dataclass
class TrainConfig:
    train: TrainParams = field(default_factory=TrainParams)
    eval: EvalParams = field(default_factory=EvalParams)
    checkpoint: CheckpointParams = field(default_factory=CheckpointParams)


cs = ConfigStore.instance()
cs.store(name="train_config", node=TrainConfig)


class PolicyTrainer:
    def __init__(
        self,
        policy: Policy,  # e.g., FlowMatchingPolicy
        task: BaseRobotTask,  # e.g., PushTTask
        logger: ExperimentLogger,
        config: TrainConfig,
        device: torch.device,
        run_cfg: DictConfig,
    ):
        self.policy = policy.to(device)
        self.task = task
        self.logger = logger
        self.config = config
        self.device = device
        self.run_cfg = run_cfg

        # Setup Optimizer
        vision_params = []
        other_params = []
        for name, p in self.policy.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("vision_encoder."):
                vision_params.append(p)
            else:
                other_params.append(p)

        vision_lr = (
            self.config.train.vision_lr
            if self.config.train.vision_lr is not None
            else self.config.train.lr
        )
        self.optimizer = AdamW(
            [
                {"params": other_params, "lr": self.config.train.lr},
                {"params": vision_params, "lr": vision_lr},
            ],
            weight_decay=self.config.train.weight_decay,
        )

        # Setup Scheduler
        self.scheduler = get_custom_scheduler(
            self.optimizer,
            num_warmup_steps=self.config.train.warmup_steps
            // self.config.train.gradient_accumulation_steps,
            num_training_steps=self.config.train.training_steps
            // self.config.train.gradient_accumulation_steps,
            schedule=self.config.train.schedule,
        )

        # Setup EMA
        self.ema = EMA(self.policy, decay=self.config.train.ema_decay)

        # Tracking
        self.global_step = 0
        self.best_success_rate = -1.0
        self.best_val_loss = 10000.0

        os.makedirs(self.config.checkpoint.checkpoint_dir, exist_ok=True)

        self.uploader: HubCheckpointUploader | None = None
        if self.config.checkpoint.push_to_hub:
            if not self.config.checkpoint.hub_repo_id:
                raise ValueError(
                    "trainer.checkpoint.push_to_hub=true needs "
                    "trainer.checkpoint.hub_repo_id (e.g. <hf_user>/<run name>)"
                )
            self.uploader = HubCheckpointUploader(
                self.config.checkpoint.hub_repo_id,
                private=self.config.checkpoint.hub_private,
                run_name=OmegaConf.select(run_cfg, "name"),
                model_target=OmegaConf.select(run_cfg, "model._target_"),
                dataset=OmegaConf.select(run_cfg, "task.config.repo_id"),
            )

    def save_checkpoint(self, name: str, metrics: dict[str, float] | None = None):
        """Saves `checkpoint_dir/<name>/` (EMA weights + everything needed to
        resume) and, with push_to_hub, uploads it to `<hub_repo_id>/<name>/`."""
        path = Path(self.config.checkpoint.checkpoint_dir) / name
        if self.uploader is not None:
            # Don't replace the folder while its previous upload still reads it.
            self.uploader.wait_for(name)

        save_pretrained(
            path,
            self.run_cfg,
            self.task,
            self.ema.shadow.state_dict(),
            training_state={
                "model_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss,
                "best_success_rate": self.best_success_rate,
            },
            step=self.global_step,
            metrics=metrics,
        )

        if self.uploader is not None:
            self.uploader.push(path, name, self.global_step, metrics)

    def wait_for_uploads(self):
        if self.uploader is not None:
            self.uploader.wait_all()

    def load_checkpoint(self, path: str):
        """Restores full training state from a local checkpoint saved by
        save_checkpoint(): a checkpoint dir, or a run dir (uses its latest/)."""
        checkpoint_dir = Path(path)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"No local checkpoint at {path} (training state is never pushed "
                "to the Hub, so resuming needs the local checkpoint dir)"
            )
        if not (checkpoint_dir / "manifest.json").is_file():
            checkpoint_dir = checkpoint_dir / "latest"
        logger.info("Resuming training from checkpoint: {}", checkpoint_dir)
        contents = read_checkpoint(checkpoint_dir)
        checkpoint = contents.training_state
        if checkpoint is None:
            raise ValueError(f"{checkpoint_dir} has no training state to resume from")

        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.ema.shadow.load_state_dict(contents.ema_weights)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", self.best_val_loss)
        self.best_success_rate = checkpoint.get(
            "best_success_rate", self.best_success_rate
        )

        logger.success(
            "Resumed at step {} (best_val_loss={:.6f}, best_success_rate={:.3f})",
            self.global_step,
            self.best_val_loss,
            self.best_success_rate,
        )

    def train(self):
        logger.info("Starting training for {} steps…", self.config.train.training_steps)

        train_loader = self.task.get_train_loader()
        train_iter = iter(train_loader)

        val_loader = self.task.get_val_loader()

        def _sigint_handler(sig, frame):
            logger.warning(
                "Interrupted at step {} — saving latest checkpoint…", self.global_step
            )
            self.save_checkpoint("latest")
            self.wait_for_uploads()
            self.logger.finish()
            raise SystemExit(0)

        signal.signal(signal.SIGINT, _sigint_handler)

        # Use a flat tqdm loop based on steps, rather than epochs
        pbar = tqdm(total=self.config.train.training_steps, desc="Training")
        accum_counter = 0

        while self.global_step < self.config.train.training_steps:
            self.policy.train()

            # 1. Fetch next batch (cycle loader if empty)
            try:
                raw_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                raw_batch = next(train_iter)

            # 2. Format batch using the Task (augmentation decays with global_step)
            batch = self.task.format_batch(raw_batch, step=self.global_step)

            with torch.autocast(self.device.type, dtype=torch.bfloat16):
                # 3. Forward Pass & Loss (Policy handles all ODE matching internally!)
                loss = self.policy.compute_loss(batch)
                scaled_loss = loss / self.config.train.gradient_accumulation_steps

            scaled_loss.backward()
            accum_counter += 1

            # Optimiser step and scheduler step
            if accum_counter % self.config.train.gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), 1.0
                )
                self.optimizer.step()
                self.scheduler.step()
                self.ema.update()
                self.optimizer.zero_grad()

                self.logger.log_scalars(
                    {
                        "train/loss": loss.item(),
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                        "train/vision_lr": self.optimizer.param_groups[1]["lr"],
                        "train/grad_norm": grad_norm.item(),
                    },
                    step=self.global_step,
                )
                accum_counter = 0

            # 7. Evaluation & Checkpointing
            if (
                self.global_step > 0
                and self.global_step % self.config.eval.eval_freq == 0
            ):
                logger.info("── Validating at step {} ──", self.global_step)
                val_losses = []
                reconstruction_losses = []
                with torch.no_grad():
                    for batch in tqdm(val_loader):
                        batch = self.task.format_batch(batch)
                        loss = self.policy.compute_loss(batch)
                        out = self.policy.compute_action(batch).actions

                        reconstruction_loss = torch.norm(
                            batch[ObservationKey.gt_actions] - out, dim=-1
                        ).mean()

                        reconstruction_losses.append(reconstruction_loss.item())
                        val_losses.append(loss.item())

                avg_val_loss = torch.as_tensor(val_losses).mean().item()
                avg_reconstruction_loss = (
                    torch.as_tensor(reconstruction_losses).mean().item()
                )

                self.logger.log_scalars(
                    {
                        "val/loss": avg_val_loss,
                        "val/reconstruction_loss": avg_reconstruction_loss,
                    },
                    step=self.global_step,
                )

                logger.info("Reconstruction Loss: {:.6f}", avg_reconstruction_loss)
                val_metrics = {
                    "val_loss": avg_val_loss,
                    "val_reconstruction_loss": avg_reconstruction_loss,
                }
                self.save_checkpoint("latest", val_metrics)

                if avg_reconstruction_loss < self.best_val_loss:
                    self.best_val_loss = avg_reconstruction_loss
                    self.save_checkpoint("best_model", val_metrics)
                    logger.success("New best val loss: {:.6f}", self.best_val_loss)

            # 8. Live sim eval & video upload (own cadence — typically much rarer
            # than the val-loss check above, since it rolls out full episodes)
            if (
                self.config.eval.eval_using_sim
                and self.global_step > 0
                and self.global_step % self.config.eval.sim_eval_freq == 0
            ):
                logger.info("── Live sim eval at step {} ──", self.global_step)
                self.ema.shadow.eval()
                success_rate, mean_reward = evaluate_and_log(
                    policy=self.ema.shadow,
                    task=self.task,
                    logger=self.logger,
                    step=self.global_step,
                    num_episodes=self.config.eval.eval_episodes,
                    execute_steps=self.config.eval.eval_execute_steps,
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
                    self.save_checkpoint(
                        "best_model_success",
                        {"success_rate": success_rate, "mean_reward": mean_reward},
                    )
                    logger.success(
                        "New best success rate: {:.3f}", self.best_success_rate
                    )

            self.global_step += 1
            pbar.update(1)

        pbar.close()

        self.wait_for_uploads()
        self.logger.finish()
        print("Training Complete!")
