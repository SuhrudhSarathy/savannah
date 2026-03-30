import copy
import math
import os
from dataclasses import dataclass
from traceback import format_exc

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from savannah.models.policy import Policy
from savannah.tasks import BaseRobotTask
from savannah.trainer.ema import EMA
from savannah.trainer.scheduler import get_custom_scheduler
from savannah.utils.device import get_device
from savannah.utils.eval_and_log import evaluate_and_log
from savannah.utils.logger import ExperimentLogger


@dataclass
class TrainConfig:
    lr: float = 1e-4
    training_steps: int = 100_000
    warmup_steps: int = 1000
    eval_freq: int = 5000
    eval_using_sim: bool = False
    eval_episodes: int = 5
    eval_execute_steps: int = 8
    ema_decay: float = 0.9999
    checkpoint_dir: str = "checkpoints"


class PolicyTrainer:
    def __init__(
        self,
        policy: Policy,  # e.g., FlowMatchingPolicy
        task: BaseRobotTask,  # e.g., PushTTask
        logger: ExperimentLogger,
        config: TrainConfig,
        device: torch.device,
    ):
        self.policy = policy.to(device)
        self.task = task
        self.logger = logger
        self.config = config
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
        print(f"Starting training for {self.config.training_steps} steps...")

        train_loader = self.task.get_train_loader()
        train_iter = iter(train_loader)

        val_loader = self.task.get_val_loader()
        val_iter = iter(val_loader)

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
                print(
                    f"\n----------- Validating at Step {self.global_step} -------------"
                )
                val_losses = []
                with torch.no_grad():
                    for batch in tqdm(val_loader):
                        batch = self.task.format_batch(raw_batch)
                        loss = self.policy.compute_loss(batch)

                        val_losses.append(loss.item())

                avg_val_loss = torch.as_tensor(val_losses).mean().item()
                self.save_checkpoint("latest", avg_val_loss)
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    self.save_checkpoint("best_model", avg_val_loss)
                    print(f"New best model with Val loss: {self.best_val_loss}")

                if self.config.eval_using_sim:
                    success_rate, _ = evaluate_and_log(
                        policy=self.ema.shadow,
                        task=self.task,
                        logger=self.logger,
                        step=self.global_step,
                        num_episodes=self.config.eval_episodes,
                        execute_steps=self.config.eval_execute_steps,
                    )

                    if success_rate > self.best_success_rate:
                        self.best_success_rate = success_rate
                        self.save_checkpoint("best_model_success", success_rate)
                        print(
                            f"New best model with Success Rate: {self.best_success_rate}"
                        )

                # Use validation loss to check for best model
                # Save latest checkpoint

            self.global_step += 1
            pbar.update(1)

        pbar.close()

        best_model_path = os.path.join(self.config.checkpoint_dir, "best_model.ckpt")
        latest_model_path = os.path.join(self.config.checkpoint_dir, "latest.ckpt")
        best_model_success_path = os.path.join(
            self.config.checkpoint_dir, "best_model_success.ckpt"
        )

        if os.path.exists(best_model_path):
            self.logger.log_model_artifact(
                model_path=best_model_path,
                artifact_name=f"flow_matching_pusht",
                aliases=["best"],
            )
            self.logger.log_model_artifact(
                model_path=best_model_success_path,
                artifact_name=f"flow_matching_pusht",
                aliases=["best_success"],
            )
        elif os.path.exists(latest_model_path):
            # Fallback in case we never beat the initial validation score
            self.logger.log_model_artifact(
                model_path=latest_model_path,
                artifact_name=f"flow_matching_pusht",
                aliases=["latest"],
            )
        self.logger.finish()
        print("Training Complete!")


if __name__ == "__main__":
    from dataclasses import asdict

    from savannah.data.dataset import DataSetConfig
    from savannah.models.backbones.resnet_backbone import ResNetBackbone
    from savannah.models.flow_matching import FlowMatchingPolicy
    from savannah.models.vision_encoder import VisionEncoder
    from savannah.tasks.pusht import PushTTask

    # Bunch of params.
    # TODO: Get all of these from hydra config
    embed_dim = 128
    num_attn_heads = 4
    num_blocks = 6
    feedforward_dim = 4 * embed_dim
    num_cameras = 3

    state_dim = 2
    action_dim = 2
    action_horizon = 8
    obs_horizon = 3

    device = get_device()
    print("Currently using device:", device)

    dataset_config = DataSetConfig(
        repo_id="lerobot/pusht",
        fps=10,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        batch_size=8,
    )

    # 3. Instantiate Architecture
    task = PushTTask(config=dataset_config, device=device)

    # Create Polciy
    resnet_backbone = ResNetBackbone(out_channels=96).to(device)
    vision_encoder = VisionEncoder(backbone=resnet_backbone, embed_dim=embed_dim)
    policy = FlowMatchingPolicy(
        embed_dim,
        num_attn_heads,
        feedforward_dim,
        num_blocks,
        state_dim,
        action_dim,
        action_horizon,
        vision_encoder,
        num_cameras,
    ).to(device)

    print(f"Running model of size: {policy.num_params()}")

    # 2. Setup Configs
    train_config = TrainConfig(
        training_steps=70_000,
        eval_freq=2000,  # Evaluate every 5000 steps
        eval_episodes=1,
        eval_using_sim=True,
    )

    logger = ExperimentLogger(
        project_name="flow-matching-robotics",
        config={"train": asdict(train_config), "model": "FlowMatching"},
        use_wandb=True,
    )

    # 4. Launch Trainer
    trainer = PolicyTrainer(policy, task, logger, train_config, device)

    try:
        trainer.train()
    except Exception as e:
        print(
            f"Encountered Exception while Training: {e}. Traceback: \n{'-' * 10}\n{format_exc()}\n{'-' * 10}\n"
        )
    finally:
        print("Done with Training")
