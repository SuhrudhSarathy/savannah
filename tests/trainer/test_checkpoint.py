import pytest
import torch
from dummies import SyntheticTask
from omegaconf import OmegaConf

from savannah.bundle import read_bundle, save_bundle
from savannah.data.dataset import DataSetConfig
from savannah.models.backbones import DummyVisionBackbone
from savannah.models.dit_cross_attn import DITCrossAttnPolicy
from savannah.models.encoders import VisionEncoder
from savannah.objectives import OverfitObjective
from savannah.trainer.trainer import (
    CheckpointParams,
    EvalParams,
    PolicyTrainer,
    TrainConfig,
    TrainParams,
)
from savannah.utils.logger import ExperimentLogger
from savannah.utils.observation import ObservationKey

STATE_DIM, ACTION_DIM, ACTION_HORIZON, EMBED_DIM = 7, 6, 8, 32
NUM_CAMERAS = 1


def build_tiny_policy() -> DITCrossAttnPolicy:
    backbone = DummyVisionBackbone(out_channels=16)
    vision_encoder = VisionEncoder(
        backbone, embed_dim=EMBED_DIM, num_cameras=NUM_CAMERAS, num_obs=1
    )
    return DITCrossAttnPolicy(
        embed_dim=EMBED_DIM,
        time_embed_dim=8,
        state_embed_dim=8,
        decoder_num_blocks=1,
        decoder_num_attn_heads=2,
        decoder_feedforward_dim=64,
        state_dim=STATE_DIM,
        num_obs=1,
        action_dim=ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        num_cameras=NUM_CAMERAS,
        use_rope=True,
        vision_encoder=vision_encoder,
        language_encoder=None,
        objective=OverfitObjective(),
    )


def build_task() -> SyntheticTask:
    return SyntheticTask(
        DataSetConfig(repo_id="synthetic", fps=10, cameras=["cam0"], image_size=64),
        torch.device("cpu"),
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
    )


RUN_CFG = OmegaConf.create(
    {
        "model": {
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "action_horizon": ACTION_HORIZON,
        },
        "task": {
            "config": {
                "cameras": ["cam0"],
                "image_size": 64,
                "obs_horizon": 1,
                "fps": 10,
            }
        },
    }
)


def build_trainer(checkpoint_dir) -> PolicyTrainer:
    config = TrainConfig(
        train=TrainParams(
            lr=1e-3,
            training_steps=10,
            warmup_steps=1,
            gradient_accumulation_steps=1,
        ),
        eval=EvalParams(),
        checkpoint=CheckpointParams(checkpoint_dir=str(checkpoint_dir)),
    )
    exp_logger = ExperimentLogger(project_name="test", config={}, enable_logging=False)
    return PolicyTrainer(
        policy=build_tiny_policy(),
        task=build_task(),
        logger=exp_logger,
        config=config,
        device=torch.device("cpu"),
        run_cfg=RUN_CFG,
    )


def take_optimizer_step(trainer: PolicyTrainer):
    trainer.policy.train()
    batch = {
        ObservationKey.images: [
            torch.randn(2, 1, 3, 64, 64) for _ in range(NUM_CAMERAS)
        ],
        ObservationKey.state: torch.randn(2, 1, STATE_DIM),
        ObservationKey.gt_actions: torch.randn(2, ACTION_HORIZON, ACTION_DIM),
    }
    loss = trainer.policy.compute_loss(batch)
    loss.backward()
    trainer.optimizer.step()
    trainer.scheduler.step()
    trainer.ema.update()
    trainer.optimizer.zero_grad()
    trainer.global_step += 1


def test_load_checkpoint_restores_training_state(tmp_path):
    trainer = build_trainer(tmp_path)
    for _ in range(3):
        take_optimizer_step(trainer)
    trainer.best_val_loss = 0.5
    trainer.best_success_rate = 0.75
    trainer.save_checkpoint("latest", trainer.best_val_loss)

    resumed = build_trainer(tmp_path)
    resumed.load_checkpoint(str(tmp_path / "latest" / "latest.bundle"))

    assert resumed.global_step == trainer.global_step
    assert resumed.best_val_loss == pytest.approx(trainer.best_val_loss)
    assert resumed.best_success_rate == pytest.approx(trainer.best_success_rate)

    for p1, p2 in zip(
        trainer.policy.parameters(), resumed.policy.parameters(), strict=True
    ):
        assert torch.allclose(p1, p2)

    for p1, p2 in zip(
        trainer.ema.shadow.parameters(),
        resumed.ema.shadow.parameters(),
        strict=True,
    ):
        assert torch.allclose(p1, p2)

    orig_opt_state = trainer.optimizer.state_dict()
    resumed_opt_state = resumed.optimizer.state_dict()
    assert orig_opt_state["param_groups"] == resumed_opt_state["param_groups"]

    assert (
        trainer.scheduler.state_dict()["last_epoch"]
        == resumed.scheduler.state_dict()["last_epoch"]
    )


def test_save_checkpoint_retains_per_step_history_and_updates_latest(tmp_path):
    trainer = build_trainer(tmp_path)

    take_optimizer_step(trainer)
    step_a = trainer.global_step
    trainer.save_checkpoint("latest", 0.9)

    take_optimizer_step(trainer)
    step_b = trainer.global_step
    trainer.save_checkpoint("latest", 0.8)

    # Both step archives are retained, not overwritten.
    assert (tmp_path / str(step_a) / "latest.bundle").exists()
    assert (tmp_path / str(step_b) / "latest.bundle").exists()

    # The "latest" mirror reflects only the most recent save.
    latest = read_bundle(tmp_path / "latest" / "latest.bundle").training_state
    assert latest["global_step"] == step_b
    assert latest["success_rate"] == pytest.approx(0.8)


def test_load_checkpoint_falls_back_for_missing_best_metric_keys(tmp_path):
    trainer = build_trainer(tmp_path)
    take_optimizer_step(trainer)

    training_state = {
        "model_state_dict": trainer.policy.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "scheduler_state_dict": trainer.scheduler.state_dict(),
        "global_step": trainer.global_step,
        "success_rate": 0.0,
    }
    path = tmp_path / "legacy.bundle"
    save_bundle(
        path,
        RUN_CFG,
        trainer.task,
        trainer.ema.shadow.state_dict(),
        training_state=training_state,
    )

    resumed = build_trainer(tmp_path)
    default_best_val_loss = resumed.best_val_loss
    default_best_success_rate = resumed.best_success_rate

    resumed.load_checkpoint(str(path))

    assert resumed.global_step == trainer.global_step
    assert resumed.best_val_loss == default_best_val_loss
    assert resumed.best_success_rate == default_best_success_rate
