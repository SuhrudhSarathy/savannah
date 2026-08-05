import pytest
import torch

from savannah.models.backbones.resnet_backbone import ResNetBackbone
from savannah.models.dit import DiTPolicy
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


def build_tiny_policy() -> DiTPolicy:
    backbone = ResNetBackbone(out_channels=16)
    vision_encoder = VisionEncoder(
        backbone, embed_dim=EMBED_DIM, num_cameras=NUM_CAMERAS, num_obs=1
    )
    return DiTPolicy(
        embed_dim=EMBED_DIM,
        encoder_num_blocks=1,
        encoder_num_attn_heads=2,
        encoder_feedforward_dim=64,
        encoder_dropout=0.0,
        decoder_num_blocks=1,
        decoder_num_attn_heads=2,
        decoder_feedforward_dim=64,
        decoder_dropout=0.0,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        num_cameras=NUM_CAMERAS,
        vision_encoder=vision_encoder,
        language_encoder=None,
        objective=OverfitObjective(),
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
        task=None,
        logger=exp_logger,
        config=config,
        device=torch.device("cpu"),
    )


def take_optimizer_step(trainer: PolicyTrainer):
    trainer.policy.train()
    batch = {
        ObservationKey.images: [
            torch.randn(2, 1, 3, 64, 64) for _ in range(NUM_CAMERAS)
        ],
        ObservationKey.state: torch.randn(2, 1, STATE_DIM),
        ObservationKey.time: torch.rand(2) * 100.0,
        ObservationKey.actions: torch.randn(2, ACTION_HORIZON, ACTION_DIM),
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
    resumed.load_checkpoint(str(tmp_path / "latest" / "latest.ckpt"))

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
    assert (tmp_path / str(step_a) / "latest.ckpt").exists()
    assert (tmp_path / str(step_b) / "latest.ckpt").exists()

    # The "latest" mirror reflects only the most recent save.
    latest_ckpt = torch.load(tmp_path / "latest" / "latest.ckpt", map_location="cpu")
    assert latest_ckpt["global_step"] == step_b
    assert latest_ckpt["success_rate"] == pytest.approx(0.8)


def test_load_checkpoint_falls_back_for_missing_best_metric_keys(tmp_path):
    trainer = build_trainer(tmp_path)
    take_optimizer_step(trainer)

    checkpoint = {
        "model_state_dict": trainer.policy.state_dict(),
        "ema_state_dict": trainer.ema.shadow.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "scheduler_state_dict": trainer.scheduler.state_dict(),
        "global_step": trainer.global_step,
        "success_rate": 0.0,
    }
    path = tmp_path / "legacy.ckpt"
    torch.save(checkpoint, path)

    resumed = build_trainer(tmp_path)
    default_best_val_loss = resumed.best_val_loss
    default_best_success_rate = resumed.best_success_rate

    resumed.load_checkpoint(str(path))

    assert resumed.global_step == trainer.global_step
    assert resumed.best_val_loss == default_best_val_loss
    assert resumed.best_success_rate == default_best_success_rate
