"""configs/model/dit_cross_attn_with_dino.yaml composed with the franka_sim
tasks, built and run end to end -- offline: the gated DINOv3 checkpoint is
swapped for a tiny randomly initialized DINOv3 of the same architecture, and
the dataset metadata is faked."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.errors import InstantiationException
from hydra.utils import instantiate
from transformers import DINOv3ViTConfig, DINOv3ViTModel

import savannah.models.backbones.dinov3_backbone as dinov3_backbone
import savannah.models.backbones.image_normalizer as image_normalizer
import savannah.tasks.franka_sim as franka_sim
from savannah.factory import build_policy  # registers the len/eval resolvers
from savannah.models.backbones.dinov3_backbone import DinoV3Backbone

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
B = 2


def tiny_dinov3(*args, **kwargs):
    # Same layout as dinov3-vits16 (patch 16, 4 register tokens), but tiny.
    config = DINOv3ViTConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        num_register_tokens=4,
    )
    return DINOv3ViTModel(config)


def no_processor(*args, **kwargs):
    raise OSError("offline")  # ImageNormalizer falls back to ImageNet stats


@pytest.fixture(autouse=True)
def offline_dinov3(monkeypatch):
    monkeypatch.setattr(dinov3_backbone.AutoModel, "from_pretrained", tiny_dinov3)
    monkeypatch.setattr(image_normalizer.AutoProcessor, "from_pretrained", no_processor)


def compose_cfg(task: str, *overrides: str):
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        return compose(
            config_name="config",
            overrides=[
                "model=dit_cross_attn_with_dino",
                f"task={task}",
                "model.embed_dim=32",
                "model.decoder_num_blocks=1",
                "model.vision_encoder.attn_pooling_num_attn_heads=2",
                *overrides,
            ],
        )


def fake_meta(cameras):
    features = {
        franka_sim.POSE_KEY: {"shape": [7]},
        franka_sim.ACTION_KEY: {"shape": [8]},
        **{f"observation.images.{c}": {"shape": [256, 256, 3]} for c in cameras},
    }
    stats = {
        franka_sim.ACTION_KEY: {
            "min": [0.2, -0.5, 0.05, 0.0, 0.85, -0.5, 0.0, 0.0],
            "max": [0.7, 0.5, 0.6, 0.0, 1.0, 0.5, 0.0, 1.0],
        }
    }
    return SimpleNamespace(features=features, stats=stats)


def dataset_batch(cfg):
    """A LeRobot-shaped batch for FrankaSimTask.format_batch."""
    data = cfg.task.config
    oh, horizon, size = data.obs_horizon, cfg.model.action_horizon, data.image_size
    pose = torch.tensor([0.4, 0.0, 0.3, 0.0, 1.0, 0.0, 0.0])
    command = torch.cat([pose, torch.tensor([1.0])])
    batch = {
        franka_sim.POSE_KEY: pose.expand(B, oh, 7).clone(),
        franka_sim.ACTION_KEY: command.expand(B, oh + horizon, 8).clone(),
    }
    for cam in data.cameras:
        batch[f"observation.images.{cam}"] = torch.rand(B, oh, 3, size, size)
    return batch


@pytest.mark.parametrize("image_size", [224, 128])
def test_dino_policy_trains_and_acts_on_hammer_nails(monkeypatch, image_size):
    cfg = compose_cfg("franka_hammer_nails", f"task.config.image_size={image_size}")
    assert cfg.model.state_dim == 8 and cfg.model.action_dim == 8
    assert cfg.model.num_cameras == 3  # front, wrist, target_mask

    monkeypatch.setattr(
        franka_sim,
        "_load_lerobot_meta",
        lambda repo_id: fake_meta(cfg.task.config.cameras),
    )
    task = instantiate(cfg.task, device=torch.device("cpu"))
    policy = build_policy(cfg)
    assert isinstance(policy.vision_encoder.backbone, DinoV3Backbone)
    assert policy.vision_encoder.backbone.tokens_per_image == (
        (image_size // 16) ** 2 + 1
    )

    batch = task.format_batch(dataset_batch(cfg))
    loss = policy.compute_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()

    policy.eval()
    with torch.no_grad():
        actions = policy.compute_action(batch).actions
    assert actions.shape == (B, cfg.model.action_horizon, 8)
    assert task.postprocess_action(actions[0, 0]).shape == (8,)


def test_metaworld_default_keeps_its_dims():
    cfg = compose_cfg("metaworld_assembly")
    assert cfg.model.state_dim == 4 and cfg.model.action_dim == 4


def test_franka_task_rejects_other_dims(monkeypatch):
    cfg = compose_cfg("franka_hammer_nails", "task.state_dim=4")
    with pytest.raises(InstantiationException, match="8-dim"):
        instantiate(cfg.task, device=torch.device("cpu"))


def test_image_size_must_be_a_multiple_of_the_patch_size():
    with pytest.raises(ValueError, match="divisible"):
        DinoV3Backbone(image_size=100, use_eos_only=False)
