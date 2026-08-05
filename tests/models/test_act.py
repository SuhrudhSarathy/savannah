# tests/test_act_policy.py
import pytest
import torch

from savannah.models.act import (
    ACT_CVAE_Decoder,
    ACT_CVAE_Encoder,
    ACTPolicy,
    ObservationKey,
)
from savannah.models.backbones.resnet_backbone import ResNetBackbone
from savannah.models.encoders import VisionEncoder

B, H, W = 2, 96, 96
STATE_DIM, CHUNK_SIZE, EMBED_DIM = 14, 50, 256


def _make_policy(num_cameras=1):
    state_dim = STATE_DIM
    embed_dim = EMBED_DIM
    num_attn_heads = 2
    feedforward_dim = 128
    chunk_size = CHUNK_SIZE
    encoder_layers = 1

    backbone = ResNetBackbone(out_channels=32)
    # ACTPolicy has no obs-history dim, so the vision encoder is always
    # constructed with num_obs=1.
    vision_encoder = VisionEncoder(
        backbone, embed_dim=embed_dim, num_cameras=num_cameras, num_obs=1
    )

    cvae_encoder = ACT_CVAE_Encoder(
        state_dim=state_dim,
        encoder_layers=encoder_layers,
        embed_dim=embed_dim,
        num_attn_heads=num_attn_heads,
        feedforward_dim=feedforward_dim,
    )

    cvae_decoder = ACT_CVAE_Decoder(
        state_dim=state_dim,
        encoder_layers=encoder_layers,
        embed_dim=embed_dim,
        num_attn_heads=num_attn_heads,
        feedforward_dim=feedforward_dim,
        chunk_size=chunk_size,
    )

    return ACTPolicy(
        vision_encoder=vision_encoder,
        cvae_encoder=cvae_encoder,
        cvae_decoder=cvae_decoder,
        num_cameras=num_cameras,
    )


@pytest.fixture
def policy():
    return _make_policy(num_cameras=1)


@pytest.fixture
def policy_multi_camera():
    return _make_policy(num_cameras=2)


def test_output_shape_single_camera(policy):
    obs = {
        ObservationKey.images: [torch.randn(B, 3, H, W)],
        ObservationKey.state: torch.randn(B, 1, STATE_DIM),
    }
    out = policy.compute_action(obs)
    assert out.actions.shape == (B, CHUNK_SIZE, STATE_DIM)


def test_output_shape_multi_camera(policy_multi_camera):
    obs = {
        ObservationKey.images: [torch.randn(B, 3, H, W), torch.randn(B, 3, H, W)],
        ObservationKey.state: torch.randn(B, 1, STATE_DIM),
    }
    out = policy_multi_camera.compute_action(obs)
    assert out.actions.shape == (B, CHUNK_SIZE, STATE_DIM)


def test_training_forward_returns_aux(policy):
    obs = {
        ObservationKey.images: [torch.randn(B, 3, H, W)],
        ObservationKey.state: torch.randn(B, 1, STATE_DIM),
        ObservationKey.gt_actions: torch.randn(B, CHUNK_SIZE, STATE_DIM),
    }
    out = policy(obs)
    assert out.aux["mu"] is not None
    assert out.aux["logvar"] is not None
    assert out.aux["mu"].shape == (B, 1, 32)


def test_inference_aux_is_none(policy):
    obs = {
        ObservationKey.images: [torch.randn(B, 3, H, W)],
        ObservationKey.state: torch.randn(B, 1, STATE_DIM),
    }
    out = policy.compute_action(obs)
    assert out.aux["mu"] is None
    assert out.aux["logvar"] is None


def test_loss_is_scalar(policy):
    obs = {
        ObservationKey.images: [torch.randn(B, 3, H, W)],
        ObservationKey.state: torch.randn(B, 1, STATE_DIM),
        ObservationKey.gt_actions: torch.randn(B, CHUNK_SIZE, STATE_DIM),
    }
    out = policy(obs)
    loss = policy.compute_loss(obs, torch.randn(B, CHUNK_SIZE, STATE_DIM), out)
    assert loss.shape == ()  # scalar
    assert not torch.isnan(loss)


def test_fewer_cameras_than_registered_raises(policy_multi_camera):
    # VisionEncoder strictly validates len(images) == num_cameras it was
    # constructed with.
    obs = {
        ObservationKey.images: [torch.randn(B, 3, H, W)],
        ObservationKey.state: torch.randn(B, 1, STATE_DIM),
    }
    with pytest.raises(AssertionError):
        policy_multi_camera.compute_action(obs)
