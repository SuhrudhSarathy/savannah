# tests/test_act_policy.py
import torch
import pytest
from savannah.models.act import (
    ACTPolicy,
    ObservationKey,
    ACT_CVAE_Encoder,
    ACT_CVAE_Decoder,
)
from savannah.models.vision_encoder import VisionEncoder
from savannah.models.backbones.resnet_backbone import ResNetBackbone

B, H, W = 2, 96, 96
STATE_DIM, CHUNK_SIZE, EMBED_DIM = 14, 50, 256


@pytest.fixture
def policy():
    state_dim = STATE_DIM
    embed_dim = EMBED_DIM
    num_attn_heads = 2
    feedforward_dim = 128
    chunk_size = CHUNK_SIZE
    encoder_layers = 1
    num_cameras = 2

    backbone = ResNetBackbone(out_channels=32)
    vision_encoder = VisionEncoder(backbone, embed_dim=embed_dim)

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

    policy = ACTPolicy(
        vision_encoder=vision_encoder,
        cvae_encoder=cvae_encoder,
        cvae_decoder=cvae_decoder,
        num_cameras=num_cameras,
    )
    return policy


def test_output_shape_single_camera(policy):
    obs = {
        ObservationKey.images: [torch.randn(B, 3, H, W)],
        ObservationKey.state: torch.randn(B, 1, STATE_DIM),
    }
    out = policy.compute_action(obs)
    assert out.actions.shape == (B, CHUNK_SIZE, STATE_DIM)


def test_output_shape_multi_camera(policy):
    obs = {
        ObservationKey.images: [torch.randn(B, 3, H, W), torch.randn(B, 3, H, W)],
        ObservationKey.state: torch.randn(B, 1, STATE_DIM),
    }
    out = policy.compute_action(obs)
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
