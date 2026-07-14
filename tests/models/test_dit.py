import pytest
import torch

from savannah.models.backbones.resnet_backbone import ResNetBackbone
from savannah.models.dit import DiTPolicy
from savannah.models.encoders import VisionEncoder
from savannah.objectives import OverfitObjective
from savannah.utils.observation import ObservationKey

B, H, W = 2, 64, 64
STATE_DIM, ACTION_DIM, ACTION_HORIZON, EMBED_DIM = 7, 6, 8, 64
NUM_CAMERAS = 2


@pytest.fixture
def policy():
    backbone = ResNetBackbone(out_channels=32)
    vision_encoder = VisionEncoder(backbone, embed_dim=EMBED_DIM)

    policy = DiTPolicy(
        embed_dim=EMBED_DIM,
        encoder_num_blocks=2,
        encoder_num_attn_heads=4,
        encoder_feedforward_dim=128,
        encoder_dropout=0.0,
        decoder_num_blocks=2,
        decoder_num_attn_heads=4,
        decoder_feedforward_dim=128,
        decoder_dropout=0.0,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        num_cameras=NUM_CAMERAS,
        vision_encoder=vision_encoder,
        language_encoder=None,
        objective=OverfitObjective(),
    )
    policy.eval()
    return policy


def make_obs(num_cameras=NUM_CAMERAS):
    return {
        ObservationKey.images: [torch.randn(B, 1, 3, H, W) for _ in range(num_cameras)],
        ObservationKey.state: torch.randn(B, 1, STATE_DIM),
        ObservationKey.time: torch.rand(B) * 100.0,
        ObservationKey.actions: torch.randn(B, ACTION_HORIZON, ACTION_DIM),
    }


def test_output_shape(policy):
    obs = make_obs()
    out = policy.compute_action(obs)
    assert out.actions.shape == (B, ACTION_HORIZON, ACTION_DIM)


def test_output_has_no_nan_or_inf(policy):
    obs = make_obs()
    out = policy.compute_action(obs)
    assert not torch.isnan(out.actions).any()
    assert not torch.isinf(out.actions).any()


def test_fewer_cameras_than_registered(policy):
    # camera_embedding has NUM_CAMERAS rows; passing fewer cameras than that
    # just uses a prefix of cam indices and should still work.
    obs = make_obs(num_cameras=1)
    out = policy.compute_action(obs)
    assert out.actions.shape == (B, ACTION_HORIZON, ACTION_DIM)


def test_loss_is_scalar_and_finite(policy):
    obs = make_obs()
    loss = policy.compute_loss(obs)
    assert loss.shape == ()
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_backward_pass_populates_gradients(policy):
    policy.train()
    obs = make_obs()
    loss = policy.compute_loss(obs)
    loss.backward()

    missing_grad = [
        name
        for name, p in policy.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not missing_grad, f"parameters with no gradient: {missing_grad}"


def test_mask_shape_and_layout(policy):
    obs = make_obs()
    policy.compute_action(obs)

    n_cond = policy.mask.shape[0] - ACTION_HORIZON
    assert policy.mask.dtype == torch.bool
    assert policy.mask[:n_cond, :n_cond].all()
    assert policy.mask[n_cond:, :n_cond].all()
    assert policy.mask[n_cond:, n_cond:].all()
    assert not policy.mask[:n_cond, n_cond:].any()


def test_language_encoder_none_is_safe(policy):
    assert policy.language_encoder is None
    obs = make_obs()
    obs.pop(ObservationKey.language, None)
    out = policy.compute_action(obs)
    assert out.actions.shape == (B, ACTION_HORIZON, ACTION_DIM)
