import pytest
import torch

from savannah.models.backbones.resnet_backbone import ResNetBackbone
from savannah.models.encoders import VisionEncoder
from savannah.models.unet import UNetPolicy
from savannah.objectives import FlowMatchingObjective, OverfitObjective
from savannah.utils.observation import ObservationKey

B, H, W = 2, 64, 64
STATE_DIM, ACTION_DIM, ACTION_HORIZON, EMBED_DIM = 7, 6, 8, 64
NUM_CAMERAS = 2
NUM_OBS = 1
UNET_CHANNELS = [32, 64]


def _make_vision_encoder(num_obs=NUM_OBS):
    backbone = ResNetBackbone(out_channels=32)
    return VisionEncoder(
        backbone, embed_dim=EMBED_DIM, num_cameras=NUM_CAMERAS, num_obs=num_obs
    )


def _make_policy(num_obs=NUM_OBS, objective=None):
    policy = UNetPolicy(
        embed_dim=EMBED_DIM,
        unet_channels=UNET_CHANNELS,
        num_residual_layers=1,
        num_obs=num_obs,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        num_cameras=NUM_CAMERAS,
        vision_encoder=_make_vision_encoder(num_obs=num_obs),
        objective=objective or OverfitObjective(),
    )
    policy.eval()
    return policy


@pytest.fixture
def policy():
    return _make_policy()


@pytest.fixture
def policy_flow():
    return _make_policy(objective=FlowMatchingObjective())


@pytest.fixture
def policy_multi_obs():
    return _make_policy(num_obs=2)


def make_obs(num_cameras=NUM_CAMERAS, num_obs=NUM_OBS):
    return {
        ObservationKey.images: [
            torch.randn(B, num_obs, 3, H, W) for _ in range(num_cameras)
        ],
        ObservationKey.state: torch.randn(B, num_obs, STATE_DIM),
        ObservationKey.time: torch.rand(B) * 100.0,
        ObservationKey.actions: torch.randn(B, ACTION_HORIZON, ACTION_DIM),
        ObservationKey.gt_actions: torch.randn(B, ACTION_HORIZON, ACTION_DIM),
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


def test_fewer_cameras_than_registered_raises(policy):
    obs = make_obs(num_cameras=1)
    with pytest.raises(AssertionError):
        policy.compute_action(obs)


def test_multi_obs_conditioning(policy_multi_obs):
    obs = make_obs(num_obs=2)
    out = policy_multi_obs.compute_action(obs)
    assert out.actions.shape == (B, ACTION_HORIZON, ACTION_DIM)


def test_loss_is_scalar_and_finite(policy):
    obs = make_obs()
    loss = policy.compute_loss(obs)
    assert loss.shape == ()
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_loss_is_scalar_and_finite_flow_matching(policy_flow):
    obs = make_obs()
    loss = policy_flow.compute_loss(obs)
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


def test_missing_noisy_actions_raises(policy):
    obs = make_obs()
    with pytest.raises(AssertionError):
        policy.forward(obs)


def test_n_vision_tokens_matches_vision_encoder(policy):
    assert policy.n_vision_tokens == policy.vision_encoder.num_tokens * EMBED_DIM
