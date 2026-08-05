import pytest
import torch
import torch.nn as nn

from savannah.models.backbones.resnet_backbone import ResNetBackbone
from savannah.models.dit_cross_attn import DiTCrossAttnPolicy
from savannah.models.encoders import VisionEncoder
from savannah.objectives import FlowMatchingObjective, OverfitObjective
from savannah.utils.observation import ObservationKey

B, H, W = 2, 64, 64
STATE_DIM, ACTION_DIM, ACTION_HORIZON, EMBED_DIM = 7, 6, 8, 64
NUM_CAMERAS = 2


class StubLanguageEncoder(nn.Module):
    """Lightweight stand-in for LanguageEncoder (avoids downloading CLIP weights)."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Linear(1, embed_dim)

    def forward(self, text_inputs: list[str]) -> torch.Tensor:
        batch = len(text_inputs)
        dummy = torch.ones(batch, 1, 1)
        return self.proj(dummy)


def _make_vision_encoder():
    backbone = ResNetBackbone(out_channels=32)
    return VisionEncoder(
        backbone, embed_dim=EMBED_DIM, num_cameras=NUM_CAMERAS, num_obs=1
    )


def _make_policy(objective=None, language_encoder=None):
    policy = DiTCrossAttnPolicy(
        embed_dim=EMBED_DIM,
        num_attn_heads=4,
        feedforward_dim=128,
        num_blocks=2,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        num_cameras=NUM_CAMERAS,
        vision_encoder=_make_vision_encoder(),
        language_encoder=language_encoder,
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
def policy_with_language():
    return _make_policy(language_encoder=StubLanguageEncoder(EMBED_DIM))


def make_obs(num_cameras=NUM_CAMERAS, with_language=False):
    obs = {
        ObservationKey.images: [torch.randn(B, 1, 3, H, W) for _ in range(num_cameras)],
        ObservationKey.state: torch.randn(B, 1, STATE_DIM),
        ObservationKey.time: torch.rand(B) * 100.0,
        ObservationKey.actions: torch.randn(B, ACTION_HORIZON, ACTION_DIM),
        ObservationKey.gt_actions: torch.randn(B, ACTION_HORIZON, ACTION_DIM),
    }
    if with_language:
        obs[ObservationKey.language] = [f"instruction {i}" for i in range(B)]
    return obs


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


def test_with_language_encoder(policy_with_language):
    obs = make_obs(with_language=True)
    out = policy_with_language.compute_action(obs)
    assert out.actions.shape == (B, ACTION_HORIZON, ACTION_DIM)
