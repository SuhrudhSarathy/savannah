import pytest
import torch
import torch.nn as nn

from savannah.models.backbones.resnet_backbone import ResNetBackbone
from savannah.models.encoders import VisionEncoder
from savannah.models.lbm import DiTBlock, LBMPolicy
from savannah.objectives import FlowMatchingObjective, OverfitObjective
from savannah.utils.observation import ObservationKey

B, H, W = 2, 64, 64
STATE_DIM, ACTION_DIM, ACTION_HORIZON, EMBED_DIM = 7, 6, 8, 64
NUM_CAMERAS = 2
NUM_OBS = 1


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
    return VisionEncoder(backbone, embed_dim=EMBED_DIM)


def _make_policy(num_obs=NUM_OBS, objective=None, language_encoder=None):
    policy = LBMPolicy(
        embed_dim=EMBED_DIM,
        decoder_num_blocks=2,
        decoder_num_attn_heads=4,
        decoder_feedforward_dim=128,
        decoder_dropout=0.0,
        state_dim=STATE_DIM,
        num_obs=num_obs,
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
def policy_multi_obs():
    return _make_policy(num_obs=2)


@pytest.fixture
def policy_with_language():
    return _make_policy(language_encoder=StubLanguageEncoder(EMBED_DIM))


def make_obs(num_cameras=NUM_CAMERAS, num_obs=NUM_OBS, with_language=False):
    obs = {
        ObservationKey.images: [
            torch.randn(B, num_obs, 3, H, W) for _ in range(num_cameras)
        ],
        ObservationKey.state: torch.randn(B, num_obs, STATE_DIM),
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


def test_language_provided_without_encoder_warns(policy):
    assert policy.language_encoder is None
    obs = make_obs(with_language=True)
    with pytest.raises(RuntimeError):
        policy.compute_action(obs)


def test_with_language_encoder(policy_with_language):
    obs = make_obs(with_language=True)
    out = policy_with_language.compute_action(obs)
    assert out.actions.shape == (B, ACTION_HORIZON, ACTION_DIM)


def test_multi_obs_conditioning(policy_multi_obs):
    obs = make_obs(num_obs=2)
    out = policy_multi_obs.compute_action(obs)
    assert out.actions.shape == (B, ACTION_HORIZON, ACTION_DIM)


def test_condition_dim_matches_decoder_adaln_input(policy):
    n_language = 0
    n_vision = NUM_CAMERAS * NUM_OBS * policy.vision_encoder.tokens_per_image
    n_state = NUM_OBS
    n_time = 1
    expected_condition_dim = (n_language + n_vision + n_state + n_time) * EMBED_DIM

    assert policy.condition_dim == expected_condition_dim
    assert policy.decoder[0].adaln_block.in_features == expected_condition_dim


def test_decoder_blocks_adaln_zero_init(policy):
    for block in policy.decoder:
        assert torch.equal(
            block.adaln_block.weight, torch.zeros_like(block.adaln_block.weight)
        )
        assert torch.equal(
            block.adaln_block.bias, torch.zeros_like(block.adaln_block.bias)
        )


DIT_BLOCK_EMBED_DIM, COND_DIM, NUM_HEADS, FF_DIM, T = 32, 48, 4, 64, 5


@pytest.fixture
def dit_block():
    return DiTBlock(
        embed_dim=DIT_BLOCK_EMBED_DIM,
        cond_dim=COND_DIM,
        num_attn_heads=NUM_HEADS,
        feedforward_dim=FF_DIM,
        dropout=0.0,
    )


def test_dit_block_output_shape(dit_block):
    x = torch.randn(B, T, DIT_BLOCK_EMBED_DIM)
    x_cond = torch.randn(B, 1, COND_DIM)
    out = dit_block(x, x_cond)
    assert out.shape == x.shape


def test_dit_block_zero_init_is_identity(dit_block):
    dit_block.eval()
    x = torch.randn(B, T, DIT_BLOCK_EMBED_DIM)
    x_cond = torch.randn(B, 1, COND_DIM)
    out = dit_block(x, x_cond)
    assert torch.allclose(out, x)


def test_dit_block_gradient_flow(dit_block):
    dit_block.train()
    x = torch.randn(B, T, DIT_BLOCK_EMBED_DIM)
    x_cond = torch.randn(B, 1, COND_DIM)
    out = dit_block(x, x_cond)
    out.sum().backward()

    missing_grad = [
        name
        for name, p in dit_block.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not missing_grad, f"parameters with no gradient: {missing_grad}"
