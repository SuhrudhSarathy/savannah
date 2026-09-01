"""Parametric pipeline tests for the sequence-conditioned policies.

Exercises DITCrossAttnPolicy, LBMPolicy, and UNetPolicy end-to-end against a
real DinoV3Backbone vision encoder in both its "EOS" (single pooled token)
and "without EOS" (patch tokens) modes, crossed with a real CLIP-based
LanguageEncoder being present or absent.

The non-EOS DinoV3Backbone is configured with `reduce=True` so it still
exercises the real patch-token + 2D positional-encoding path, but caps the
token count at a handful of tokens -- LBMPolicy/UNetPolicy pool every vision
token into a single flat conditioning vector, so the raw ~197-patch DINOv3
output would blow up their conditioning MLPs to hundreds of millions of
parameters.
"""

import pytest
import torch

from savannah.models.backbones.dinov3_backbone import DinoV3Backbone
from savannah.models.dit_cross_attn import DITCrossAttnPolicy
from savannah.models.encoders import LanguageEncoder, VisionEncoder
from savannah.models.lbm import LBMPolicy
from savannah.models.unet import UNetPolicy
from savannah.objectives import OverfitObjective
from savannah.utils.observation import ObservationKey

B = 2
IMAGE_SIZE = 224
DINOV3_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"
CLIP_MODEL = "openai/clip-vit-base-patch32"

STATE_DIM, ACTION_DIM, ACTION_HORIZON = 7, 6, 8
EMBED_DIM, TIME_EMBED_DIM, STATE_EMBED_DIM = 32, 8, 8
NUM_CAMERAS, NUM_OBS = 1, 1
REDUCED_TOKENS = 4


@pytest.fixture(scope="module", params=[True, False], ids=["eos", "patches"])
def vision_encoder(request):
    use_eos_only = request.param
    backbone = DinoV3Backbone(
        image_size=IMAGE_SIZE,
        base_model=DINOV3_MODEL,
        use_eos_only=use_eos_only,
        trainable=False,
        reduce=not use_eos_only,
        reduced_tokens=REDUCED_TOKENS,
    )
    return VisionEncoder(
        backbone, embed_dim=EMBED_DIM, num_cameras=NUM_CAMERAS, num_obs=NUM_OBS
    )


@pytest.fixture(
    scope="module", params=[False, True], ids=["no_language", "with_language"]
)
def language_encoder(request):
    if not request.param:
        return None
    return LanguageEncoder(
        embed_dim=EMBED_DIM, base_model=CLIP_MODEL, use_eos_only=True
    )


def _make_dit_cross_attn(vision_encoder, language_encoder):
    return DITCrossAttnPolicy(
        embed_dim=EMBED_DIM,
        time_embed_dim=TIME_EMBED_DIM,
        state_embed_dim=STATE_EMBED_DIM,
        decoder_num_blocks=1,
        decoder_num_attn_heads=4,
        decoder_feedforward_dim=64,
        decoder_dropout=0.0,
        state_dim=STATE_DIM,
        num_obs=NUM_OBS,
        action_dim=ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        num_cameras=NUM_CAMERAS,
        use_rope=True,
        vision_encoder=vision_encoder,
        language_encoder=language_encoder,
        objective=OverfitObjective(),
    )


def _make_lbm(vision_encoder, language_encoder):
    return LBMPolicy(
        embed_dim=EMBED_DIM,
        time_embed_dim=TIME_EMBED_DIM,
        state_embed_dim=STATE_EMBED_DIM,
        decoder_num_blocks=1,
        decoder_num_attn_heads=4,
        decoder_feedforward_dim=64,
        decoder_dropout=0.0,
        state_dim=STATE_DIM,
        num_obs=NUM_OBS,
        action_dim=ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        num_cameras=NUM_CAMERAS,
        use_rope=True,
        vision_encoder=vision_encoder,
        language_encoder=language_encoder,
        objective=OverfitObjective(),
    )


def _make_unet(vision_encoder, language_encoder):
    return UNetPolicy(
        embed_dim=EMBED_DIM,
        unet_channels=[16, 32],
        num_residual_layers=1,
        num_obs=NUM_OBS,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        num_cameras=NUM_CAMERAS,
        vision_encoder=vision_encoder,
        language_encoder=language_encoder,
        objective=OverfitObjective(),
    )


POLICY_BUILDERS = {
    "dit_cross_attn": _make_dit_cross_attn,
    "lbm": _make_lbm,
    "unet": _make_unet,
}


@pytest.fixture(params=list(POLICY_BUILDERS), ids=list(POLICY_BUILDERS))
def policy(request, vision_encoder, language_encoder):
    built = POLICY_BUILDERS[request.param](vision_encoder, language_encoder)
    # Rebuilt fresh per test, but always forced back to eval -- this also
    # resets the module-scoped (and therefore shared-across-tests) vision /
    # language encoder submodules back to eval mode before every test.
    built.eval()
    return built


def make_obs(with_language: bool):
    obs = {
        ObservationKey.images: [
            torch.randn(B, NUM_OBS, 3, IMAGE_SIZE, IMAGE_SIZE)
            for _ in range(NUM_CAMERAS)
        ],
        ObservationKey.state: torch.randn(B, NUM_OBS, STATE_DIM),
        ObservationKey.time: torch.rand(B) * 100.0,
        ObservationKey.actions: torch.randn(B, ACTION_HORIZON, ACTION_DIM),
        ObservationKey.gt_actions: torch.randn(B, ACTION_HORIZON, ACTION_DIM),
    }
    if with_language:
        obs[ObservationKey.language] = [f"instruction {i}" for i in range(B)]
    return obs


def test_output_shape(policy, language_encoder):
    obs = make_obs(with_language=language_encoder is not None)
    out = policy.compute_action(obs)
    assert out.actions.shape == (B, ACTION_HORIZON, ACTION_DIM)


def test_output_has_no_nan_or_inf(policy, language_encoder):
    obs = make_obs(with_language=language_encoder is not None)
    out = policy.compute_action(obs)
    assert not torch.isnan(out.actions).any()
    assert not torch.isinf(out.actions).any()


def test_loss_is_scalar_and_finite(policy, language_encoder):
    obs = make_obs(with_language=language_encoder is not None)
    loss = policy.compute_loss(obs)
    assert loss.shape == ()
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_backward_pass_populates_gradients(policy, language_encoder):
    policy.train()
    obs = make_obs(with_language=language_encoder is not None)
    policy.zero_grad(set_to_none=True)
    loss = policy.compute_loss(obs)
    loss.backward()

    missing_grad = [
        name
        for name, p in policy.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not missing_grad, f"parameters with no gradient: {missing_grad}"
