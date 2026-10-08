"""Integration tests: every sequence-conditioned policy x every real vision
backbone (pretrained weights, downloaded on first run) x with/without the real
CLIP-based LanguageEncoder.

The unit tests in tests/models/ cover each policy's own logic against
DummyVisionBackbone; this module checks the real backbones plug into every
policy -- token counts, output dims, frozen-vs-trainable gradient boundaries --
the way the configs in configs/model/ combine them.

ACTPolicy is not included: it takes single-frame (B, 3, H, W) images and has
its own compute_loss signature, so it doesn't share this Policy API (and can't
be trained by PolicyTrainer).
"""

from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch

from savannah.models.backbones import VisionFeatureExtractor
from savannah.models.backbones.clip_backbone import CLIPBackbone
from savannah.models.backbones.dinov3_backbone import DinoV3Backbone
from savannah.models.backbones.lora_resnet_backbone import LoRAResNetBackbone
from savannah.models.backbones.resnet_backbone import ResNetBackbone
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
BACKBONE_OUT = 32

# LBMPolicy/UNetPolicy flatten every vision token into one conditioning
# vector, so DINOv3's ~197 patch tokens would blow their conditioning MLPs up
# to hundreds of millions of parameters. Above this many tokens per image,
# attention pooling caps it (as configs/model/dit_cross_attn_with_dino.yaml
# does); below it, the backbone's raw token grid is used directly.
MAX_RAW_TOKENS_PER_IMAGE = 64
ATTN_POOLING_NUM_QUERIES = 4


@dataclass(frozen=True)
class BackboneSpec:
    build: Callable[[], VisionFeatureExtractor]
    # Frozen HF backbones (trainable=False) run under torch.no_grad(), so
    # gradient intentionally never reaches the raw pixels.
    pixels_get_grad: bool
    # Trainable backbone params that legitimately never receive a gradient
    # (matched by name suffix).
    unused_params: tuple[str, ...] = ()


BACKBONES = {
    "resnet_pooled": BackboneSpec(
        lambda: ResNetBackbone(
            out_channels=BACKBONE_OUT, image_size=IMAGE_SIZE, use_spatial_softmax=True
        ),
        pixels_get_grad=True,
    ),
    "resnet_spatial": BackboneSpec(
        lambda: ResNetBackbone(
            out_channels=BACKBONE_OUT, image_size=IMAGE_SIZE, use_spatial_softmax=False
        ),
        pixels_get_grad=True,
    ),
    "lora_resnet_pooled": BackboneSpec(
        lambda: LoRAResNetBackbone(
            out_channels=BACKBONE_OUT, image_size=IMAGE_SIZE, use_spatial_softmax=True
        ),
        pixels_get_grad=True,
    ),
    "lora_resnet_spatial": BackboneSpec(
        lambda: LoRAResNetBackbone(
            out_channels=BACKBONE_OUT, image_size=IMAGE_SIZE, use_spatial_softmax=False
        ),
        pixels_get_grad=True,
    ),
    "dinov3_eos": BackboneSpec(
        lambda: DinoV3Backbone(
            IMAGE_SIZE, DINOV3_MODEL, use_eos_only=True, trainable=False
        ),
        pixels_get_grad=False,
    ),
    "dinov3_patches": BackboneSpec(
        lambda: DinoV3Backbone(
            IMAGE_SIZE, DINOV3_MODEL, use_eos_only=False, trainable=False
        ),
        pixels_get_grad=False,
    ),
    "dinov3_patches_trainable": BackboneSpec(
        lambda: DinoV3Backbone(
            IMAGE_SIZE, DINOV3_MODEL, use_eos_only=False, trainable=True
        ),
        pixels_get_grad=True,
        # Only used for masked-image-modeling pretraining.
        unused_params=("embeddings.mask_token",),
    ),
    "clip_eos": BackboneSpec(
        lambda: CLIPBackbone(
            IMAGE_SIZE, CLIP_MODEL, use_eos_only=True, trainable=False
        ),
        pixels_get_grad=False,
    ),
    "clip_patches": BackboneSpec(
        lambda: CLIPBackbone(
            IMAGE_SIZE, CLIP_MODEL, use_eos_only=False, trainable=False
        ),
        pixels_get_grad=False,
    ),
}


@pytest.fixture(scope="module", params=list(BACKBONES), ids=list(BACKBONES))
def backbone_spec(request):
    # Module-scoped: each pretrained backbone is loaded once and shared by
    # every policy/language combination below.
    spec = BACKBONES[request.param]
    torch.manual_seed(0)
    return spec, spec.build()


@pytest.fixture(
    scope="module", params=[False, True], ids=["no_language", "with_language"]
)
def language_encoder(request):
    if not request.param:
        return None
    return LanguageEncoder(
        embed_dim=EMBED_DIM, base_model=CLIP_MODEL, use_eos_only=True
    )


def _make_vision_encoder(backbone: VisionFeatureExtractor) -> VisionEncoder:
    return VisionEncoder(
        backbone,
        embed_dim=EMBED_DIM,
        num_cameras=NUM_CAMERAS,
        num_obs=NUM_OBS,
        enable_attn_pooling=backbone.tokens_per_image > MAX_RAW_TOKENS_PER_IMAGE,
        attn_pooling_num_queries=ATTN_POOLING_NUM_QUERIES,
    )


def _make_dit_cross_attn(vision_encoder, language_encoder):
    return DITCrossAttnPolicy(
        embed_dim=EMBED_DIM,
        time_embed_dim=TIME_EMBED_DIM,
        state_embed_dim=STATE_EMBED_DIM,
        decoder_num_blocks=1,
        decoder_num_attn_heads=4,
        decoder_feedforward_dim=64,
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
def policy(request, backbone_spec, language_encoder):
    _, backbone = backbone_spec
    built = POLICY_BUILDERS[request.param](
        _make_vision_encoder(backbone), language_encoder
    )
    # Rebuilt fresh per test, but always forced back to eval -- this also
    # resets the module-scoped (and therefore shared-across-tests) backbone /
    # language encoder submodules back to eval mode before every test.
    built.eval()
    return built


def make_obs(with_language: bool, requires_grad: bool = False):
    obs = {
        ObservationKey.images: [
            torch.rand(
                B, NUM_OBS, 3, IMAGE_SIZE, IMAGE_SIZE, requires_grad=requires_grad
            )
            for _ in range(NUM_CAMERAS)
        ],
        ObservationKey.state: torch.randn(
            B, NUM_OBS, STATE_DIM, requires_grad=requires_grad
        ),
        ObservationKey.time: (torch.rand(B) * 100.0).requires_grad_(requires_grad),
        ObservationKey.actions: torch.randn(
            B, ACTION_HORIZON, ACTION_DIM, requires_grad=requires_grad
        ),
        ObservationKey.gt_actions: torch.randn(B, ACTION_HORIZON, ACTION_DIM),
    }
    if with_language:
        obs[ObservationKey.language] = [f"instruction {i}" for i in range(B)]
    return obs


def test_compute_action_shape_and_finite(policy, language_encoder):
    obs = make_obs(with_language=language_encoder is not None)
    out = policy.compute_action(obs)
    assert out.actions.shape == (B, ACTION_HORIZON, ACTION_DIM)
    assert torch.isfinite(out.actions).all()


def test_loss_is_scalar_and_finite(policy, language_encoder):
    obs = make_obs(with_language=language_encoder is not None)
    loss = policy.compute_loss(obs)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_backward_pass_populates_gradients(policy, backbone_spec, language_encoder):
    spec, _ = backbone_spec
    policy.train()
    obs = make_obs(with_language=language_encoder is not None, requires_grad=True)
    policy.zero_grad(set_to_none=True)
    loss = policy.compute_loss(obs)
    loss.backward()

    missing_grad = [
        name
        for name, p in policy.named_parameters()
        if p.requires_grad and p.grad is None and not name.endswith(spec.unused_params)
    ]
    assert not missing_grad, f"parameters with no gradient: {missing_grad}"

    # x_state / x_time / the noisy-action input all flow through trainable
    # encoders (StateEncoder, TimeEmbedding, action_embedding), so gradient
    # must reach all the way back to these leaf observation tensors.
    missing_source_grad = [
        name
        for name, tensor in {
            "state": obs[ObservationKey.state],
            "time": obs[ObservationKey.time],
            "actions": obs[ObservationKey.actions],
        }.items()
        if tensor.grad is None
    ]
    assert not missing_source_grad, (
        f"observation inputs with no gradient reaching them: {missing_source_grad}"
    )

    # Pixels get gradient iff the backbone is differentiable end to end --
    # assert the boundary explicitly in both directions, so a backbone that
    # silently becomes frozen (or unfrozen) fails here.
    for cam_idx, img in enumerate(obs[ObservationKey.images]):
        assert (img.grad is not None) == spec.pixels_get_grad, (
            f"camera {cam_idx}: expected pixels_get_grad={spec.pixels_get_grad}, "
            f"got grad={'set' if img.grad is not None else 'None'}"
        )

    # LanguageEncoder always freezes CLIPTextModel and runs it under
    # torch.no_grad(), so there is no gradient path back to language input
    # (which is raw text, not a tensor, to begin with).
