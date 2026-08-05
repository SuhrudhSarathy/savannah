import pytest
import torch

from savannah.models.backbones.clip_backbone import CLIPBackbone
from savannah.models.backbones.token_learner import TokenLearner

B = 2
IMAGE_SIZE = 224
BASE_MODEL = "openai/clip-vit-base-patch32"
REDUCED_TOKENS = 4


@pytest.fixture
def backbone():
    torch.manual_seed(0)
    return CLIPBackbone(
        image_size=IMAGE_SIZE,
        base_model=BASE_MODEL,
        trainable=False,
    )


def test_output_shape(backbone):
    x = torch.randn(B, 3, IMAGE_SIZE, IMAGE_SIZE)
    out = backbone(x)
    assert out.shape == (B, 1, backbone.out_channels)


def test_tokens_per_image_eos_only(backbone):
    assert backbone.tokens_per_image == 1


def test_tokens_per_image_full_sequence():
    # patch32 model: 224 / 32 = 7 patches per side -> 49 patches + 1 CLS token
    backbone = CLIPBackbone(
        image_size=IMAGE_SIZE,
        base_model=BASE_MODEL,
        use_eos_only=False,
    )
    x = torch.randn(B, 3, IMAGE_SIZE, IMAGE_SIZE)
    out = backbone(x)
    assert backbone.tokens_per_image == 50
    assert out.shape == (B, 50, backbone.out_channels)


def test_plugs_into_vision_encoder(backbone):
    from savannah.models.encoders.vision_encoder import VisionEncoder

    embed_dim = 16
    encoder = VisionEncoder(backbone, embed_dim=embed_dim, num_cameras=1, num_obs=1)
    out = encoder([torch.randn(B, 1, 3, IMAGE_SIZE, IMAGE_SIZE)])
    assert out.shape == (B, backbone.tokens_per_image, embed_dim)


def test_clip_model_params_are_frozen(backbone):
    for p in backbone.model.parameters():
        assert not p.requires_grad


def test_eval_mode_deterministic(backbone):
    backbone.eval()
    x = torch.randn(B, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        out1 = backbone(x)
        out2 = backbone(x)
    assert torch.allclose(out1, out2)


def test_different_input_resolutions(backbone):
    # The processor resizes any input to the model's configured image_size,
    # so arbitrary input resolutions should still produce a fixed-size output.
    backbone.eval()
    for res in [96, 160, 224]:
        x = torch.randn(B, 3, res, res)
        with torch.no_grad():
            out = backbone(x)
        assert out.shape == (B, 1, backbone.out_channels), f"Failed at resolution {res}"


def test_reduce_builds_token_learner():
    backbone = CLIPBackbone(
        image_size=IMAGE_SIZE,
        base_model=BASE_MODEL,
        use_eos_only=False,
        reduce=True,
        reduced_tokens=REDUCED_TOKENS,
    )
    assert isinstance(backbone.token_learner, TokenLearner)
    assert backbone.tokens_per_image == REDUCED_TOKENS


def test_reduce_output_shape():
    backbone = CLIPBackbone(
        image_size=IMAGE_SIZE,
        base_model=BASE_MODEL,
        trainable=False,
        use_eos_only=False,
        reduce=True,
        reduced_tokens=REDUCED_TOKENS,
    )
    x = torch.randn(B, 3, IMAGE_SIZE, IMAGE_SIZE)
    out = backbone(x)
    assert out.shape == (B, REDUCED_TOKENS, backbone.out_channels)


def test_reduce_has_no_effect_when_eos_only():
    # token_learner is only built/used in the non-eos branch.
    backbone = CLIPBackbone(
        image_size=IMAGE_SIZE,
        base_model=BASE_MODEL,
        trainable=False,
        use_eos_only=True,
        reduce=True,
        reduced_tokens=REDUCED_TOKENS,
    )
    assert not hasattr(backbone, "token_learner")
    assert backbone.tokens_per_image == 1

    x = torch.randn(B, 3, IMAGE_SIZE, IMAGE_SIZE)
    out = backbone(x)
    assert out.shape == (B, 1, backbone.out_channels)


def test_reduce_gradient_flows():
    backbone = CLIPBackbone(
        image_size=IMAGE_SIZE,
        base_model=BASE_MODEL,
        trainable=False,
        use_eos_only=False,
        reduce=True,
        reduced_tokens=REDUCED_TOKENS,
    )
    x = torch.randn(B, 3, IMAGE_SIZE, IMAGE_SIZE)
    out = backbone(x)
    loss = out.pow(2).mean()
    loss.backward()

    for p in backbone.token_learner.parameters():
        assert p.grad is not None
        assert p.grad.abs().sum() > 0

    # The frozen CLIP encoder is run under torch.no_grad(), so none of its
    # parameters should have accumulated gradients.
    for p in backbone.model.parameters():
        assert p.grad is None
