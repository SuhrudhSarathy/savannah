import pytest
import torch
import torch.nn as nn

from savannah.models.backbones.clip_backbone import CLIPBackbone

B = 2
OUT_CHANNELS = 32
IMAGE_SIZE = 224
BASE_MODEL = "openai/clip-vit-base-patch32"


@pytest.fixture
def backbone():
    torch.manual_seed(0)
    return CLIPBackbone(
        out_channels=OUT_CHANNELS,
        image_size=IMAGE_SIZE,
        base_model=BASE_MODEL,
        trainable=False,
    )


def test_output_shape(backbone):
    x = torch.randn(B, 3, IMAGE_SIZE, IMAGE_SIZE)
    out = backbone(x)
    assert out.shape == (B, 1, OUT_CHANNELS)
    assert backbone.out_channels == OUT_CHANNELS


def test_tokens_per_image_eos_only(backbone):
    assert backbone.tokens_per_image == 1


def test_tokens_per_image_full_sequence():
    # patch32 model: 224 / 32 = 7 patches per side -> 49 patches + 1 CLS token
    backbone = CLIPBackbone(
        out_channels=OUT_CHANNELS,
        image_size=IMAGE_SIZE,
        base_model=BASE_MODEL,
        use_eos_only=False,
    )
    x = torch.randn(B, 3, IMAGE_SIZE, IMAGE_SIZE)
    out = backbone(x)
    assert backbone.tokens_per_image == 50
    assert out.shape == (B, 50, OUT_CHANNELS)


def test_plugs_into_vision_encoder(backbone):
    from savannah.models.encoders.vision_encoder import VisionEncoder

    embed_dim = 16
    encoder = VisionEncoder(backbone, embed_dim=embed_dim)
    out = encoder(torch.randn(B, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert out.shape == (B, backbone.tokens_per_image, embed_dim)


def test_clip_model_params_are_frozen(backbone):
    for p in backbone.model.parameters():
        assert not p.requires_grad


def test_projection_is_linear_when_dims_differ(backbone):
    assert isinstance(backbone.projection, nn.Linear)
    assert backbone.projection.in_features == backbone.clip_dim
    assert backbone.projection.out_features == OUT_CHANNELS


def test_projection_is_identity_when_dims_match():
    backbone = CLIPBackbone(
        out_channels=768, image_size=IMAGE_SIZE, base_model=BASE_MODEL
    )
    assert isinstance(backbone.projection, nn.Identity)


def test_projection_gradient_flows(backbone):
    backbone.train()
    x = torch.randn(B, 3, IMAGE_SIZE, IMAGE_SIZE)
    out = backbone(x)
    loss = out.pow(2).mean()
    loss.backward()

    assert backbone.projection.weight.grad is not None
    assert backbone.projection.weight.grad.abs().sum() > 0

    # The frozen CLIP encoder is run under torch.no_grad(), so none of its
    # parameters should have accumulated gradients.
    for p in backbone.model.parameters():
        assert p.grad is None


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
        assert out.shape == (B, 1, OUT_CHANNELS), f"Failed at resolution {res}"
