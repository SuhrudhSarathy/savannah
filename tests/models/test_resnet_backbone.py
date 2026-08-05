import pytest
import torch
import torch.nn as nn

from savannah.models.backbones.resnet_backbone import ResNetBackbone, SpatialSoftmax

B, H, W = 2, 64, 64
OUT_CHANNELS = 32


@pytest.fixture
def backbone():
    torch.manual_seed(0)
    return ResNetBackbone(out_channels=OUT_CHANNELS)


def test_output_shape(backbone):
    x = torch.randn(B, 3, H, W)
    out = backbone(x)
    assert out.shape == (B, OUT_CHANNELS)
    assert backbone.out_channels == OUT_CHANNELS


def test_plugs_into_vision_encoder(backbone):
    from savannah.models.encoders.vision_encoder import VisionEncoder

    embed_dim = 16
    encoder = VisionEncoder(backbone, embed_dim=embed_dim, num_cameras=1, num_obs=1)
    out = encoder([torch.randn(B, 1, 3, H, W)])
    assert out.shape[-1] == embed_dim


def test_no_batchnorm_in_model(backbone):
    # BN is replaced with GroupNorm at construction time
    bn_modules = [m for m in backbone.modules() if isinstance(m, nn.BatchNorm2d)]
    assert bn_modules == []


def test_groupnorm_present(backbone):
    gn_modules = [m for m in backbone.modules() if isinstance(m, nn.GroupNorm)]
    assert len(gn_modules) > 0


def test_all_params_have_gradients(backbone):
    backbone.train()
    x = torch.randn(B, 3, H, W)
    out = backbone(x)
    loss = out.pow(2).mean()
    loss.backward()

    for name, p in backbone.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"
            assert p.grad.abs().sum() > 0 or "bias" in name


def test_repoj_gradient_flows(backbone):
    backbone.train()
    x = torch.randn(B, 3, H, W)
    backbone(x).pow(2).mean().backward()
    assert backbone.repoj.weight.grad is not None
    assert backbone.repoj.weight.grad.abs().sum() > 0


def test_spatial_softmax_output_shape():
    ssm = SpatialSoftmax()
    C = 16
    x = torch.randn(B, C, 8, 8)
    out = ssm(x)
    assert out.shape == (B, 2 * C)


def test_spatial_softmax_output_bounded():
    # Expected coordinates are a weighted mean of linspace(-1, 1), so must stay in [-1, 1]
    ssm = SpatialSoftmax()
    x = torch.randn(B, 8, 8, 8)
    out = ssm(x)
    assert out.min() >= -1.0 - 1e-5
    assert out.max() <= 1.0 + 1e-5


def test_eval_mode_deterministic(backbone):
    backbone.eval()
    x = torch.randn(B, 3, H, W)
    with torch.no_grad():
        out1 = backbone(x)
        out2 = backbone(x)
    assert torch.allclose(out1, out2)


def test_different_input_resolutions(backbone):
    backbone.eval()
    for res in [64, 96, 128]:
        x = torch.randn(B, 3, res, res)
        with torch.no_grad():
            out = backbone(x)
        assert out.shape == (B, OUT_CHANNELS), f"Failed at resolution {res}"


def test_backbone_is_resnet18(backbone):
    # Feature extractor should be a 6-stage Sequential (conv1..layer4)
    # resnet18 children minus avgpool and fc = 8 children
    assert len(list(backbone.backbone.children())) == 8


def test_full_grid_when_spatial_softmax_disabled():
    # 64x64 input -> ResNet18 downsamples by 32 -> 2x2 = 4 tokens
    backbone = ResNetBackbone(
        out_channels=OUT_CHANNELS, image_size=H, use_spatial_softmax=False
    )
    x = torch.randn(B, 3, H, W)
    out = backbone(x)
    assert backbone.tokens_per_image == 4
    assert out.shape == (B, 4, OUT_CHANNELS)


def test_full_grid_gradient_flows():
    backbone = ResNetBackbone(
        out_channels=OUT_CHANNELS, image_size=H, use_spatial_softmax=False
    )
    backbone.train()
    x = torch.randn(B, 3, H, W)
    backbone(x).pow(2).mean().backward()
    assert backbone.repoj.weight.grad is not None
    assert backbone.repoj.weight.grad.abs().sum() > 0
