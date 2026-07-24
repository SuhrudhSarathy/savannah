import pytest
import torch
import torch.nn as nn

from savannah.models.backbones.lora_resnet_backbone import (
    LoRAConv2d,
    LoRAResNetBackbone,
)

B, H, W = 2, 64, 64
OUT_CHANNELS = 32


@pytest.fixture
def backbone():
    torch.manual_seed(0)
    return LoRAResNetBackbone(out_channels=OUT_CHANNELS, image_size=H)


def _lora_modules(backbone):
    return [m for m in backbone.backbone.modules() if isinstance(m, LoRAConv2d)]


def test_output_shape(backbone):
    x = torch.randn(B, 3, H, W)
    out = backbone(x)
    assert out.shape[-1] == OUT_CHANNELS
    assert backbone.out_channels == OUT_CHANNELS


def test_output_shape_with_know_output(backbone):
    x = torch.randn(B, 3, 96, 96)
    out = backbone(x)
    assert out.shape == (B, 9, OUT_CHANNELS)
    assert backbone.out_channels == OUT_CHANNELS


def test_tokens_per_image(backbone):
    # 64×64 input → ResNet18 downsamples by 32 → 2×2 = 4 tokens
    assert backbone.tokens_per_image == (H // 32) ** 2


def test_plugs_into_vision_encoder(backbone):
    from savannah.models.encoders.vision_encoder import VisionEncoder

    embed_dim = 16
    encoder = VisionEncoder(backbone, embed_dim=embed_dim)
    out = encoder(torch.randn(B, 3, H, W))
    assert out.shape == (B, backbone.tokens_per_image, embed_dim)


def test_lora_wrapped_convs_have_original_frozen(backbone):
    lora_modules = _lora_modules(backbone)
    assert len(lora_modules) > 0
    for m in lora_modules:
        assert not m.original_conv.weight.requires_grad
        assert m.lora_A.weight.requires_grad
        assert m.lora_B.weight.requires_grad


def test_lora_B_gradient_flows_on_first_step(backbone):
    backbone.train()
    x = torch.randn(B, 3, H, W)
    out = backbone(x)
    loss = out.pow(2).mean()
    loss.backward()

    for m in _lora_modules(backbone):
        assert m.original_conv.weight.grad is None
        assert m.lora_B.weight.grad is not None
        assert m.lora_B.weight.grad.abs().sum() > 0
        # lora_B is zero-initialized, so on the very first backward pass the
        # gradient reaching lora_A through it is mathematically zero -- this
        # is expected LoRA zero-init behaviour, not a broken graph.
        assert m.lora_A.weight.grad is not None
        assert m.lora_A.weight.grad.abs().sum() == 0

    assert backbone.channel_proj.weight.grad is not None
    assert backbone.channel_proj.weight.grad.abs().sum() > 0


def test_lora_A_gradient_flows_after_B_updates(backbone):
    backbone.train()
    opt = torch.optim.SGD([p for p in backbone.parameters() if p.requires_grad], lr=0.1)
    x = torch.randn(B, 3, H, W)

    backbone(x).pow(2).mean().backward()
    opt.step()
    opt.zero_grad()

    backbone(x).pow(2).mean().backward()
    for m in _lora_modules(backbone):
        assert m.lora_A.weight.grad.abs().sum() > 0


def _bare_convs(module, prefix=""):
    # Walks the tree without descending into LoRAConv2d internals
    # (original_conv/lora_A/lora_B), so it only reports genuinely unwrapped
    # Conv2d layers, not the frozen conv living inside a LoRAConv2d wrapper.
    found = []
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, LoRAConv2d):
            continue
        elif isinstance(child, nn.Conv2d):
            found.append(full)
        else:
            found.extend(_bare_convs(child, full))
    return found


def test_all_convs_including_downsample_shortcuts_are_lora_wrapped(backbone):
    # `add_lora` wraps every Conv2d, including the 1x1 downsample shortcuts
    # in layer2/3/4, so the whole backbone is frozen and only LoRA adapters
    # (plus channel_proj) are trainable.
    assert _bare_convs(backbone.backbone) == []

    downsample_convs = [
        m
        for name, m in backbone.backbone.named_modules()
        if isinstance(m, LoRAConv2d) and name.endswith("downsample.0")
    ]
    assert len(downsample_convs) == 3
    for m in downsample_convs:
        assert not m.original_conv.weight.requires_grad
        assert m.lora_A.weight.requires_grad
        assert m.lora_B.weight.requires_grad


def test_pooled_when_spatial_softmax_enabled():
    backbone = LoRAResNetBackbone(
        out_channels=OUT_CHANNELS, image_size=H, use_spatial_softmax=True
    )
    x = torch.randn(B, 3, H, W)
    out = backbone(x)
    assert backbone.tokens_per_image == 1
    assert out.shape == (B, OUT_CHANNELS)


def test_pooled_gradient_flows_into_channel_proj():
    backbone = LoRAResNetBackbone(
        out_channels=OUT_CHANNELS, image_size=H, use_spatial_softmax=True
    )
    backbone.train()
    x = torch.randn(B, 3, H, W)
    backbone(x).pow(2).mean().backward()
    assert backbone.channel_proj.weight.grad is not None
    assert backbone.channel_proj.weight.grad.abs().sum() > 0


def test_merge_unmerge_round_trip(backbone):
    backbone.eval()
    # lora_B is zero-initialized, so give it nonzero weights first --
    # otherwise merge/unmerge would trivially be a no-op.
    for m in _lora_modules(backbone):
        with torch.no_grad():
            m.lora_B.weight.add_(torch.randn_like(m.lora_B.weight) * 0.1)

    x = torch.randn(B, 3, H, W)
    with torch.no_grad():
        out_before = backbone(x)

    backbone.merge_lora()
    with torch.no_grad():
        out_merged = backbone(x)
    assert torch.allclose(out_before, out_merged, atol=1e-3)

    backbone.unmerge_lora()
    with torch.no_grad():
        out_unmerged = backbone(x)
    assert torch.allclose(out_before, out_unmerged, atol=1e-3)
