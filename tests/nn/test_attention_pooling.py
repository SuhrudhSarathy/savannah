import pytest
import torch

from savannah.models.backbones import DummyVisionBackbone
from savannah.nn.attention_pooling import AttentionPooling

B, H, W = 2, 32, 32
EMBED_DIM, NUM_QUERIES, NUM_HEADS = 16, 3, 4
NUM_TOKENS = 9


@pytest.fixture
def pooling():
    torch.manual_seed(0)
    return AttentionPooling(
        embed_dim=EMBED_DIM, num_queries=NUM_QUERIES, num_attn_heads=NUM_HEADS, depth=2
    )


def test_output_shape(pooling):
    x = torch.randn(B, NUM_TOKENS, EMBED_DIM)
    assert pooling(x).shape == (B, NUM_QUERIES, EMBED_DIM)


@pytest.mark.parametrize("num_tokens", [1, 4, 49])
def test_output_shape_independent_of_input_tokens(pooling, num_tokens):
    x = torch.randn(B, num_tokens, EMBED_DIM)
    assert pooling(x).shape == (B, NUM_QUERIES, EMBED_DIM)


def test_output_depends_on_input(pooling):
    pooling.eval()
    out1 = pooling(torch.randn(B, NUM_TOKENS, EMBED_DIM))
    out2 = pooling(torch.randn(B, NUM_TOKENS, EMBED_DIM))
    assert not torch.allclose(out1, out2)


def test_batch_elements_pooled_independently(pooling):
    pooling.eval()
    x = torch.randn(B, NUM_TOKENS, EMBED_DIM)
    out = pooling(x)
    out_first = pooling(x[:1])
    assert torch.allclose(out[:1], out_first, atol=1e-5)


def test_backward_pass_populates_gradients(pooling):
    pooling.train()
    x = torch.randn(B, NUM_TOKENS, EMBED_DIM, requires_grad=True)
    pooling(x).sum().backward()

    missing_grad = [
        name
        for name, p in pooling.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not missing_grad, f"parameters with no gradient: {missing_grad}"
    assert x.grad is not None


def test_pools_dummy_backbone_tokens_end_to_end():
    # The real use: pooling a spatial backbone's token grid down to a fixed
    # number of tokens, with gradient reaching the backbone.
    backbone = DummyVisionBackbone(out_channels=EMBED_DIM, tokens_per_image=NUM_TOKENS)
    pooling = AttentionPooling(
        embed_dim=backbone.out_channels,
        num_queries=NUM_QUERIES,
        num_attn_heads=NUM_HEADS,
    )
    tokens = backbone(torch.randn(B, 3, H, W))
    assert tokens.shape == (B, NUM_TOKENS, EMBED_DIM)

    out = pooling(tokens)
    assert out.shape == (B, NUM_QUERIES, EMBED_DIM)

    out.sum().backward()
    assert backbone.net.weight.grad is not None
