import pytest
import torch

from savannah.nn.rope import RoPESelfAttention, RotatoryPositionalEncoding

HEAD_DIM, MAX_SEQ_LEN = 16, 32
B, NH, T = 2, 4, 8

EMBED_DIM, NUM_HEADS = 32, 4
ATTN_B, ATTN_T = 2, 6


@pytest.fixture
def rope():
    return RotatoryPositionalEncoding(HEAD_DIM, MAX_SEQ_LEN)


def test_cache_shapes(rope):
    assert rope.cos_cached.shape == (1, 1, MAX_SEQ_LEN, HEAD_DIM)
    assert rope.sin_cached.shape == (1, 1, MAX_SEQ_LEN, HEAD_DIM)


def test_forward_preserves_shape(rope):
    q = torch.randn(B, NH, T, HEAD_DIM)
    k = torch.randn(B, NH, T, HEAD_DIM)
    q_embed, k_embed = rope(q, k)
    assert q_embed.shape == q.shape
    assert k_embed.shape == k.shape


def test_forward_shorter_than_max_seq_len(rope):
    short_t = 3
    q = torch.randn(B, NH, short_t, HEAD_DIM)
    k = torch.randn(B, NH, short_t, HEAD_DIM)
    q_embed, k_embed = rope(q, k)
    assert q_embed.shape == (B, NH, short_t, HEAD_DIM)
    assert k_embed.shape == (B, NH, short_t, HEAD_DIM)


def test_position_zero_is_identity(rope):
    q = torch.randn(B, NH, T, HEAD_DIM)
    k = torch.randn(B, NH, T, HEAD_DIM)
    q_embed, k_embed = rope(q, k)
    assert torch.allclose(q_embed[..., 0, :], q[..., 0, :], atol=1e-6)
    assert torch.allclose(k_embed[..., 0, :], k[..., 0, :], atol=1e-6)


def test_rotation_preserves_norm(rope):
    q = torch.randn(B, NH, T, HEAD_DIM)
    k = torch.randn(B, NH, T, HEAD_DIM)
    q_embed, k_embed = rope(q, k)
    assert torch.allclose(torch.norm(q_embed, dim=-1), torch.norm(q, dim=-1), atol=1e-5)
    assert torch.allclose(torch.norm(k_embed, dim=-1), torch.norm(k, dim=-1), atol=1e-5)


def test_relative_position_invariance(rope):
    # RoPE's defining property: the rotated dot product q_i . k_j depends
    # only on the offset (i - j), not on the absolute positions.
    q_vec = torch.randn(1, 1, HEAD_DIM)
    k_vec = torch.randn(1, 1, HEAD_DIM)
    q_full = q_vec.expand(1, 1, MAX_SEQ_LEN, HEAD_DIM)
    k_full = k_vec.expand(1, 1, MAX_SEQ_LEN, HEAD_DIM)

    q_embed, k_embed = rope(q_full, k_full)

    def score(i, j):
        return (q_embed[0, 0, i, :] * k_embed[0, 0, j, :]).sum()

    score_a = score(5, 2)  # offset 3
    score_b = score(13, 10)  # offset 3
    assert torch.allclose(score_a, score_b, atol=1e-4)


def test_sin_cos_are_distinct(rope):
    assert not torch.equal(rope.cos_cached, rope.sin_cached)


def test_gradients_flow_to_inputs(rope):
    q = torch.randn(B, NH, T, HEAD_DIM, requires_grad=True)
    k = torch.randn(B, NH, T, HEAD_DIM, requires_grad=True)
    q_embed, k_embed = rope(q, k)
    (q_embed.sum() + k_embed.sum()).backward()
    assert q.grad is not None
    assert k.grad is not None


def make_x():
    return torch.randn(ATTN_B, ATTN_T, EMBED_DIM)


@pytest.mark.parametrize("use_sdpa", [True, False])
def test_attention_output_shape(use_sdpa):
    attn = RoPESelfAttention(EMBED_DIM, NUM_HEADS, use_sdpa=use_sdpa)
    out = attn(make_x())
    assert out.shape == (ATTN_B, ATTN_T, EMBED_DIM)


@pytest.mark.parametrize("use_sdpa", [True, False])
def test_attention_output_has_no_nan_or_inf(use_sdpa):
    attn = RoPESelfAttention(EMBED_DIM, NUM_HEADS, use_sdpa=use_sdpa)
    out = attn(make_x())
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_sdpa_and_manual_paths_agree():
    torch.manual_seed(0)
    attn = RoPESelfAttention(EMBED_DIM, NUM_HEADS, use_sdpa=True)
    attn.eval()
    x = make_x()

    attn.use_sdpa = True
    out_sdpa = attn(x)

    attn.use_sdpa = False
    out_manual = attn(x)

    assert torch.allclose(out_sdpa, out_manual, atol=1e-4)


def test_mask_applied():
    attn = RoPESelfAttention(EMBED_DIM, NUM_HEADS, use_sdpa=False)
    attn.eval()
    x = make_x()

    mask = torch.ones(1, 1, ATTN_T, ATTN_T, dtype=torch.bool)
    mask[..., -1] = False  # every query ignores the last key position

    out_unmasked = attn(x, mask=None)
    out_masked = attn(x, mask=mask)

    assert out_masked.shape == (ATTN_B, ATTN_T, EMBED_DIM)
    assert not torch.allclose(out_masked, out_unmasked, atol=1e-4)


def test_backward_pass_populates_gradients():
    attn = RoPESelfAttention(EMBED_DIM, NUM_HEADS, use_sdpa=True)
    attn.train()
    out = attn(make_x())
    out.sum().backward()

    missing_grad = [
        name
        for name, p in attn.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not missing_grad, f"parameters with no gradient: {missing_grad}"
