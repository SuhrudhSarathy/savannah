import torch

from savannah.models.backbones.token_learner import TokenLearner

B = 2
N_TOKENS = 10
EMBED_DIM = 16
OUTPUT_DIM = 4


def test_output_shape():
    torch.manual_seed(0)
    token_learner = TokenLearner(EMBED_DIM, OUTPUT_DIM, hidden=8)
    x = torch.randn(B, N_TOKENS, EMBED_DIM)
    out = token_learner(x)
    assert out.shape == (B, OUTPUT_DIM, EMBED_DIM)


def test_gradient_flows():
    torch.manual_seed(0)
    token_learner = TokenLearner(EMBED_DIM, OUTPUT_DIM, hidden=8)
    x = torch.randn(B, N_TOKENS, EMBED_DIM, requires_grad=True)
    out = token_learner(x)
    loss = out.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert x.grad.abs().sum() > 0
    for p in token_learner.parameters():
        assert p.grad is not None
        assert p.grad.abs().sum() > 0
