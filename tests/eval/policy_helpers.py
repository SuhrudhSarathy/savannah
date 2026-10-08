"""Shared helper for the sim-eval smoke tests in this directory.

Not a test module itself (no test_ prefix) — imported directly by the
sibling test_sim_eval_*.py files.
"""

import torch

from savannah.models.backbones import DummyVisionBackbone
from savannah.models.dit_cross_attn import DITCrossAttnPolicy
from savannah.models.encoders import VisionEncoder
from savannah.objectives import FlowMatchingObjective


def build_tiny_policy(
    state_dim: int,
    action_dim: int,
    action_horizon: int,
    num_cameras: int,
    device: torch.device,
) -> DITCrossAttnPolicy:
    """A tiny, untrained DITCrossAttnPolicy sized just enough to exercise
    evaluate_and_log's rollout loop (the sim-eval logic used by the trainer)
    for a given task's obs/action shapes, without the cost of a real backbone
    or a real trained checkpoint.
    """
    embed_dim = 16
    vision_encoder = VisionEncoder(
        DummyVisionBackbone(out_channels=embed_dim),
        embed_dim=embed_dim,
        num_cameras=num_cameras,
        num_obs=1,
    )

    policy = DITCrossAttnPolicy(
        embed_dim=embed_dim,
        time_embed_dim=8,
        state_embed_dim=8,
        decoder_num_blocks=1,
        decoder_num_attn_heads=2,
        decoder_feedforward_dim=32,
        state_dim=state_dim,
        num_obs=1,
        action_dim=action_dim,
        action_horizon=action_horizon,
        num_cameras=num_cameras,
        use_rope=True,
        vision_encoder=vision_encoder,
        language_encoder=None,
        objective=FlowMatchingObjective(inference_steps=2),
    ).to(device)
    policy.eval()
    return policy
