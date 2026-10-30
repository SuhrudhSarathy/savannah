"""Tests for the encode_obs/denoise policy split and the stateless samplers in
savannah.objectives, run against every DenoisingPolicy."""

import pytest
import torch
import torch.nn.functional as F

from savannah.models.backbones import DummyVisionBackbone
from savannah.models.dit_cross_attn import DITCrossAttnPolicy
from savannah.models.encoders import VisionEncoder
from savannah.models.lbm import LBMPolicy
from savannah.models.unet import UNetPolicy
from savannah.objectives import (
    DDIMObjective,
    FlowMatchingObjective,
    OverfitObjective,
)
from savannah.utils.observation import ObservationKey

B, H, W = 2, 32, 32
STATE_DIM, ACTION_DIM, ACTION_HORIZON, EMBED_DIM = 5, 4, 8, 32
NUM_CAMERAS, NUM_OBS = 2, 1


def _vision_encoder():
    return VisionEncoder(
        DummyVisionBackbone(out_channels=16),
        embed_dim=EMBED_DIM,
        num_cameras=NUM_CAMERAS,
        num_obs=NUM_OBS,
    )


def _transformer_kwargs():
    return dict(
        embed_dim=EMBED_DIM,
        time_embed_dim=16,
        state_embed_dim=16,
        decoder_num_blocks=1,
        decoder_num_attn_heads=2,
        decoder_feedforward_dim=64,
        state_dim=STATE_DIM,
        num_obs=NUM_OBS,
        action_dim=ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        num_cameras=NUM_CAMERAS,
        use_rope=True,
        vision_encoder=_vision_encoder(),
        language_encoder=None,
    )


def _build(name, objective):
    torch.manual_seed(0)
    if name == "unet":
        policy = UNetPolicy(
            embed_dim=EMBED_DIM,
            unet_channels=[16, 32],
            num_residual_layers=1,
            num_obs=NUM_OBS,
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            action_horizon=ACTION_HORIZON,
            num_cameras=NUM_CAMERAS,
            vision_encoder=_vision_encoder(),
            objective=objective,
        )
    elif name == "lbm":
        policy = LBMPolicy(**_transformer_kwargs(), objective=objective)
    else:
        policy = DITCrossAttnPolicy(**_transformer_kwargs(), objective=objective)
    # The transformer blocks are zero-initialised (identity at init, so time
    # and conditioning would be ignored); randomise so every path does work.
    with torch.no_grad():
        for p in policy.parameters():
            p.normal_(0.0, 0.2)
    return policy.eval()


POLICIES = ["unet", "lbm", "dit"]
OBJECTIVES = {
    "flow_euler": lambda: FlowMatchingObjective(inference_steps=4),
    "flow_heun": lambda: FlowMatchingObjective(
        inference_steps=4, integration_type="heun"
    ),
    "ddim": lambda: DDIMObjective(num_train_timesteps=20, num_inference_steps=4),
}


@pytest.fixture(params=POLICIES)
def policy_name(request):
    return request.param


@pytest.fixture(params=list(OBJECTIVES))
def objective_name(request):
    return request.param


@pytest.fixture
def policy(policy_name, objective_name):
    return _build(policy_name, OBJECTIVES[objective_name]())


def make_obs():
    g = torch.Generator().manual_seed(1)
    return {
        ObservationKey.images: [
            torch.rand(B, NUM_OBS, 3, H, W, generator=g) for _ in range(NUM_CAMERAS)
        ],
        ObservationKey.state: torch.randn(B, NUM_OBS, STATE_DIM, generator=g),
        ObservationKey.actions: torch.randn(B, ACTION_HORIZON, ACTION_DIM, generator=g),
        ObservationKey.gt_actions: torch.randn(
            B, ACTION_HORIZON, ACTION_DIM, generator=g
        ),
    }


def _forbid_encode(policy, monkeypatch):
    def fail(obs):
        raise AssertionError("objective called encode_obs")

    monkeypatch.setattr(policy, "encode_obs", fail)


def _random_t(objective):
    if isinstance(objective, DDIMObjective):
        return torch.randint(0, objective.scheduler.config.num_train_timesteps, (B,))
    return torch.rand(B)


# ---------------------------- encode_obs / denoise ---------------------------


def test_forward_matches_encode_then_denoise(policy):
    obs = make_obs()
    x_t = torch.randn(B, ACTION_HORIZON, ACTION_DIM)
    t = torch.rand(B)
    with torch.no_grad():
        via_forward = policy.forward(obs, noisy_actions=x_t, t=t).actions
        via_split = policy.denoise(x_t, t, policy.encode_obs(obs))
    torch.testing.assert_close(via_forward, via_split)


def test_forward_requires_noisy_actions_and_t(policy):
    obs = make_obs()
    x_t = torch.randn(B, ACTION_HORIZON, ACTION_DIM)
    with pytest.raises(AssertionError):
        policy.forward(obs, t=torch.rand(B))
    with pytest.raises(AssertionError):
        policy.forward(obs, noisy_actions=x_t)


def test_encode_obs_returns_batched_tensor(policy):
    cond = policy.encode_obs(make_obs())
    assert isinstance(cond, torch.Tensor)
    assert cond.shape[0] == B


def test_denoise_depends_on_time(policy):
    obs = make_obs()
    x_t = torch.randn(B, ACTION_HORIZON, ACTION_DIM)
    with torch.no_grad():
        cond = policy.encode_obs(obs)
        a = policy.denoise(x_t, torch.zeros(B), cond)
        b = policy.denoise(x_t, torch.full((B,), 0.9), cond)
    assert not torch.allclose(a, b)


def test_cond_can_be_reused_across_noise_samples(policy):
    obs = make_obs()
    with torch.no_grad():
        cond = policy.encode_obs(obs)
        t = torch.rand(B)
        for x in torch.randn(2, B, ACTION_HORIZON, ACTION_DIM):
            torch.testing.assert_close(
                policy.denoise(x, t, cond),
                policy.forward(obs, noisy_actions=x, t=t).actions,
            )


# ------------------------- objectives work on cond only ----------------------


def test_objective_compute_action_never_encodes(policy, monkeypatch):
    obs = make_obs()
    with torch.no_grad():
        cond = policy.encode_obs(obs)
    torch.manual_seed(7)
    expected = policy.compute_action(obs).actions

    _forbid_encode(policy, monkeypatch)
    torch.manual_seed(7)
    out = policy.objective.compute_action(policy, cond).actions
    torch.testing.assert_close(out, expected)


def test_objective_compute_loss_never_encodes(policy, monkeypatch):
    obs = make_obs()
    cond = policy.encode_obs(obs)
    _forbid_encode(policy, monkeypatch)
    loss = policy.objective.compute_loss(policy, cond, obs[ObservationKey.gt_actions])
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_policy_encodes_once_per_call(policy, monkeypatch):
    calls = {"encode": 0, "vision": 0}
    encode_obs = policy.encode_obs
    vision_forward = policy.vision_encoder.forward

    def counting_encode(obs):
        calls["encode"] += 1
        return encode_obs(obs)

    def counting_vision(*args, **kwargs):
        calls["vision"] += 1
        return vision_forward(*args, **kwargs)

    monkeypatch.setattr(policy, "encode_obs", counting_encode)
    monkeypatch.setattr(policy.vision_encoder, "forward", counting_vision)

    policy.compute_action(make_obs())
    assert calls == {"encode": 1, "vision": 1}

    policy.compute_loss(make_obs())
    assert calls == {"encode": 2, "vision": 2}


def test_obs_dict_is_not_mutated(policy):
    obs = make_obs()
    keys = set(obs)
    policy.compute_action(obs)
    policy.compute_loss(obs)
    assert set(obs) == keys


# -------------------------------- compute_loss -------------------------------


def test_loss_at_given_timestep(policy):
    obs = make_obs()
    cond = policy.encode_obs(obs)
    actions = obs[ObservationKey.gt_actions]
    t = _random_t(policy.objective)

    torch.manual_seed(0)
    a = policy.objective.compute_loss(policy, cond, actions, t)
    torch.manual_seed(0)
    b = policy.objective.compute_loss(policy, cond, actions, t)
    torch.manual_seed(0)
    c = policy.objective.compute_loss(
        policy, cond, actions, 1 - t if t.is_floating_point() else t // 2
    )
    torch.testing.assert_close(a, b)
    assert not torch.allclose(a, c)


def test_loss_backpropagates_to_encoders(policy):
    policy.train()
    policy.zero_grad(set_to_none=True)
    policy.compute_loss(make_obs()).backward()
    assert any(p.grad is not None for p in policy.vision_encoder.parameters())
    assert any(p.grad is not None for p in policy.state_encoder.parameters())


# -------------------------------- compute_action -----------------------------


def test_return_trace(policy):
    obs = make_obs()
    torch.manual_seed(7)
    noise = torch.randn(B, ACTION_HORIZON, ACTION_DIM)
    torch.manual_seed(7)
    out = policy.compute_action(obs, return_trace=True)
    steps = (
        policy.objective.scheduler.num_inference_steps
        if isinstance(policy.objective, DDIMObjective)
        else policy.objective.inference_steps
    )
    trace = out.aux["trace"]
    assert trace.shape == (B, steps + 1, ACTION_HORIZON, ACTION_DIM)
    assert len(out.aux["timesteps"]) == steps + (
        0 if isinstance(policy.objective, DDIMObjective) else 1
    )
    torch.testing.assert_close(trace[:, 0], noise)
    torch.testing.assert_close(trace[:, -1], out.actions)


def test_no_trace_by_default(policy):
    assert policy.compute_action(make_obs()).aux is None


def test_sampling_is_no_grad_by_default(policy):
    out = policy.compute_action(make_obs())
    assert not out.actions.requires_grad


# ---------------------------- Overfit objective -------------------------------


def test_overfit_objective(policy_name):
    policy = _build(policy_name, OverfitObjective())
    obs = make_obs()
    actions = obs[ObservationKey.gt_actions]
    with torch.no_grad():
        cond = policy.encode_obs(obs)
        torch.manual_seed(7)
        noise = torch.randn(B, ACTION_HORIZON, ACTION_DIM)
        torch.manual_seed(7)
        torch.testing.assert_close(
            policy.objective.compute_action(policy, cond).actions,
            policy.denoise(noise, torch.zeros(B), cond),
        )

        out = policy.denoise(actions, torch.zeros(B), cond)
        loss_t0 = policy.objective.compute_loss(policy, cond, actions)
        torch.testing.assert_close(loss_t0, F.mse_loss(out, actions))
        loss_t = policy.objective.compute_loss(
            policy, cond, actions, torch.full((B,), 0.7)
        )
    assert not torch.allclose(loss_t0, loss_t)
