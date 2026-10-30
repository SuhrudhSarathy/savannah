from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler

from savannah.utils.log import logger
from savannah.utils.policy_output import PolicyOutput


class PolicyObjective(ABC):
    """
    Defines the training objective and inference sampler for a
    DenoisingPolicy. The policy encodes the observation once
    (cond = model.encode_obs(obs)) and passes `cond` in; objectives only call
    model.denoise(x_t, t, cond). They keep no per-call state, so one objective
    can be shared between policies and calls.

    compute_loss(model, cond, actions, t=None)
        Loss on ground-truth `actions` (B, action_horizon, action_dim) at
        timestep `t` (B,), sampled if not given.

    compute_action(model, cond, *, return_trace=False)
        Fully denoise an action chunk from noise, without gradients. With
        return_trace, also return the sampling trajectory in PolicyOutput.aux.
    """

    @abstractmethod
    def compute_loss(
        self,
        model,
        cond: torch.Tensor,
        actions: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    @abstractmethod
    def compute_action(
        self,
        model,
        cond: torch.Tensor,
        *,
        return_trace: bool = False,
    ) -> PolicyOutput: ...


def _initial_noise(model, cond: torch.Tensor) -> torch.Tensor:
    return torch.randn(
        (cond.shape[0], model.action_horizon, model.action_dim), device=cond.device
    )


class OverfitObjective(PolicyObjective):
    """Debug objective: the network learns the identity on actions, at t=0
    unless a timestep is given."""

    def __init__(self):
        super().__init__()

    def compute_loss(
        self,
        model,
        cond: torch.Tensor,
        actions: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if t is None:
            t = torch.zeros(actions.shape[0], device=actions.device)
        out = model.denoise(actions, t, cond)
        loss = F.mse_loss(out, actions)
        return loss

    def compute_action(
        self,
        model,
        cond: torch.Tensor,
        *,
        return_trace: bool = False,
    ) -> PolicyOutput:
        with torch.no_grad():
            x = _initial_noise(model, cond)
            t = torch.zeros(x.shape[0], device=x.device)
            out = model.denoise(x, t, cond)

        return PolicyOutput(actions=out)


class FlowMatchingObjective(PolicyObjective):
    def __init__(self, inference_steps: int = 10, integration_type: str = "euler"):
        self.inference_steps = inference_steps
        self.integration_type = integration_type

        assert self.integration_type in ["euler", "heun"], (
            f"Only available integration type is euler or huen, configured is {integration_type}"
        )

    def compute_loss(
        self,
        model,
        cond: torch.Tensor,
        actions: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """t (B,) in [0, 1]: 0 is pure noise, 1 is the data."""
        B, device = actions.shape[0], actions.device

        x_1 = actions
        x_0 = torch.randn_like(x_1)
        if t is None:
            t = torch.rand((B,), device=device)
        x_t = (1 - t.view(B, 1, 1)) * x_0 + t.view(B, 1, 1) * x_1
        target = x_1 - x_0

        pred_vel = model.denoise(x_t, t, cond)
        loss = F.mse_loss(pred_vel, target)
        return loss

    def compute_action(
        self,
        model,
        cond: torch.Tensor,
        *,
        return_trace: bool = False,
    ) -> PolicyOutput:
        with torch.no_grad():
            x_0 = _initial_noise(model, cond)
            B, device = x_0.shape[0], x_0.device

            x_t = x_0
            dt = 1.0 / self.inference_steps
            trace = [x_t]

            for i in range(self.inference_steps):
                t_cur = i * dt
                t_next = (i + 1) * dt

                if self.integration_type == "euler":
                    t = torch.full((B,), t_cur, device=device)
                    x_t = x_t + model.denoise(x_t, t, cond) * dt
                elif self.integration_type == "heun":
                    t = torch.full((B,), t_cur, device=device)
                    v1 = model.denoise(x_t, t, cond)

                    x_pred = x_t + v1 * dt  # Euler predictor

                    t = torch.full((B,), t_next, device=device)
                    v2 = model.denoise(x_pred, t, cond)

                    x_t = x_t + 0.5 * (v1 + v2) * dt

                trace.append(x_t)

            aux = None
            if return_trace:
                aux = {
                    # (B, inference_steps + 1, action_horizon, action_dim)
                    "trace": torch.stack(trace, dim=1),
                    "timesteps": torch.linspace(
                        0.0, 1.0, self.inference_steps + 1, device=device
                    ),
                }

            return PolicyOutput(actions=x_t, aux=aux)


class DDIMObjective(PolicyObjective):
    def __init__(
        self,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 10,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        clip_sample: bool = True,
    ):
        self.scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
            clip_sample=clip_sample,
        )
        self.scheduler.set_timesteps(num_inference_steps)

    def _normalized_time(self, timesteps: torch.Tensor) -> torch.Tensor:
        return timesteps.float() / self.scheduler.config.num_train_timesteps

    def compute_loss(
        self,
        model,
        cond: torch.Tensor,
        actions: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """t (B,) integer diffusion timesteps in [0, num_train_timesteps)."""
        B, device = actions.shape[0], actions.device

        noise = torch.randn_like(actions)
        if t is None:
            t = torch.randint(
                0, self.scheduler.config.num_train_timesteps, (B,), device=device
            )
        t = t.long()
        x_t = self.scheduler.add_noise(actions, noise, t)

        target = noise

        pred = model.denoise(x_t, self._normalized_time(t), cond)
        loss = F.mse_loss(pred, target)
        return loss

    def compute_action(
        self,
        model,
        cond: torch.Tensor,
        *,
        return_trace: bool = False,
    ) -> PolicyOutput:
        with torch.no_grad():
            x_T = _initial_noise(model, cond)
            B, device = x_T.shape[0], x_T.device

            x_t = x_T
            trace = [x_t]

            for t in self.scheduler.timesteps:
                t_model = torch.full(
                    (B,), float(self._normalized_time(t)), device=device
                )
                model_output = model.denoise(x_t, t_model, cond)
                logger.debug("DDIM step {}: model_output={}", t.item(), model_output)

                x_t = self.scheduler.step(model_output, t, x_t).prev_sample
                logger.debug("DDIM step {}: x_t={}", t.item(), x_t)
                trace.append(x_t)

            aux = None
            if return_trace:
                aux = {
                    # (B, num_inference_steps + 1, action_horizon, action_dim)
                    "trace": torch.stack(trace, dim=1),
                    "timesteps": self.scheduler.timesteps.to(device),
                }

            return PolicyOutput(actions=x_t, aux=aux)
