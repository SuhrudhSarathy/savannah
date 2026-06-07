from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler

from savannah.utils.observation import ObservationKey
from savannah.utils.policy import PolicyOutput


class PolicyObjective(ABC):
    """
    Defines the training objective and inference sampler for a policy.
    Methods receive the model as an argument so they can call model.forward()
    and access model.action_horizon / model.action_dim.
    """

    @abstractmethod
    def compute_loss(self, model, obs: dict) -> torch.Tensor: ...

    @abstractmethod
    def compute_action(self, model, obs: dict) -> PolicyOutput: ...


class OverfitObjective(PolicyObjective):
    def __init__(self):
        super().__init__()

    def compute_loss(self, model, obs: dict) -> torch.Tensor:
        """The assumption is that obs is fully defined, no need to manipulate this at all"""
        x_t = obs[ObservationKey.actions]
        out = model.forward(obs, noisy_actions=x_t).actions
        loss = F.mse_loss(out, x_t)
        return loss

    def compute_action(self, model, obs: dict) -> PolicyOutput:
        x_t = obs[ObservationKey.actions]
        out = model.forward(obs, noisy_actions=x_t).actions

        return PolicyOutput(actions=out)


class FlowMatchingObjective(PolicyObjective):
    def __init__(self, inference_steps: int = 10):
        self.inference_steps = inference_steps

    def compute_loss(self, model, obs: dict) -> torch.Tensor:
        B = obs[ObservationKey.state].shape[0]
        device = obs[ObservationKey.state].device

        x_1 = obs[ObservationKey.gt_actions]
        x_0 = torch.randn_like(x_1)
        t = torch.rand((B,), device=device)
        x_t = (1 - t.view(B, 1, 1)) * x_0 + t.view(B, 1, 1) * x_1
        target = x_1 - x_0

        obs[ObservationKey.time] = t
        pred_vel = model.forward(obs, noisy_actions=x_t).actions
        loss = F.mse_loss(pred_vel, target)
        return loss

    def compute_action(self, model, obs: dict) -> PolicyOutput:
        with torch.no_grad():
            B = obs[ObservationKey.state].shape[0]
            device = obs[ObservationKey.state].device

            x_t = torch.randn(
                (B, model.action_horizon, model.action_dim), device=device
            )
            dt = 1.0 / self.inference_steps

            for i in range(self.inference_steps):
                obs[ObservationKey.time] = torch.full((B,), i * dt, device=device)
                x_t = x_t + model.forward(obs, noisy_actions=x_t).actions * dt

            return PolicyOutput(actions=x_t)


class DDIMObjective(PolicyObjective):
    def __init__(
        self,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 10,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        clip_sample: bool = True,
    ):
        self.num_inference_steps = num_inference_steps
        self.scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
            clip_sample=clip_sample,
        )

    def compute_loss(self, model, obs: dict) -> torch.Tensor:
        x_1 = obs[ObservationKey.gt_actions]
        B, device = x_1.shape[0], x_1.device

        noise = torch.randn_like(x_1)
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (B,), device=device
        ).long()
        x_t = self.scheduler.add_noise(x_1, noise, timesteps)

        target = noise

        obs[ObservationKey.time] = timesteps.float()
        pred = model.forward(obs, noisy_actions=x_t).actions
        loss = F.mse_loss(pred, target)
        return loss

    def compute_action(self, model, obs: dict) -> PolicyOutput:
        with torch.no_grad():
            B = obs[ObservationKey.state].shape[0]
            device = obs[ObservationKey.state].device

            self.scheduler.set_timesteps(self.num_inference_steps, device=device)
            x_t = torch.randn(
                (B, model.action_horizon, model.action_dim), device=device
            )

            for t in self.scheduler.timesteps:
                obs[ObservationKey.time] = torch.full(
                    (B,), float(t.item()), device=device
                )
                model_output = model.forward(obs, noisy_actions=x_t).actions
                x_t = self.scheduler.step(model_output, t, x_t).prev_sample

            return PolicyOutput(actions=x_t)
