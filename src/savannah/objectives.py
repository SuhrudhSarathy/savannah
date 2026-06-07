from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

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
