from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from savannah.utils.observation import ObservationKey
from savannah.utils.policy_output import PolicyOutput

if TYPE_CHECKING:
    from savannah.objectives import PolicyObjective


class Policy(ABC, nn.Module):
    @abstractmethod
    def forward(
        self, obs: dict[str, torch.Tensor], *args, **kwargs
    ) -> PolicyOutput: ...

    @abstractmethod
    def compute_loss(self, obs: dict[str, torch.Tensor], *args) -> torch.Tensor: ...

    @abstractmethod
    def compute_action(self, obs: dict[str, torch.Tensor]) -> PolicyOutput: ...

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def action_horizon(self):
        return self._action_horizon

    @property
    def state_dim(self):
        return self._state_dim

    def num_params(self) -> float:
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6

    def param_breakdown(self) -> dict[str, float]:
        """Trainable parameter counts (in millions), split into vision,
        language, and everything else by top-level module name prefix
        (`vision_encoder.` / `language_encoder.`) -- the same groups
        PolicyTrainer uses for per-group learning rates.
        """
        counts = {"vision": 0.0, "language": 0.0, "other": 0.0}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("vision_encoder."):
                counts["vision"] += p.numel()
            elif name.startswith("language_encoder."):
                counts["language"] += p.numel()
            else:
                counts["other"] += p.numel()
        return {k: v / 1e6 for k, v in counts.items()}


class DenoisingPolicy(Policy):
    """A policy whose network predicts a denoising target (velocity, epsilon,
    ...) for noisy actions, in two distinct calls:

        cond = policy.encode_obs(obs)        # vision / state / language, once
        out = policy.denoise(x_t, t, cond)   # one network evaluation

    The policy encodes the observation once and hands `cond` to its
    objective, which only ever calls `denoise`: compute_loss evaluates it at a
    timestep, compute_action runs the full sampler from noise to an action
    chunk.
    """

    objective: "PolicyObjective"

    @abstractmethod
    def encode_obs(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode everything in `obs` that doesn't depend on the noisy
        actions or the time into a condition tensor (B, ...) for `denoise`."""

    @abstractmethod
    def denoise(
        self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """x_t (B, action_horizon, action_dim), t (B,) -> same shape as x_t."""

    def compute_loss(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        cond = self.encode_obs(obs)
        return self.objective.compute_loss(self, cond, obs[ObservationKey.gt_actions])

    def compute_action(self, obs: dict[str, torch.Tensor], **kwargs) -> PolicyOutput:
        cond = self.encode_obs(obs)
        return self.objective.compute_action(self, cond, **kwargs)

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        *args,
        noisy_actions: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        **kwargs,
    ) -> PolicyOutput:
        """One-off encode + denoise, e.g. for hooks/debug scripts. Samplers
        should call encode_obs once and denoise per step instead."""
        if noisy_actions is None or t is None:
            raise AssertionError("Pass noisy_actions and t for the model to run")
        return PolicyOutput(
            actions=self.denoise(noisy_actions, t, self.encode_obs(obs))
        )
