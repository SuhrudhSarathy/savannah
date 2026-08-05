from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from savannah.utils.policy import PolicyOutput


class Policy(ABC, nn.Module):
    @abstractmethod
    def forward(
        self, obs: dict[str, torch.Tensor], *args, **kwargs
    ) -> PolicyOutput: ...

    def compute_loss(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.objective.compute_loss(self, obs)

    def compute_action(self, obs: dict[str, torch.Tensor]) -> PolicyOutput:
        return self.objective.compute_action(self, obs)

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
