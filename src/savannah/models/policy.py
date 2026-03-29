from typing import Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn

from savannah.utils.policy import PolicyOutput
from typing import Optional


class Policy(ABC, nn.Module):
    @abstractmethod
    def forward(
        self, obs: dict[str, torch.Tensor], *args, **kwargs
    ) -> PolicyOutput: ...

    @abstractmethod
    def compute_loss(
        self,
        obs: dict[str, torch.Tensor],
    ): ...

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
