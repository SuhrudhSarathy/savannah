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
