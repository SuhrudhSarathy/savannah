import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from savannah.models.policy import Policy
from savannah.objectives import OverfitObjective
from savannah.utils.device import get_device
from savannah.utils.observation import ObservationKey
from savannah.utils.policy import PolicyOutput


class DumbPolicy(Policy):
    def __init__(self, input_dim: int, output_dim: int, embed_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, output_dim),
        )

        self._state_dim = 2
        self._action_dim = 2
        self._action_horizon = 2
        self.num_cameras = 1

        self.objective = OverfitObjective()

    def forward(self, obs: dict[str, torch.Tensor], *args, **kwargs) -> PolicyOutput:
        x_gt = kwargs["noisy_actions"]
        out = self.model(x_gt)

        return PolicyOutput(actions=out)


if __name__ == "__main__":
    device = get_device()

    policy = DumbPolicy(2, 2, 64).to(device)

    x_gt = torch.randn(8, 2, 2).to(device)
    input = {ObservationKey.gt_actions: x_gt}

    optimiser = AdamW(policy.parameters(), lr=1e-4)

    first_loss = None
    for i in range(500):
        out = policy(input).actions
        loss = F.mse_loss(out, x_gt)

        if first_loss is None:
            first_loss = loss

        optimiser.zero_grad()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        loss.backward()
        optimiser.step()

    print(f"Iter [{i + 1}/500]: Loss: {first_loss.item()} -> {loss.item()}")
