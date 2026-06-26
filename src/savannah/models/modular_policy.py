import torch
import torch.nn as nn

from savannah.models.condition_encoder import ConditionEncoder
from savannah.models.heads.base import ActionHead
from savannah.models.policy import Policy
from savannah.nn.time_embedding import TimeEmbedding
from savannah.objectives import PolicyObjective
from savannah.utils.observation import ObservationKey
from savannah.utils.policy import PolicyOutput


class ModularPolicy(Policy):
    """
    Composable policy that wires a ConditionEncoder, ActionHead, and PolicyObjective.

    The three components are fully independent and can be mixed and matched:
        condition_encoder: (images, state) → (B, N_cond, embed_dim)
        action_head:       (noisy_actions, cond_tokens, t_embed) → (B, T, action_dim)
        objective:         defines noise schedule, loss, and inference sampler

    time_scale controls how the scalar timestep from the objective is passed to
    TimeEmbedding, which is calibrated for ~[0, 100]:
        - FlowMatchingObjective emits t ∈ [0, 1]  → set time_scale=100.0
        - DDIMObjective         emits t ∈ [0, 100] → set time_scale=1.0  (default)
    """

    def __init__(
        self,
        condition_encoder: ConditionEncoder,
        action_head: ActionHead,
        objective: PolicyObjective,
        action_dim: int,
        action_horizon: int,
        state_dim: int,
        time_scale: float = 1.0,
    ):
        super().__init__()

        if condition_encoder.output_dim != action_head.embed_dim:
            raise ValueError(
                f"condition_encoder.output_dim={condition_encoder.output_dim} must equal "
                f"action_head.embed_dim={action_head.embed_dim}"
            )

        self.condition_encoder = condition_encoder
        self.action_head = action_head
        self.objective = objective
        self._action_dim = action_dim
        self._action_horizon = action_horizon
        self._state_dim = state_dim
        self.time_scale = time_scale

        self.time_embedding = TimeEmbedding(action_head.embed_dim)

    def forward(self, obs: dict[str, torch.Tensor], *args, **kwargs) -> PolicyOutput:
        noisy_actions = kwargs.get("noisy_actions")
        if noisy_actions is None:
            raise ValueError("ModularPolicy.forward requires noisy_actions kwarg.")

        t = obs[ObservationKey.time]
        if t.ndim == 1:
            t = t.unsqueeze(1)                                      # (B, 1)
        t_embed = self.time_embedding(t * self.time_scale).unsqueeze(1)  # (B, 1, D)

        language = obs.get(ObservationKey.language, None)
        cond_tokens = self.condition_encoder(
            obs[ObservationKey.images],
            obs[ObservationKey.state],
            language=language,
        )

        pred = self.action_head(noisy_actions, cond_tokens, t_embed)
        return PolicyOutput(actions=pred)
