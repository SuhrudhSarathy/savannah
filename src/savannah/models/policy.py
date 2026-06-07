from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from einops import rearrange

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

    def _encode_cameras(self, images: list[torch.Tensor]) -> torch.Tensor:
        """
        images: list of (B, N, 3, H, W), one per camera
        returns: (B, N_cams * N_tokens, embed_dim)

        Requires subclass to define: self.vision_encoder, self.time_embedding,
        self.camera_embedding.
        """
        cam_tokens = []

        for cam_idx, img in enumerate(images):
            n = img.shape[1]

            img_r = rearrange(img, "b n c h w -> (b n) c h w")
            tokens = self.vision_encoder(img_r)  # (B*N, N_tokens, embed_dim)
            tokens_r = rearrange(tokens, "(b n) t embed_dim -> b n t embed_dim", n=n)

            t = torch.arange(n, device=img.device).unsqueeze(1)  # (N, 1)
            time_embedding = self.time_embedding(t)  # (N, embed_dim)
            time_embedding = time_embedding.view(1, n, 1, -1)

            camera_embedding = self.camera_embedding(
                torch.tensor(cam_idx, device=img.device)
            )
            camera_embedding = camera_embedding.view(1, 1, 1, -1)

            added_tokens = tokens_r + time_embedding + camera_embedding
            tokens = rearrange(added_tokens, "b n t embed_dim -> b (n t) embed_dim")
            cam_tokens.append(tokens)

        return torch.cat(cam_tokens, dim=1)  # (B, N_cams * N_tokens, embed_dim)

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
