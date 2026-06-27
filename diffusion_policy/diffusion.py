from dataclasses import dataclass

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from torch import Tensor
from unet import UNet1D, UNetConfig


@dataclass
class DiffusionPolicyConfig:
    timesteps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 0.02
    device: torch.device = torch.device("cpu")


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        model_config: UNetConfig = UNetConfig(),
        config: DiffusionPolicyConfig = DiffusionPolicyConfig(),
    ):
        super().__init__()

        self.model_config = model_config
        self.model = nn.ModuleDict({"unet_policy": UNet1D(model_config)})
        self.config = config
        self.ddim_scheduler = DDIMScheduler(
            num_train_timesteps=self.config.timesteps,
            beta_start=self.config.beta_start,
            beta_end=self.config.beta_end,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
            clip_sample=True,
        )

        self.model.to(config.device)

    def forward(
        self, x: Tensor, x_image: Tensor, x_state: Tensor, noise: Tensor
    ) -> Tensor:
        # x (B, n_pred, state_dim)
        # x_state (B, n_obs, state_dim)
        # x_image (B, n_obs, imgC, H, W)
        # noise (B, n_pred, state_dim)
        t = (
            torch.randint(0, self.config.timesteps, size=(x.shape[0], 1))
            .to(self.config.device)
            .to(torch.long)
        )

        x_noised = self.ddim_scheduler.add_noise(
            original_samples=x, noise=noise, timesteps=t
        ).to(device=self.config.device)

        x_time = t

        # Noise predicted by the model
        epsilon = self.model["unet_policy"](x_noised, x_image, x_state, x_time)

        return epsilon

    @torch.no_grad()
    def get_action(
        self,
        x: Tensor,
        x_image: Tensor,
        x_state: Tensor,
        inference_timesteps: int = 50,
    ):
        B = x.shape[0]

        self.ddim_scheduler.set_timesteps(inference_timesteps, device=x.device)

        x_t = x.clone()

        for t in self.ddim_scheduler.timesteps:
            x_time = torch.full((B,), t, device=x.device, dtype=torch.long)

            pred_noise = self.model["unet_policy"](x_t, x_image, x_state, x_time)

            x_t = self.ddim_scheduler.step(pred_noise, t, x_t).prev_sample

        return x_t
