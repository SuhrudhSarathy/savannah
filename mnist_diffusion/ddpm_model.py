from dataclasses import dataclass

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from unet import UNet


@dataclass
class DDPMConfig:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    device: torch.device = torch.device("cpu")


class DDPM:
    def __init__(self, model: UNet, config: DDPMConfig = DDPMConfig()):
        self.model = model
        self.config = config
        self.ddpm_scheduler = DDPMScheduler(
            num_train_timesteps=self.config.timesteps,
            beta_start=self.config.beta_start,
            beta_end=self.config.beta_end,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
        )

        self.model.to(device=self.config.device)

    def forward(self, x: torch.tensor, noise: torch.tensor) -> torch.tensor:
        # X (B, C, H, W), Noise (B, C, H, W), t (B, 1)
        t = torch.randint(0, self.config.timesteps, size=(x.shape[0],)).to(
            self.config.device
        )
        x_noised = self.ddpm_scheduler.add_noise(
            original_samples=x, noise=noise, timesteps=t
        ).to(device=self.config.device)

        # Noise of the model
        epsilon = self.model(x_noised, t)

        return epsilon

    @torch.no_grad
    def sample(self, img_size: tuple[int, int, int, int]):
        # This is unconditional sampling. So we just complete the sampling steps directly
        x_t = torch.randn(img_size).to(self.config.device)
        for t in range(self.config.timesteps - 1, -1, -1):
            t_tensor = torch.full((img_size[0],), t, dtype=torch.long).to(
                self.config.device
            )

            pred_noise = self.model(x_t, t_tensor)

            x_t = self.ddpm_scheduler.step(pred_noise, t, x_t).prev_sample

        return x_t
