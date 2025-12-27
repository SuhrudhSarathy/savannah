import torch
import torch.nn.functional as F

from lerobot_dataset import create_dataset
from diffusion import DiffusionPolicy, DiffusionPolicyConfig
from unet import UNet1D, UNetConfig

import argparse

def train(args):
    model_config = UNetConfig()
    diffusion_config = DiffusionPolicyConfig()

    model = DiffusionPolicy()

    train_dataloader, val_dataloader = create_dataset()

    batch = next(iter(train_dataloader))

    x = torch.randn((8, 16, 2))
    x_image = batch["observation.image"]
    x_state = batch["observation.state"]
    x_action = batch["action"]

    noise = torch.randn_like(x_action)

    out = model(x, x_image, x_state, noise)

    loss = F.mse_loss(out, noise)
    print(loss)

if __name__ == "__main__":
    train(None)
