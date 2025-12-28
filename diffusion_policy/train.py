import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from dataclasses import asdict
from tqdm import tqdm

from lerobot_dataset import create_dataset
from diffusion import DiffusionPolicy, DiffusionPolicyConfig
from unet import UNet1D, UNetConfig
import os

import argparse
import wandb
import copy

dtype = torch.float32
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
    torch.mps.empty_cache()
    torch.mps.synchronize()
print("Currently using device: ", device)

class EMA:
    """
    Exponential Moving Average utility for model weights.
    A shadow copy of the model is maintained and updated after each optimizer step.
    """

    def __init__(self, model, decay=0.9999):
        # Initialize EMA weights from the current model weights
        self.model = model
        self.decay = decay
        self.shadow = copy.deepcopy(self.model)
        self.shadow.train()  # Set to train mode for compatibility with UNet

    @torch.no_grad()
    def update(self):
        """Update the shadow weights using the current model weights."""
        model_params = dict(self.model.named_parameters())
        shadow_params = dict(self.shadow.named_parameters())

        for name, param in model_params.items():
            shadow_params[name].data = shadow_params[
                name
            ].data * self.decay + param.data * (1.0 - self.decay)

    def copy_to(self, target_model):
        """Copy EMA weights to a target model (e.g., for checkpointing or inference)."""
        target_params = dict(target_model.named_parameters())
        shadow_params = dict(self.shadow.named_parameters())
        for name, param in target_params.items():
            param.data.copy_(shadow_params[name].data)


def train(args):
    model_config = UNetConfig()
    diffusion_config = DiffusionPolicyConfig()
    diffusion_config.device = device

    model = DiffusionPolicy(model_config=model_config, config=diffusion_config).to(device)

    train_dataloader, val_dataloader = create_dataset()

    wandb.init(
        project="act-training",
        config={
            "learning_rate": args.lr,
            "architecture": "ACT",
            "dataset": "push_T",
            "training_steps": int(args.max_timesteps),
            "description": "full push-t training",
            "full_mode_config": asdict(model_config),
            "full_diffusion_config": asdict(diffusion_config),
        },
    )

    optimiser = AdamW(model.parameters(), lr=args.lr)

    global_step = 0
    best_val_loss = float("inf")
    ema = EMA(model, decay=0.99)

    os.makedirs("checkpoints", exist_ok=True)

    while global_step < args.max_timesteps:

        print("----------------- Running Training Run -------------")
        model.train()
        for batch in tqdm(train_dataloader):
            # 1. Data Prep
            x_image = batch["observation.image"].to(device)
            x_state = (batch["observation.state"].to(device) - 256.0) / 512.0
            x_action = (batch["action"].to(device) - 256.0) / 512.0
            x = torch.randn_like(x_action).to(device)

            noise = torch.randn_like(x_action).to(device)

            # Forward pass
            out = model(x, x_image, x_state, noise)

            # Loss calculation
            loss = F.mse_loss(out, noise)

            # Backward pass
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # Update EMA Model
            ema.update()

            wandb.log(
                {
                    "train/total_loss": loss.item(),
                    "train/lr": optimiser.param_groups[0]["lr"],
                    "global_step": global_step,
                }
            )
            global_step += 1

        # Validation
        print("----------- Validating Learning -------------")
        model.eval()
        val_losses = []

        with torch.no_grad():
            # Validate on a few batches to save time
            for v_batch in tqdm(val_dataloader):
                v_image = v_batch["observation.image"].to(device)
                v_state = (v_batch["observation.state"].to(device) - 256.0) / 512.0
                v_target = (v_batch["action"].to(device) - 256.0) / 512.0
                v_x = torch.randn_like(v_target).to(device)

                out = model.get_action(v_x, v_image, v_state, 20)

                loss = F.mse_loss(out, v_target)

                val_losses.append(loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        wandb.log({"val/total_loss": avg_val_loss, "global_step": global_step})

        print("------------- Checkpointing Logic -------------")
        checkpoint_model = DiffusionPolicy(model_config=model_config, config=diffusion_config).to(device)
        ema.copy_to(checkpoint_model)
        checkpoint = {
            "model_state_dict": checkpoint_model.state_dict(),
            "optimizer_state_dict": optimiser.state_dict(),
            "global_step": global_step,
            "val_loss": avg_val_loss,
        }

        # Save 'latest' locally
        latest_path = "checkpoints/latest.ckpt"
        torch.save(checkpoint, latest_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = "checkpoints/best_model.ckpt"
            torch.save(checkpoint, best_path)

            print(
                f"New best model saved at step {global_step} (Loss: {avg_val_loss:.4f})"
            )

        if global_step > args.max_timesteps:
            print("Finished Training for full max timesteps")
            break

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Diffusion Policy", description="Train diffusion policy for Push-T")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch Size for training")
    parser.add_argument("--max-timesteps", type=int, default=100000, help="Max timesteps for training")

    args = parser.parse_args()

    train(args)
