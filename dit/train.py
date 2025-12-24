import argparse
import copy
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from ddpm_model import DDPM, DDPMConfig
from model import DiT, DiTConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from torchvision.utils import make_grid
from tqdm import tqdm

dtype = torch.float32
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
    torch.mps.empty_cache()
    torch.mps.synchronize()
print("Currently using device: ", device)

max_timesteps = 1000
model_name = "dit_small.tar"
tensorboard_dir = "runs/mnist_dit_small"


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
    # Args
    num_epochs = args.epochs
    max_lr = args.lr
    batch_size = args.batch_size

    # Prepare data
    transforms = Compose([Resize(28), ToTensor(), Normalize([0.5], [0.5])])
    training_data = datasets.MNIST(
        root="data", train=True, transform=transforms, download=True
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    model = DiT()
    config = DDPMConfig(timesteps=max_timesteps, device=device)
    ddpm_model = DDPM(model, config)

    if model_name in os.listdir(os.getcwd()):
        print("Loading Checkpoints")
        checkpoint_path = model_name
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model is {num_params / 1000000:.4f}M")

    # Optimiser and loss function
    optimiser = AdamW(model.parameters(), lr=max_lr)
    loss_fn = nn.MSELoss()

    ema = EMA(model, decay=0.9999)

    writer = SummaryWriter(tensorboard_dir)
    global_step = 0
    for epoch in range(num_epochs):
        model.train()

        pbar = tqdm(train_dataloader, desc=f"Epoch: {epoch + 1}", leave=True)
        for batch_idx, (image, target) in enumerate(pbar):
            image = image.to(device)
            target = target.unsqueeze(1).to(device)
            noise = torch.randn_like(image).to(device)
            noise_pred = ddpm_model.forward(image, target, noise).to(device)

            loss = loss_fn(noise_pred, noise)

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            ema.update()

            pbar.set_description(f"Epoch: {epoch + 1}")
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "LR": f"{optimiser.param_groups[0]['lr']:.6f}",
                    "Batch Idx": batch_idx,
                }
            )

            writer.add_scalar("Loss/Train", loss.item(), global_step)
            writer.add_scalar("LR/Train", optimiser.param_groups[0]["lr"], global_step)

            global_step += 1

        # After every epoch, write loss to
        model.eval()
        with torch.no_grad():
            samples = ddpm_model.sample((8, 1, 28, 28), num_classes=10)
            samples = (samples + 1) / 2

        grid_images = make_grid(samples, 4)
        writer.add_image("Generated_Samples", grid_images, global_step=epoch)

        if epoch % 5 == 0 or epoch == num_epochs:
            # Create a model to hold the EMA weights for saving
            checkpoint_model = DiT().to(device)
            ema.copy_to(checkpoint_model)

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": checkpoint_model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
                "loss": loss,
            }
            torch.save(checkpoint, model_name)
            print(f"--- Saved checkpoint at epoch {epoch} ---")
    writer.close()


def sample(args):
    num_samples = args.num_samples
    model = DiT()
    config = DDPMConfig(timesteps=max_timesteps, device=device)
    ddpm_model = DDPM(model, config)

    if model_name in os.listdir(os.getcwd()):
        print("Loading Checkpoints")
        checkpoint_path = model_name
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    with torch.no_grad():
        samples = ddpm_model.sample((num_samples, 1, 28, 28), 10)
        samples = torch.clamp(samples, -1, 1)
        samples = (samples + 1) / 2

    grid_images = make_grid(samples, 4)

    from torchvision.utils import save_image

    save_image(grid_images, "grid_image.png")


def sample_with_class(args):
    num_samples = args.num_samples
    model = DiT()
    config = DDPMConfig(timesteps=max_timesteps, device=device)
    ddpm_model = DDPM(model, config)

    if model_name in os.listdir(os.getcwd()):
        print("Loading Checkpoints")
        checkpoint_path = model_name
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    with torch.no_grad():
        classes = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
        samples = ddpm_model.sample_with_class(
            (num_samples, 1, 28, 28), class_tensor=classes
        )
        samples = torch.clamp(samples, -1, 1)
        samples = (samples + 1) / 2

    grid_images = make_grid(samples, 4)

    from torchvision.utils import save_image

    save_image(grid_images, "grid_image_with_class.png")


def main():
    parser = argparse.ArgumentParser(
        prog="DiT-Mnist", description="Train or Sample from DiT"
    )
    subparsers = parser.add_subparsers(
        dest="mode", help="Execution mode", required=True
    )

    # --- Train Subparser ---
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--batch_size", type=int, default=64)

    # --- Sample Subparser ---
    sample_parser = subparsers.add_parser(
        "sample", help="Generate samples from a checkpoint"
    )
    sample_parser.add_argument(
        "--num_samples", type=int, default=16, help="Number of images to generate"
    )

    args = parser.parse_args()

    # Route to the correct function based on the mode
    if args.mode == "train":
        train(args)
    elif args.mode == "sample":
        sample_with_class(args)


if __name__ == "__main__":
    main()
