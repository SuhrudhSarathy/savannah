import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize, Normalize, Compose
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from tqdm import tqdm
import os

from unet import UNet
from ddpm_model import DDPM, DDPMConfig
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

time_embed_dim = 512
batch_size = 32
max_timesteps = 1000
model_name = "mnist_diffusion_adagn2.tar"
channels = [32, 64, 128]


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


if __name__ == "__main__":
    # Prepare data
    transforms = Compose([Resize(28), ToTensor(), Normalize([0.5], [0.5])])
    training_data = datasets.MNIST(
        root="data", train=True, transform=transforms, download=True
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    # Prepare Model
    unet = UNet(
        input_channel=1,
        channels=channels,
        max_timesteps=1000,
        time_embed_dim=time_embed_dim,
    )
    config = DDPMConfig(timesteps=max_timesteps, device=device)
    ddpm_model = DDPM(unet, config)

    if model_name in os.listdir(os.getcwd()):
        print("Loading Checkpoints")
        checkpoint_path = model_name
        checkpoint = torch.load(checkpoint_path, map_location=device)

        unet.load_state_dict(checkpoint["model_state_dict"])

    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"Model is {num_params / 1000000:.4f}M")

    # Train loop
    num_epochs = 100
    max_lr = 3e-4
    warmup_epochs = 5

    # Optimiser and loss function
    optimiser = AdamW(unet.parameters(), lr=max_lr)
    loss_fn = nn.MSELoss()

    total_batches = len(train_dataloader)
    total_steps = total_batches * num_epochs
    warmup_steps = total_batches * warmup_epochs

    scheduler = get_cosine_schedule_with_warmup(
        optimiser, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    ema = EMA(unet, decay=0.9999)

    writer = SummaryWriter("runs/mnist_diffusion3")
    global_step = 0
    for epoch in range(num_epochs):
        unet.train()

        pbar = tqdm(train_dataloader, desc=f"Epoch: {epoch + 1}", leave=True)
        for batch_idx, (image, target) in enumerate(pbar):
            image = image.to(device)
            noise = torch.randn_like(image).to(device)
            noise_pred = ddpm_model.forward(image, noise).to(device)

            loss = loss_fn(noise_pred, noise)

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
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
        unet.eval()
        with torch.no_grad():
            samples = ddpm_model.sample((8, 1, 28, 28))
            samples = (samples + 1) / 2

        grid_images = make_grid(samples, 4)
        writer.add_image("Generated_Samples", grid_images, global_step=epoch)

        if epoch % 5 == 0 or epoch == num_epochs:
            # Create a model to hold the EMA weights for saving
            checkpoint_model = UNet(
                input_channel=1,
                channels=channels,
                max_timesteps=max_timesteps,
                time_embed_dim=time_embed_dim,
            ).to(device)
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
