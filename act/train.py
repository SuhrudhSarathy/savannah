import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from model import ACTConfig, ACTPolicy
from lerobot_dataset import create_dataset

from dataclasses import asdict, dataclass
from tqdm import tqdm
import os
import wandb
import math

from eval import evaluate_and_log

dtype = torch.float32
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
    torch.mps.empty_cache()
    torch.mps.synchronize()
print("Currently using device: ", device)


@dataclass
class ACTTrainingConfig:
    lr: float = 3e-5
    beta: float = 100
    batch_size: int = 64
    warmup_steps: int = 1000
    training_steps: int = 100_000 


def get_scheduler(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def kl_divergence_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    kl_elementwise = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl_elementwise.sum(dim=1).mean()


def train(config: ACTTrainingConfig = ACTTrainingConfig()):
    # Create Dataset
    train_dataloader, val_dataloader = create_dataset(config.batch_size)

    # Create the Model
    act_config = ACTConfig()
    act_config.chunk_size = 10
    act_config.state_dim = 2

    wandb.init(
        project="act-training",
        config={
            "learning_rate": config.lr,
            "architecture": "ACT",
            "dataset": "push_T",
            "training_steps": int(config.training_steps),
            "description": "full push-t train on mac",
            "full_train_config": asdict(config),
            "full_model_config": asdict(act_config),
        },
    )

    act = ACTPolicy(act_config).to(device)

    # Use the AdamW optimizer as defined in your setup
    optimiser = AdamW(act.parameters(), lr=config.lr)

    # lr scheduler
    scheduler = get_scheduler(
        optimiser,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.training_steps,
    )

    global_step = 0
    best_val_loss = float("inf")

    os.makedirs("checkpoints", exist_ok=True)
    
    while global_step < config.training_steps:
        print("----------------- Running Training Run -------------")
        act.train()

        for batch in tqdm(train_dataloader):
            # 1. Data Preparation
            x_image = batch["observation.image"].to(device)
            x_state = (batch["observation.state"].to(device) - 256.0) / 512.0
            x_target = (batch["action"].to(device) - 256.0) / 512.0

            # 2. Forward Pass
            x_out, [mu, log_var] = act(x_image, x_state, x_target)

            # 3. Loss Calculation
            # per_action_loss = x_target - x_out
            recon_loss = F.l1_loss(x_out, x_target)
            kl_loss = kl_divergence_loss(mu, log_var)
            total_loss = recon_loss + config.beta * kl_loss

            # 4. Backward Pass
            optimiser.zero_grad()
            total_loss.backward()
            optimiser.step()
            scheduler.step()

            # 5. Logging Training Metrics
            wandb.log(
                {
                    "train/total_loss": total_loss.item(),
                    "train/recon_loss": recon_loss.item(),
                    "train/kl_loss": kl_loss.item(),
                    "train/lr": optimiser.param_groups[0]['lr'],
                    "global_step": global_step,
                }
            )
            global_step += 1

        # Validation
        print("----------- Validating Learning -------------")
        act.eval()
        val_losses = []

        with torch.no_grad():
            # Validate on a few batches to save time
            for _, v_batch in enumerate(val_dataloader):
                v_image = v_batch["observation.image"].to(device)
                v_state = (
                    v_batch["observation.state"].to(device) - 256.0
                ) / 512.0
                v_target = (v_batch["action"].to(device) - 256.0) / 512.0

                v_out, [v_mu, v_log_var] = act(v_image, v_state, v_target)
                v_recon = F.l1_loss(v_out, v_target)
                v_kl = kl_divergence_loss(v_mu, v_log_var)
                val_losses.append(v_recon.item() + config.beta * v_kl.item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        wandb.log({"val/total_loss": avg_val_loss, "global_step": global_step})

        # --- CHECKPOINTING LOGIC ---
        checkpoint = {
            "model_state_dict": act.state_dict(),
            "optimizer_state_dict": optimiser.state_dict(),
            "global_step": global_step,
            "val_loss": avg_val_loss,
            "config": act_config,
        }

        # Save 'latest' locally
        latest_path = "checkpoints/latest.ckpt"
        torch.save(checkpoint, latest_path)

        # If this is the best model so far, save it
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = "checkpoints/best_model.ckpt"
            torch.save(checkpoint, best_path)

            print(
                f"New best model saved at step {global_step} (Loss: {avg_val_loss:.4f})"
            )
        
        evaluate_and_log(act, device, idx=global_step)

        if global_step >= config.training_steps:
            break

    wandb.finish()


def test():
    device = torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
    config = ACTConfig()
    config.chunk_size = 50
    config.state_dim = 2

    B = 8
    beta = 10

    x_image = torch.randn((B, 3, *config.img_shape)).to(device)
    x_state = torch.randn((B, 1, config.state_dim)).to(device)

    x_target = torch.randn((B, config.chunk_size, config.state_dim)).to(device)

    act = ACTPolicy(config).to(device)
    print(f"Num of params: {act._num_params()}M")

    x_out, [mu, log_var] = act(x_image, x_state, x_target)
    print(x_out.shape)

    reconstruction_loss = F.l1_loss(x_out, x_target)
    regularisation_loss = kl_divergence_loss(mu, log_var)

    print(reconstruction_loss + beta * regularisation_loss)


if __name__ == "__main__":
    train_config = ACTTrainingConfig()
    train(train_config)
