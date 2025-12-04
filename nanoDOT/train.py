import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from model import NanoDOT, NanoDOTConfig
import tiktoken
from tqdm import tqdm

import time

from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup
import os

# Parameters
dtype = torch.float32
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
print("Currently using device: ", device)


class CharLevelEncoder:
    def __init__(self, input_txt: str):
        with open(input_txt, "r") as f:
            text = f.read()

        chars = sorted(set(text))
        vocab_size = len(chars)

        self.encode = lambda s: [chars.index(c) for c in s]
        self.decode = lambda l: "".join([chars[i] for i in l])
        self.n_vocab = vocab_size


batch_size = 64
# enc = tiktoken.get_encoding("o200k_base")
enc = CharLevelEncoder("input.txt")

config = NanoDOTConfig(
    ctx_size=32,
    n_embed=32,
    n_head=4,
    dropout=0.2,
    vocab_size=enc.n_vocab,
    device=device,
    num_attention_blocks=6,
)


class ShakespeareDataLoader(Dataset):
    def __init__(self, encoded_tokens: list, config: NanoDOTConfig):
        self.encoded_tokens = encoded_tokens
        self.config = config

    def __len__(self):
        return len(self.encoded_tokens) - self.config.ctx_size

    def __getitem__(self, idx):
        x = self.encoded_tokens[idx : idx + self.config.ctx_size]
        y = self.encoded_tokens[idx + 1 : idx + self.config.ctx_size + 1]

        return x, y


def test_model_generation(use_checkpoint: bool = True):
    model = NanoDOT(config).to(device)

    if use_checkpoint:
        checkpoint_path = "nanoDOT_smol.tar"
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
    model.compile()

    tokens = enc.encode("\n")
    tokens = torch.tensor([tokens]).reshape(1, 1).to(device)

    output_tensor = model.generate(tokens, 5000)
    output_tokens_list = output_tensor[0].cpu().tolist()

    # 3. Decode the clean list of integers.
    text = enc.decode(output_tokens_list)
    with open("output.txt", "w") as file:
        file.write(text)


def test_data_loader():
    dataset_src_path = "input.txt"

    with open(dataset_src_path, "r") as file:
        src = file.read()

    dataset_tokens = torch.tensor(enc.encode(src), dtype=torch.long)
    split_index = int(0.9 * len(dataset_tokens))

    train_tokens = dataset_tokens[:split_index]
    val_tokens = dataset_tokens[split_index:]

    train_dataset = ShakespeareDataLoader(encoded_tokens=train_tokens, config=config)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    val_dataset = ShakespeareDataLoader(encoded_tokens=val_tokens, config=config)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
    )

    for _, (x, y) in enumerate(train_dataloader):
        print(x.shape, y.shape)
        break


def train():
    dataset_src_path = "input.txt"

    with open(dataset_src_path, "r") as file:
        src = file.read()

    dataset_tokens = torch.tensor(enc.encode(src), dtype=torch.long)
    split_index = int(0.9 * len(dataset_tokens))

    train_tokens = dataset_tokens[:split_index]
    val_tokens = dataset_tokens[split_index:]

    print(f"Total Train tokens: {train_tokens.shape}")

    train_dataset = ShakespeareDataLoader(encoded_tokens=train_tokens, config=config)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    val_dataset = ShakespeareDataLoader(encoded_tokens=val_tokens, config=config)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    model = NanoDOT(config).to(device)
    if "nanoDOT_smol.tar" in os.listdir(os.getcwd()):
        print("Loading Checkpoints")
        checkpoint_path = "nanoDOT_smol.tar"
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model is {num_params}")

    # Training parameters
    num_epochs = 50
    max_lr = 3e-4
    warmup_epochs = 5

    total_batches = len(train_dataloader)
    total_steps = total_batches * num_epochs
    warmup_steps = total_batches * warmup_epochs

    # Define the optimiser and caluclate the loss
    optimiser = AdamW(model.parameters(), max_lr)

    scheduler = get_cosine_schedule_with_warmup(
        optimiser, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    val_loss = float("inf")
    model.compile()

    # Tensorboard writer
    writer = SummaryWriter("runs/nanoDOT_smol")
    global_step = 0
    for epoch in range(num_epochs):
        pbar = tqdm(train_dataloader, desc=f"Epoch: {epoch + 1}", leave=True)
        # Go Through the training loop
        # Set model to train
        model.train()
        for batch_idx, (x, y) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            # Compute the loss
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = y.view(B * T)

            loss = F.cross_entropy(logits, targets)

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()

            # 2. Update the progress bar and description
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

        model.eval()
        losses = []
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_dataloader):
                x = x.to(device)
                y = y.to(device)

                logits = model(x)

                # Compute the loss
                B, T, C = logits.shape
                logits = logits.view(B * T, C)
                targets = y.view(B * T)

                loss = F.cross_entropy(logits, targets)
                losses.append(loss.cpu())

        val_loss_current = torch.tensor(losses).mean().item()
        print(f"*** Epoch {epoch + 1} Validation Loss: {val_loss_current:.4f} ***")
        writer.add_scalar("Loss/Validation", val_loss_current, epoch)
        if val_loss_current < val_loss:
            # Save Model checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimiser.state_dict(),
            }
            torch.save(checkpoint, "nanoDOT_smol.tar")
            val_loss = val_loss_current

    writer.close()


if __name__ == "__main__":
    # train()
    test_model_generation()
