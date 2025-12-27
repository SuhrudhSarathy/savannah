import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader, Subset
import numpy as np


def create_dataset(
    batch_size: int = 8,
    device: torch.device = torch.device("cpu"),
    train_split: float = 0.95,
):
    delta_timesteps = {
        "observation.image": [-0.1, 0.0],
        "observation.state": [-0.1, 0.0],
        "action": [i / 10 for i in range(-1, 15)],
    }

    # Load the full dataset
    full_dataset = LeRobotDataset("lerobot/pusht", delta_timestamps=delta_timesteps)

    # Calculate split sizes
    full_size = len(full_dataset)
    train_size = int(train_split * full_size)

    # Create shuffled indices for the split
    indices = np.arange(full_size)
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create Subsets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # Training DataLoader
    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=device.type != "cpu",
    )

    # Validation DataLoader
    val_loader = DataLoader(
        val_dataset,
        num_workers=4,
        shuffle=False,  # No need to shuffle validation data
        batch_size=batch_size,
        pin_memory=device.type != "cpu",
    )

    return train_loader, val_loader


if __name__ == "__main__":
    train, _ = create_dataset()
    batch = next(iter(train))
    print("--"*10)
    print(batch["observation.image"].shape)
    print(batch["observation.state"].shape)
    print(batch["action"].shape)
    print("--"*10)
