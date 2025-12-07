import torch
import time
from unet import UNet


dtype = torch.float32
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
    torch.mps.empty_cache()
    torch.mps.synchronize()
print("Currently using device: ", device)

time_embed_dim = 64
batch_size = 32
max_timesteps = 10

if __name__ == "__main__":
    model = UNet(
        input_channel=1,
        channels=[3, 16, 64, 128],
        max_timesteps=max_timesteps,
        time_embed_dim=64,
    ).to(device=device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model is {num_params / 1_000_000:.4f}M")

    x = torch.randn((batch_size, 1, 128, 128)).to(device=device)
    t = torch.randint(0, max_timesteps, (batch_size,)).to(device=device)

    model.eval()
    start = time.time()
    out = model(x, t)
    stop = time.time()
    print(out.cpu().shape, stop - start)
