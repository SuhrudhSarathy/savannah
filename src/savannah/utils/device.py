import torch


def get_device():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
        torch.mps.empty_cache()
        torch.mps.synchronize()

    print("Currently using device: ", device)

    return device
