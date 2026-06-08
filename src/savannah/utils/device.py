import torch

from savannah.utils.log import logger


def get_device():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
        torch.mps.empty_cache()
        torch.mps.synchronize()

    logger.info("Using device: {}", device)
    return device
