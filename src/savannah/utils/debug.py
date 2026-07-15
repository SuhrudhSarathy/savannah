import torch

from savannah.utils.log import logger


def debug_stat(name: str, t: torch.Tensor) -> None:
    logger.debug(
        "{}: shape={}, min={:.6f}, max={:.6f}, mean={:.6f}, nan={}, inf={}",
        name,
        tuple(t.shape),
        t.min().item(),
        t.max().item(),
        t.mean().item(),
        torch.isnan(t).any().item(),
        torch.isinf(t).any().item(),
    )
