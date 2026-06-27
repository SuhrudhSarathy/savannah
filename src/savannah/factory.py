import torch
from hydra.utils import instantiate
from omegaconf import DictConfig


def build_policy(cfg: DictConfig) -> torch.nn.Module:
    return instantiate(cfg.model)


def build_task(cfg: DictConfig, device: torch.device):
    return instantiate(cfg.task, device=device)
