import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

if not OmegaConf.has_resolver("len"):
    OmegaConf.register_new_resolver("len", len)

if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)


def build_policy(cfg: DictConfig) -> torch.nn.Module:
    return instantiate(cfg.model)


def build_task(cfg: DictConfig, device: torch.device):
    return instantiate(cfg.task, device=device)


def build_policy_from_pretrained(pretrained_name_or_path, **kwargs) -> torch.nn.Module:
    """A policy, in eval mode, from a Hub repo id or local checkpoint dir. See
    savannah.pretrained.load_pretrained for the keyword arguments."""
    # Imported here: savannah.pretrained itself imports build_policy.
    from savannah.pretrained import build_policy_from_pretrained as _build

    return _build(pretrained_name_or_path, **kwargs)
