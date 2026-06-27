import sys

from savannah.utils.observation import ObservationKey

try:
    import imp
except ImportError:
    import importlib.util
    from types import ModuleType

    imp = ModuleType("imp")
    sys.modules["imp"] = imp

import os

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from tqdm import tqdm

from savannah.factory import build_policy, build_task
from savannah.trainer.trainer import PolicyTrainer, TrainConfig
from savannah.utils.device import get_device
from savannah.utils.log import logger, setup_logging
from savannah.utils.logger import ExperimentLogger


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log_path = setup_logging(log_dir="logs", level=cfg.log_level)

    fraction = float(os.environ.get("CUDA_MEMORY_FRACTION", "1.0"))
    if fraction < 1.0 and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)
        logger.info("VRAM cap: {:.0%} of device memory", fraction)

    device = get_device()
    policy = build_policy(cfg).to(device)

    logger.info("Policy params: {:,}", policy.num_params())

    # Construct fixed observation, state, time
    B = 1

    torch.manual_seed(42)

    x_img = [torch.randn(B, 2, 3, 96, 96).to(device)]

    x_state = torch.randn(B, 2, 2).to(device)
    x_time = torch.randint(0, 100, (B, 1)).float().to(device)

    noisy_actions = torch.randn(B, 8, 2).to(device)
    x_gt_actions = torch.randn_like(noisy_actions).to(device)

    observation = {
        ObservationKey.images: x_img,
        ObservationKey.state: x_state,
        ObservationKey.time: x_time,
    }

    optimiser = AdamW(policy.parameters(), lr=3e-4)
    loss_init = None
    pbar = tqdm(range(2000))

    policy.eval()

    for i in pbar:
        out = policy.forward(observation, noisy_actions=noisy_actions).actions
        loss = F.mse_loss(out, x_gt_actions)

        if i == 0:
            loss_init = loss

        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimiser.step()

        pbar.set_postfix(loss=f"{loss.item():.6f}")

    logger.warning(f"Loss at start: {loss_init.item()}. Final loss: {loss.item()}")


if __name__ == "__main__":
    main()
