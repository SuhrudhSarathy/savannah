from savannah.models.encoders.vision_encoder import VisionEncoder
from savannah.objectives import DDIMObjective
from savannah.models.dit import DiTPolicy
from savannah.models.backbones.lora_resnet_backbone import LoRAResNetBackbone
from savannah.utils.log import logger
from savannah.utils.device import get_device

import torch


def main():
    device = get_device()
    checkpoint_path = (
        "/home/suhrudhs/savannah/savannah/checkpoints/dit_lora_pusht/latest.ckpt"
    )
    logger.info("Loading checkpoint: {}", checkpoint_path)

    vision_backbone = LoRAResNetBackbone(96)
    encoder = VisionEncoder(vision_backbone, 64)

    objective = DDIMObjective(100, 10, 1e-4, 0.02, True)
    policy = DiTPolicy(
        64, 4, 4, 256, 0.1, 4, 4, 256, 0.1, 2, 2, 12, 1, encoder, None, objective
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint["ema_state_dict"])
    policy.eval()


if __name__ == "__main__":
    main()
