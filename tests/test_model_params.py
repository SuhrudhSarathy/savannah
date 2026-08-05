from savannah.models.backbones.clip_backbone import CLIPBackbone
import torch

from savannah.models.backbones.lora_resnet_backbone import LoRAResNetBackbone
from savannah.models.dit import DiTPolicy
from savannah.models.encoders.vision_encoder import VisionEncoder
from savannah.objectives import DDIMObjective
from savannah.utils.device import get_device
from savannah.utils.log import logger


def main():
    device = get_device()
    checkpoint_path = (
        "/home/suhrudh/software/python/savannah/checkpoints/overfit/overfit_final.ckpt"
    )
    logger.info("Loading checkpoint: {}", checkpoint_path)

    vision_backbone = CLIPBackbone(
        512, 226, "openai/clip-vit-base-patch32", True, False
    )
    encoder = VisionEncoder(vision_backbone, 128, num_cameras=3, num_obs=1)

    objective = DDIMObjective(100, 10, 1e-4, 0.02, True)
    policy = DiTPolicy(
        128, 4, 4, 512, 0.1, 4, 4, 512, 0.1, 4, 4, 20, 3, encoder, None, objective
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint)
    policy.eval()

    print(policy.multi_choice_attn.weight_matrix)


if __name__ == "__main__":
    main()
