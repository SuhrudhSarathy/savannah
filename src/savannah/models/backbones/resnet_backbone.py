import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange

from savannah.models.backbones import VisionFeatureExtractor


class ResNetBackbone(VisionFeatureExtractor):
    def __init__(self, out_channels: int):
        super().__init__()
        self._out_channels = out_channels
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        resnet.maxpool = nn.Identity()

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.proj = nn.Conv2d(512, out_channels, kernel_size=1)

        # self.replace_bn_with_gn(self.backbone)

    def replace_bn_with_gn(self, model, num_groups=32):
        for name, module in model.named_children():
            if isinstance(module, nn.BatchNorm2d):
                # Get the number of channels from the existing BN layer
                num_channels = module.num_features

                # Ensure num_groups is valid (must divide num_channels)
                # Fallback to 1 group (LayerNorm style) if 32 doesn't divide evenly
                actual_groups = num_groups if num_channels % num_groups == 0 else 1

                # Create the new GroupNorm layer
                gn = nn.GroupNorm(actual_groups, num_channels)

                # Replace the layer in the model
                setattr(model, name, gn)
            else:
                # Recurse through sub-modules (like Sequential or custom Blocks)
                self.replace_bn_with_gn(module, num_groups)

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x):
        features = self.backbone(x)
        return self.proj(features)
