import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from savannah.models.backbones import VisionFeatureExtractor


class SpatialSoftmax(nn.Module):
    def __init__(self, height: int = 3, width: int = 3, channel: int = 512):
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel

        # Create coordinates from -1 to 1
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, self.width),
            torch.linspace(-1, 1, self.height),
            indexing="ij",
        )
        self.register_buffer("pos_x", pos_x.reshape(-1))
        self.register_buffer("pos_y", pos_y.reshape(-1))

    def forward(self, x):
        # x: [B, 512, 3, 3]
        b, c, h, w = x.size()
        logits = x.view(b, c, h * w)
        probs = F.softmax(logits, dim=-1)

        # Expected x, y coordinates
        expected_x = torch.sum(probs * self.pos_x, dim=-1)
        expected_y = torch.sum(probs * self.pos_y, dim=-1)

        # Output: [B, 1024] (512 x-coords and 512 y-coords)
        return torch.stack([expected_x, expected_y], dim=-1).view(b, c * 2)


class ResNetBackbone(VisionFeatureExtractor):
    def __init__(self, out_channels: int):
        super().__init__()
        self._out_channels = out_channels
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        self.backbone = nn.Sequential(*list(self.resnet.children())[:-2])
        self.spatial_softmax = SpatialSoftmax()
        self.repoj = nn.Linear(2 * 512, out_channels)

        self.module = nn.Sequential(self.backbone, self.spatial_softmax, self.repoj)

        self.replace_bn_with_gn(self.module)

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x):
        return self.module(x)

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
