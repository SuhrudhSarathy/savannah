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

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x):
        features = self.backbone(x)
        return self.proj(features)
