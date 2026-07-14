import math

import torch
import torch.nn as nn
import torchvision.models as models

from savannah.models.backbones import VisionFeatureExtractor
from savannah.models.backbones.resnet_backbone import SpatialSoftmax


class LoRAConv2d(nn.Module):
    def __init__(self, original_conv: nn.Conv2d, r: int = 8, lora_alpha: int = 16):
        super().__init__()
        self.original_conv = original_conv
        self.original_conv.weight.requires_grad = False
        if self.original_conv.bias is not None:
            self.original_conv.bias.requires_grad = False

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # Low-rank layers
        self.lora_A = nn.Conv2d(
            original_conv.in_channels,
            r,
            original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )
        self.lora_B = nn.Conv2d(
            r, original_conv.out_channels, kernel_size=1, bias=False
        )

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Track the merge state
        self.merged = False

    def merge(self):
        if not self.merged:
            weight_B = self.lora_B.weight.view(self.lora_B.out_channels, self.r)
            weight_A = self.lora_A.weight.view(self.r, -1)

            delta_w = torch.matmul(weight_B, weight_A)

            delta_w = delta_w.view(self.original_conv.weight.shape)

            self.original_conv.weight.data += delta_w * self.scaling
            self.merged = True

    def unmerge(self):
        if self.merged:
            weight_B = self.lora_B.weight.view(self.lora_B.out_channels, self.r)
            weight_A = self.lora_A.weight.view(self.r, -1)
            delta_w = torch.matmul(weight_B, weight_A).view(
                self.original_conv.weight.shape
            )

            self.original_conv.weight.data -= delta_w * self.scaling
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.original_conv(x)
        else:
            return self.original_conv(x) + self.lora_B(self.lora_A(x)) * self.scaling


class LoRAResNetBackbone(VisionFeatureExtractor):
    def __init__(self, out_channels: int):
        super().__init__()
        self._out_channels = out_channels
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        self.backbone = nn.Sequential(*list(self.resnet.children())[:-2])
        self.add_lora(self.backbone)

        # Matches ResNetBackbone's contract: pool the (B, 512, H', W') feature
        # map from resnet18's layer4 into a flat (B, out_channels) descriptor,
        # since VisionEncoder expects a single token per image, not one token
        # per spatial location.
        self.spatial_softmax = SpatialSoftmax()
        self.channel_proj = nn.Linear(2 * 512, out_channels)

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = self.backbone(x)
        x_pooled = self.spatial_softmax(x_out)
        return self.channel_proj(x_pooled)

    def add_lora(self, model):
        for name, module in model.named_children():
            if isinstance(module, nn.Conv2d):
                lora_module = LoRAConv2d(module)
                setattr(model, name, lora_module)
            else:
                self.add_lora(module)

    def merge_lora(self, model=None):
        if model == None:
            model = self.backbone

        for name, module in model.named_children():
            if isinstance(module, LoRAConv2d):
                module.merge()
                print(f"Merged LoRA weights for layer: {name}")
            else:
                self.merge_lora(module)

    def unmerge_lora(self, model=None):
        if model == None:
            model = self.backbone

        for name, module in model.named_children():
            if isinstance(module, LoRAConv2d):
                module.unmerge()
                print(f"Unmerged LoRA weights for layer: {name}")
            else:
                self.unmerge_lora(module)
