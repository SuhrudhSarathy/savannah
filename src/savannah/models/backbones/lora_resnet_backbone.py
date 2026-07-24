import math

import torch
import torch.nn as nn
import torchvision.models as models

from savannah.models.backbones import VisionFeatureExtractor
from savannah.models.backbones.resnet_backbone import SpatialSoftmax
from einops import rearrange


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
    def __init__(
        self,
        out_channels: int,
        image_size: int = 128,
        lora_rank: int = 20,
        lora_alpha: int = 40,
        use_spatial_softmax: bool = False,
    ):
        super().__init__()
        self._out_channels = out_channels
        self.use_spatial_softmax = use_spatial_softmax

        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        self.backbone = nn.Sequential(*list(self.resnet.children())[:-2])
        self.replace_bn_with_gn(self.backbone)
        self.add_lora(self.backbone)

        if use_spatial_softmax:
            self.spatial_softmax = SpatialSoftmax()
            self._tokens_per_image = 1
            proj_in = 2 * 512
        else:
            # ResNet18 downsamples spatially by 32x (5 pooling/stride-2 ops).
            # tokens_per_image = (H // 32) * (W // 32) for a square input.
            spatial = image_size // 32
            self._tokens_per_image = spatial * spatial
            proj_in = 512

        if out_channels != proj_in:
            self.channel_proj = nn.Linear(proj_in, out_channels)
        else:
            self.channel_proj = nn.Identity()

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def tokens_per_image(self) -> int:
        return self._tokens_per_image

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = self.backbone(x)  # (B, C, H, W)
        if self.use_spatial_softmax:
            x_out = self.spatial_softmax(x_out)  # (B, 2*C)
        else:
            x_out = rearrange(x_out, "b c h w -> b (h w) c")
        return self.channel_proj(x_out)

    def add_lora(self, model):
        for name, module in model.named_children():
            if isinstance(module, nn.Conv2d):
                lora_module = LoRAConv2d(module, self.lora_rank, self.lora_alpha)
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
