import torch
from einops import rearrange
from transformers import AutoModel

from savannah.models.backbones.image_normalizer import ImageNormalizer
from savannah.models.backbones import VisionFeatureExtractor
from savannah.utils.debug import debug_stat


class DinoV3Backbone(VisionFeatureExtractor):
    def __init__(
        self,
        image_size: int,
        base_model: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        use_eos_only: bool = True,
        trainable: bool = True,
    ):
        super().__init__()

        self.base_model = base_model
        self.use_eos_only = use_eos_only

        self.model = AutoModel.from_pretrained(self.base_model)
        self.normalizer = ImageNormalizer(self.base_model)

        self.trainable = trainable
        if not self.trainable:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model_output_dim = self.model.config.hidden_size
        self._out_channels = self.model_output_dim

        if use_eos_only:
            self._tokens_per_image = 1
        else:
            self.patches_per_side = (
                self.model.config.image_size // self.model.config.patch_size
            )
            self._tokens_per_image = (
                self.patches_per_side * self.patches_per_side + 1
            )  # + CLS token

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def tokens_per_image(self) -> int:
        return self._tokens_per_image

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inputs = self.normalizer(x)

        if not self.trainable:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(inputs)
        else:
            outputs = self.model(inputs)

        if self.use_eos_only:
            features = outputs.last_hidden_state[:, 0, ...].unsqueeze(1)
            debug_stat("features", features)
        else:
            patch_features_flat = outputs.last_hidden_state[
                :, 1 + self.model.config.num_register_tokens :, ...
            ]
            cls_token = outputs.last_hidden_state[:, 0, ...].unsqueeze(1)

            patch_features = rearrange(
                patch_features_flat,
                "b (h w) c -> b c h w",
                h=self.patches_per_side,
                w=self.patches_per_side,
            )
            patch_features_flat = rearrange(patch_features, "b c h w -> b (h w) c")

            features = torch.cat([cls_token, patch_features_flat], dim=1)

        return features


if __name__ == "__main__":
    from savannah.utils.device import get_device

    device = get_device()
    backbone = DinoV3Backbone(224, use_eos_only=False).to(device)
    x_img = torch.rand(1, 3, 224, 224).to(device)

    out = backbone(x_img)
    print(out.shape, torch.isnan(out).any())
