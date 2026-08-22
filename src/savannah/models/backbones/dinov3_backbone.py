import torch
from transformers import AutoModel, AutoProcessor

from savannah.models.backbones import VisionFeatureExtractor
from savannah.models.backbones.token_learner import TokenLearner
from savannah.nn.positional_embeddings import SinusoidalPositionalEncoding
from savannah.utils.debug import debug_stat


class DinoV3Backbone(VisionFeatureExtractor):
    def __init__(
        self,
        image_size: int,
        base_model: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        use_eos_only: bool = True,
        trainable: bool = True,
        reduce: bool = False,
        reduced_tokens: int = 4,
    ):
        super().__init__()

        self.base_model = base_model
        self.use_eos_only = use_eos_only
        self.reduce = reduce
        self.reduced_tokens = reduced_tokens

        self.model = AutoModel.from_pretrained(self.base_model)
        self.processor = AutoProcessor.from_pretrained(self.base_model)

        self.trainable = trainable
        if not self.trainable:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model_output_dim = self.model.config.hidden_size
        self._out_channels = self.model_output_dim

        if use_eos_only:
            self._tokens_per_image = 1
        else:
            patches_per_side = (
                self.model.config.image_size // self.model.config.patch_size
            )
            self._tokens_per_image = (
                patches_per_side * patches_per_side + 1
            )  # + CLS token

            self.sinusoidal_position_encoding = SinusoidalPositionalEncoding(
                self.model_output_dim
            )

            if reduce:
                self.token_learner = TokenLearner(
                    self.model_output_dim, reduced_tokens, 8
                )
                self._tokens_per_image = reduced_tokens

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def tokens_per_image(self) -> int:
        return self._tokens_per_image

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inputs = self.processor(x, return_tensors="pt", do_rescale=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        if not self.trainable:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**inputs)
        else:
            outputs = self.model(**inputs)

        if self.use_eos_only:
            features = outputs.last_hidden_state[:, 0, ...].unsqueeze(1)
            debug_stat("features", features)
        else:
            patch_features_flat = outputs.last_hidden_state[
                :, 1 + self.model.config.num_register_tokens :, ...
            ]
            cls_token = outputs.last_hidden_state[:, 0, ...].unsqueeze(1)
            features = torch.cat([cls_token, patch_features_flat], dim=1)
            features = self.sinusoidal_position_encoding(features)
            if self.reduce:
                features = self.token_learner(features)

        return features


if __name__ == "__main__":
    from savannah.utils.device import get_device

    device = get_device()
    backbone = DinoV3Backbone(224, use_eos_only=False).to(device)
    x_img = torch.rand(1, 3, 224, 224).to(device)

    out = backbone(x_img)
    print(torch.isnan(out).any())
