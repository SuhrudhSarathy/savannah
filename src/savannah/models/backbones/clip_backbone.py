import torch
import torch.nn as nn
from transformers import AutoProcessor, CLIPVisionModel

from savannah.models.backbones import VisionFeatureExtractor
from savannah.models.backbones.token_learner import TokenLearner
from savannah.nn.positional_embeddings import SinusoidalPositionalEncoding
from savannah.utils.debug import debug_stat
from savannah.utils.log import logger


class CLIPBackbone(VisionFeatureExtractor):
    def __init__(
        self,
        image_size: int,
        base_model: str = "openai/clip-vit-base-patch32",
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

        self.model = CLIPVisionModel.from_pretrained(self.base_model)
        self.processor = AutoProcessor.from_pretrained(self.base_model)

        self.trainable = trainable
        if not self.trainable:
            for param in self.model.parameters():
                param.requires_grad = False

        # CLIPVisionModel (unlike CLIPModel) has no projection head, so both
        # pooler_output and last_hidden_state are sized by hidden_size.
        self.clip_dim = self.model.config.hidden_size
        self._out_channels = self.clip_dim

        if use_eos_only:
            self._tokens_per_image = 1
        else:
            # The processor resizes any input to the model's configured
            # image_size, so the patch grid is fixed regardless of the
            # image_size passed to this constructor.
            patches_per_side = (
                self.model.config.image_size // self.model.config.patch_size
            )
            self._tokens_per_image = (
                patches_per_side * patches_per_side + 1
            )  # + CLS token

            self.sinusoidal_position_encoding = SinusoidalPositionalEncoding(
                self.clip_dim
            )

            if reduce:
                self.token_learner = TokenLearner(self.clip_dim, reduced_tokens, 8)
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
            features = outputs.pooler_output.unsqueeze(1)
            debug_stat("features", features)
        else:
            features = outputs.last_hidden_state  # (B, n_t, output_dim)
            features = self.sinusoidal_position_encoding(features)
            if self.reduce:
                features = self.token_learner(features)

        return features


if __name__ == "__main__":
    from savannah.utils.device import get_device

    device = get_device()
    backbone = CLIPBackbone(226).to(device)
    x_img = torch.rand(1, 3, 226, 226).to(device)

    out = backbone(x_img)
    print(torch.isnan(out).any())
