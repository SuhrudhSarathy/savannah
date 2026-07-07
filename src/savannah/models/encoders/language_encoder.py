import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel


class LanguageEncoder(nn.Module):
    """
    Use pretrained CLIP to get clip token outputs.
    If use_eos_only is True, output shape is (B, 1, embed_dim), else (B, T_L, embed_dim) where T_L is the number of tokens
    """

    def __init__(
        self,
        embed_dim: int,
        base_model: str = "openai/clip-vit-base-patch32",
        use_eos_only: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.base_model = base_model
        self.use_eos_only = use_eos_only

        self.model = CLIPTextModel.from_pretrained(self.base_model)
        for param in self.model.parameters():
            param.requires_grad = False

        self.tokeniser = AutoTokenizer.from_pretrained(self.base_model)

        self.clip_dim = (
            self.model.config.hidden_size
            if not use_eos_only
            else self.model.config.projection_dim
        )

        if self.clip_dim != self.embed_dim:
            self.projection = nn.Linear(self.clip_dim, self.embed_dim)
        else:
            self.projection = nn.Identity()

    def forward(self, text_inputs: list[str]):
        # Tokenize inputs
        inputs = self.tokeniser(text_inputs, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)

        if self.use_eos_only:
            # (B, clip_dim) -> (B, 1, clip_dim)
            features = outputs.pooler_output.unsqueeze(1)
        else:
            # (B, T_L, clip_dim)
            features = outputs.last_hidden_state

        # (B, xx, clip_dim) -> (B, xx, embed_dim)
        return self.projection(features)
