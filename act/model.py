import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from einops import rearrange, repeat


@dataclass
class ACTConfig:
    encoder_layers: int = 4
    decoder_layers: int = 7
    num_attn_heads: int = 8
    chunk_size: int = 10
    dropout: float = 0.1
    embed_dim: int = 512
    feedforward_dim: int = 3200

    # State dims
    state_dim: int = 14

    # Vision Backbone params
    img_shape: tuple[int, int] = (480, 640)
    vision_backbone_out_channels: int = 728

    @classmethod
    def from_dict(cls, data_dict):
        # Filter out keys that aren't in the dataclass to avoid errors
        return cls(
            **{k: v for k, v in data_dict.items() if k in cls.__dataclass_fields__}
        )


class SelfAttention(nn.Module):
    def __init__(self, config: ACTConfig = ACTConfig()):
        super().__init__()
        self.config = config
        self.embed_dim = self.config.embed_dim
        self.n_heads = self.config.num_attn_heads

        self.attn_projection = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.reproj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x (B, T, n_embed)
        #
        # (B, T, n_embed) -> (B, T, 3 * n_embed)
        attn_projection_x = self.attn_projection(x)

        # (B, T, 3 * n_embed) -> 3 * (B, T, n_embed)
        q, k, v = torch.split(attn_projection_x, self.embed_dim, dim=-1)

        # Multihead
        q = rearrange(
            q,
            "b t (nh hs) -> b nh t hs",
            nh=self.n_heads,
            hs=self.embed_dim // self.n_heads,
        )
        k = rearrange(
            k,
            "b t (nh hs) -> b nh t hs",
            nh=self.n_heads,
            hs=self.embed_dim // self.n_heads,
        )
        v = rearrange(
            v,
            "b t (nh hs) -> b nh t hs",
            nh=self.n_heads,
            hs=self.embed_dim // self.n_heads,
        )

        k_T = rearrange(k, "b nh t hs -> b nh hs t")

        # Attention
        # (b, nh, t, hs) @ (b, nh, hs, t) -> (b, nh, t, t)
        attn = q @ k_T / math.sqrt(k.shape[-1])
        attn = F.softmax(attn, dim=-1)
        # (b, nh, t, t) @ (b, nh, t, hs) -> (b, nh, t, hs)
        attn = attn @ v

        # Rearrrange
        attn = rearrange(
            attn,
            "b nh t hs -> b t (nh hs)",
            nh=self.n_heads,
            hs=self.embed_dim // self.n_heads,
        )

        attn = self.reproj(attn)

        return attn


class CrossAttention(nn.Module):
    def __init__(self, config: ACTConfig = ACTConfig()):
        super().__init__()
        self.config = config
        self.embed_dim = self.config.embed_dim
        self.n_heads = self.config.num_attn_heads

        self.q_projection = nn.Linear(self.embed_dim, self.embed_dim)
        self.kv_projection = nn.Linear(self.embed_dim, 2 * self.embed_dim)

        self.reproj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # x (B, T, n_embed), kv (B, T, n_embed)
        q = self.q_projection(x)

        # (B, T, n_embed) -> (B, T, 2 * n_embed)
        kv_projection = self.kv_projection(kv)

        # (B, T, 2* n_embed) -> 2 * (B, T, n_embed)
        k, v = torch.split(kv_projection, self.embed_dim, dim=-1)

        # Multihead
        q = rearrange(
            q,
            "b t (nh hs) -> b nh t hs",
            nh=self.n_heads,
            hs=self.embed_dim // self.n_heads,
        )
        k = rearrange(
            k,
            "b t (nh hs) -> b nh t hs",
            nh=self.n_heads,
            hs=self.embed_dim // self.n_heads,
        )
        v = rearrange(
            v,
            "b t (nh hs) -> b nh t hs",
            nh=self.n_heads,
            hs=self.embed_dim // self.n_heads,
        )

        k_T = rearrange(k, "b nh t hs -> b nh hs t")

        # Attention
        # (b, nh, t, hs) @ (b, nh, hs, t) -> (b, nh, t, t)
        attn = q @ k_T / math.sqrt(k.shape[-1])
        attn = F.softmax(attn, dim=-1)

        # (b, nh, t, t) @ (b, nh, t, hs) -> (b, nh, t, hs)
        attn = attn @ v

        # Rearrrange
        attn = rearrange(
            attn,
            "b nh t hs -> b t (nh hs)",
            nh=self.n_heads,
            hs=self.embed_dim // self.n_heads,
        )

        attn = self.reproj(attn)

        return attn


class TransformerEncoderBlock(nn.Module):
    def __init__(self, config: ACTConfig = ACTConfig()):
        super().__init__()
        self.config = config

        self.attn_block = SelfAttention(self.config)

        self.layer_norm1 = nn.LayerNorm(self.config.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.config.embed_dim)

        self.ffn_block = nn.Sequential(
            nn.Linear(self.config.embed_dim, self.config.feedforward_dim),
            nn.GELU(),
            nn.Linear(self.config.feedforward_dim, self.config.embed_dim),
        )

        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use Pre norm and then Attn
        x_norm = self.layer_norm1(x)
        x_attn = self.attn_block(x_norm)
        x = x + x_attn

        # Use pre norm and then FFN
        x_norm2 = self.layer_norm2(x)
        x_ffn = self.ffn_block(x_norm2)
        x = x + x_ffn

        x = self.dropout(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, config: ACTConfig = ACTConfig()):
        super().__init__()
        self.config = config

        self.encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(config=self.config)
                for _ in range(self.config.encoder_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.encoder_blocks:
            x = block(x)

        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, config: ACTConfig = ACTConfig()):
        super().__init__()
        self.config = config

        self.self_attn_block = SelfAttention(self.config)
        self.cross_attn_block = CrossAttention(self.config)

        self.layer_norm1 = nn.LayerNorm(self.config.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.config.embed_dim)
        self.layer_norm_kv = nn.LayerNorm(self.config.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.config.embed_dim)

        self.ffn_block = nn.Sequential(
            nn.Linear(self.config.embed_dim, self.config.feedforward_dim),
            nn.GELU(),
            nn.Linear(self.config.feedforward_dim, self.config.embed_dim),
        )

        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # Pre Norm + Self Attn of the input embeddings
        x_norm1 = self.layer_norm1(x)
        x_self_attn = self.self_attn_block(x_norm1)

        x = x + x_self_attn

        # Pre Norm + Cross Attn of input and kv embeddings
        x_norm2 = self.layer_norm2(x)
        kv_norm = self.layer_norm_kv(kv)

        x_cross_attn = self.cross_attn_block(x_norm2, kv_norm)
        x = x + x_cross_attn

        # Pre Norm + FFN Block
        x_norm3 = self.layer_norm3(x)
        x_ffn = self.ffn_block(x_norm3)
        x = x + x_ffn

        x = self.dropout(x)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, config: ACTConfig = ACTConfig()):
        super().__init__()
        self.config = config

        self.decoder_blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(self.config)
                for _ in range(self.config.decoder_layers)
            ]
        )

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        for block in self.decoder_blocks:
            x = block(x, kv)

        return x


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x (B, T, n_embed)
        x = x + self.pe[:, : x.size(1), :]
        return x

class SinusoidalPositionalEncoding2D(nn.Module):
    def __init__(self, channels: int):
        """
        Args:
            channels: The total embedding dimension (must be divisible by 4).
        """
        super().__init__()
        self.channels = channels
        # Half for height (y), half for width (x)
        self.dim = channels // 2 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        
        y_coords = torch.arange(height, device=x.device).unsqueeze(1).repeat(1, width)
        x_coords = torch.arange(width, device=x.device).unsqueeze(0).repeat(height, 1)

        div_term = torch.exp(
            torch.arange(0, self.dim, 2, device=x.device) * -(math.log(10000.0) / self.dim)
        )

        pos_y = y_coords.unsqueeze(-1) * div_term # (H, W, dim/2)
        pe_y = torch.zeros(height, width, self.dim, device=x.device)
        pe_y[:, :, 0::2] = torch.sin(pos_y)
        pe_y[:, :, 1::2] = torch.cos(pos_y)

        pos_x = x_coords.unsqueeze(-1) * div_term # (H, W, dim/2)
        pe_x = torch.zeros(height, width, self.dim, device=x.device)
        pe_x[:, :, 0::2] = torch.sin(pos_x)
        pe_x[:, :, 1::2] = torch.cos(pos_x)

        pe = torch.cat([pe_y, pe_x], dim=-1).permute(2, 0, 1)
        
        # Add to input with broadcast across batch
        return x + pe.unsqueeze(0)

class ACT_CVAE_Encoder(nn.Module):
    def __init__(self, config: ACTConfig = ACTConfig()):
        super().__init__()
        self.config = config

        # CLS Token
        self.cls_token = nn.Parameter(torch.randn((self.config.embed_dim,)))

        # State Embedding
        self.state_embedding = nn.Linear(self.config.state_dim, self.config.embed_dim)

        # Action Embedding
        self.action_embedding = nn.Linear(self.config.state_dim, self.config.embed_dim)
        self.action_position_embedding = SinusoidalPositionalEncoding(
            self.config.embed_dim
        )

        # Transformer Encoder
        self.encoder = TransformerEncoder(self.config)

        # Z Variable downsample
        # 32 is hardcoded in the Paper
        self.z_variable_downsample = nn.Linear(self.config.embed_dim, 2 * 32)

    def forward(
        self, state: torch.Tensor, action_sequence: torch.Tensor
    ) -> torch.Tensor:
        # (B, 1, state_dim) -> (B, 1, n_embed)
        x_state_embed = self.state_embedding(state)

        # (B, k, state_dim) -> (B, k, n_embed)
        x_action_embed = self.action_embedding(action_sequence)
        x_action_embed = self.action_position_embedding(x_action_embed)

        # cls (state_dim) -> (B, 1, state_dim)
        x_cls = repeat(self.cls_token, "d -> b 1 d", b=x_state_embed.shape[0])

        x = torch.cat([x_cls, x_state_embed, x_action_embed], dim=1)

        x_out = self.encoder(x)

        # Get the first token corresponding to CLS
        cls_feature = x_out[:, 0, :].unsqueeze(1)

        # Get the Mean and Std Dev for reparametrization
        z_mean_std = self.z_variable_downsample(cls_feature)
        mean, logvar = torch.split(z_mean_std, 32, dim=-1)

        return mean, logvar


# class ResNetBackbone(nn.Module):
#     def __init__(self, out_channels: int):
#         super().__init__()
#         resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

#         self.backbone = nn.Sequential(*list(resnet.children())[:-2])
#         self.proj = nn.Identity()

#         if out_channels != 512:
#             self.proj = nn.Conv2d(512, out_channels, kernel_size=3, padding=1)

#     def forward(self, x):
#         features = self.backbone(x)
#         out = self.proj(features)
#         return out


class ResNetBackbone(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        # Load pre-trained weights
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # 1. Replace the initial 7x7 conv (stride 2) with a 3x3 conv (stride 1)
        # This prevents the initial massive loss of resolution
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # 2. Remove the MaxPool layer
        # This keeps the feature map larger for longer
        resnet.maxpool = nn.Identity()

        # 3. Keep the main layers (layer1 through layer4)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # 4. Final projection to your Transformer's embed_dim
        self.proj = nn.Conv2d(512, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, 3, 96, 96)
        features = self.backbone(x)
        # With the fix: features will be (B, 512, 12, 12)
        # (Standard ResNet would have given you 3x3)
        return self.proj(features)


class VisionBackbone(nn.Module):
    def __init__(self, config: ACTConfig = ACTConfig()):
        super().__init__()
        self.config = config

        self.resnet_back_bone = ResNetBackbone(self.config.vision_backbone_out_channels)

        self.spatial_proj = nn.Conv2d(
            self.config.vision_backbone_out_channels, 
            self.config.embed_dim, 
            kernel_size=1
        )

        self.positional_embeddings = SinusoidalPositionalEncoding2D(self.config.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.resnet_back_bone(x)

        features = self.spatial_proj(features)
        
        features = self.positional_embeddings(features)
        
        features_out = rearrange(features, "b c h w -> b (h w) c")

        return features_out


class ACT_CVAE_Decoder(nn.Module):
    def __init__(self, config: ACTConfig = ACTConfig()):
        super().__init__()
        self.config = config

        # TODO: Extend this to multiple vision inputs
        self.vision_backbone = VisionBackbone(config)

        # State Embeddings
        self.state_embedding = nn.Linear(self.config.state_dim, self.config.embed_dim)

        # CLS Embeddings
        self.cls_embedding = nn.Linear(32, self.config.embed_dim)

        # Encoder
        self.encoder = TransformerEncoder(config)

        # Decoder
        self.decoder = TransformerDecoder(config)
        self.decoder_embedding = nn.Embedding(
            self.config.chunk_size, self.config.embed_dim
        )
        self.decoder_position_embeddings = SinusoidalPositionalEncoding(
            self.config.embed_dim
        )
        self.action_repoj_layer = nn.Linear(
            self.config.embed_dim, self.config.state_dim
        )

    def forward(
        self, x_cam: torch.Tensor, x_state: torch.Tensor, x_cls: torch.Tensor
    ) -> torch.Tensor:
        # x_cam (B, 3, H, W) -> (B, 300, n_embed)
        x_cam_embeddings = self.vision_backbone(x_cam)

        # x_state (B, 1, state_dim) -> (B, 1, n_embed)
        x_state_embeddings = self.state_embedding(x_state)

        # x_cls (B, 1, 32) -> (B, 1, n_embed)
        x_cls_embeddings = self.cls_embedding(x_cls)

        # x_input_tokens = [x_cam, x_state, x_cls] embeddings
        x_input_tokens = torch.cat(
            [x_cam_embeddings, x_state_embeddings, x_cls_embeddings], dim=1
        )

        # (B, 302, n_embed)
        x_memory = self.encoder(x_input_tokens)

        # (chunk_size,)
        x_input = torch.arange(0, self.config.chunk_size).to(x_cam.device)

        # (chunk_size, n_embed)
        x_input_embedding = self.decoder_embedding(x_input)
        # (B, chunk_size, n_embed)
        x_input_embedding = repeat(
            x_input_embedding, "c n -> b c n", b=x_cam_embeddings.shape[0]
        )
        # (B, chunk_size, n_embed)
        x_input_embedding = self.decoder_position_embeddings(x_input_embedding)

        # x_input_embedding (B, chunk_size, n_embed), x_memory (B, 302, n_embed)
        x_out = self.decoder(x_input_embedding, x_memory)

        # Repojection
        # (B, chunk_size, n_embed) -> (B, chunk_size, state_dim)
        actions = self.action_repoj_layer(x_out)

        return actions


class ACTPolicy(nn.Module):
    def __init__(self, config: ACTConfig = ACTConfig()):
        super().__init__()
        self.config = config
        self.cvae_encoder = ACT_CVAE_Encoder(config)
        self.cvae_decoder = ACT_CVAE_Decoder(config)

    def _num_params(self):
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return num_params / 1000000

    def forward(
        self,
        x_image: torch.Tensor,
        x_state: torch.Tensor,
        x_target: torch.Tensor | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if x_target is not None:
            mu, logvar = self.cvae_encoder(x_state, x_target)
            std = torch.exp(0.5 * logvar)

            x_cls = mu + torch.randn_like(mu) * std
        else:
            mu, std = None, None
            x_cls = torch.zeros((x_state.shape[0], 1, 32)).to(x_state.device)

        x_out = self.cvae_decoder(x_image, x_state, x_cls)

        return x_out, [mu, std]


if __name__ == "__main__":
    device = torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
    config = ACTConfig()
    config.chunk_size = 50
    config.state_dim = 2

    B = 8

    x_image = torch.randn((B, 3, *config.img_shape)).to(device)
    x_state = torch.randn((B, 1, config.state_dim)).to(device)

    x_target = torch.randn((B, config.chunk_size, config.state_dim)).to(device)

    act = ACTPolicy(config).to(device)
    print(f"Num of params: {act._num_params()}M")

    x_out, _ = act(x_image, x_state, x_target)
    print(x_out.shape)
