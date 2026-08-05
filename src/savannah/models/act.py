import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from savannah.models.policy import Policy
from savannah.models.encoders.vision_encoder import VisionEncoder
from savannah.nn.positional_embeddings import SinusoidalPositionalEncoding
from savannah.nn.transformer import TransformerDecoder, TransformerEncoder
from savannah.utils import ObservationKey
from savannah.utils.policy import PolicyOutput


class ACT_CVAE_Encoder(nn.Module):
    def __init__(
        self,
        state_dim: int,
        encoder_layers: int,
        embed_dim: int,
        num_attn_heads: int,
        feedforward_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # CLS Token
        self.cls_token = nn.Parameter(torch.randn((embed_dim,)))

        # State Embedding
        self.state_embedding = nn.Linear(state_dim, embed_dim)

        # Action Embedding
        self.action_embedding = nn.Linear(state_dim, embed_dim)
        self.action_position_embedding = SinusoidalPositionalEncoding(embed_dim)

        # Transformer Encoder
        self.encoder = TransformerEncoder(
            encoder_layers, embed_dim, num_attn_heads, feedforward_dim, dropout
        )

        # Z Variable downsample
        # 32 is hardcoded in the Paper
        self.z_variable_downsample = nn.Linear(embed_dim, 2 * 32)

    def forward(
        self, state: torch.Tensor, action_sequence: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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


class ACT_CVAE_Decoder(nn.Module):
    def __init__(
        self,
        state_dim: int,
        encoder_layers: int,
        embed_dim: int,
        num_attn_heads: int,
        feedforward_dim: int,
        chunk_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Vision encoder removed — injected at ACTPolicy level

        self.state_embedding = nn.Linear(state_dim, embed_dim)
        self.cls_embedding = nn.Linear(32, embed_dim)

        self.encoder = TransformerEncoder(
            encoder_layers, embed_dim, num_attn_heads, feedforward_dim, dropout
        )

        self.decoder = TransformerDecoder(
            encoder_layers, embed_dim, num_attn_heads, feedforward_dim, dropout
        )
        self.decoder_embedding = nn.Embedding(chunk_size, embed_dim)
        self.decoder_position_embeddings = SinusoidalPositionalEncoding(embed_dim)
        self.action_reproj_layer = nn.Linear(embed_dim, state_dim)
        self.chunk_size = chunk_size

    def forward(
        self,
        x_cam_tokens: torch.Tensor,
        x_state: torch.Tensor,
        x_cls: torch.Tensor,
    ) -> torch.Tensor:
        x_state_embeddings = self.state_embedding(x_state)
        x_cls_embeddings = self.cls_embedding(x_cls)

        x_input_tokens = torch.cat(
            [x_cam_tokens, x_state_embeddings, x_cls_embeddings], dim=1
        )

        x_memory = self.encoder(x_input_tokens)

        x_input = torch.arange(0, self.chunk_size).to(x_state.device)
        x_input_embedding = self.decoder_embedding(x_input)
        x_input_embedding = repeat(
            x_input_embedding, "c n -> b c n", b=x_cam_tokens.shape[0]
        )
        x_input_embedding = self.decoder_position_embeddings(x_input_embedding)

        x_out = self.decoder(x_input_embedding, x_memory)
        return self.action_reproj_layer(x_out)


class ACTPolicy(Policy):
    def __init__(
        self,
        vision_encoder: VisionEncoder,
        cvae_encoder: ACT_CVAE_Encoder,
        cvae_decoder: ACT_CVAE_Decoder,
        num_cameras: int = 1,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.cvae_encoder = cvae_encoder
        self.cvae_decoder = cvae_decoder
        self.num_cameras = num_cameras

    def forward(self, obs: dict[str, torch.Tensor]) -> PolicyOutput:
        images = obs[ObservationKey.images]  # list of (B, 3, H, W)
        x_state = obs[ObservationKey.state]  # (B, 1, state_dim)
        x_target = obs.get(ObservationKey.gt_actions, None)

        # VisionEncoder expects a per-obs history dim: (B, 3, H, W) -> (B, 1, 3, H, W)
        images_with_obs = [img.unsqueeze(1) for img in images if img.dim() == 4]
        x_cam_tokens = self.vision_encoder(images_with_obs)

        if x_target is not None:
            mu, logvar = self.cvae_encoder(x_state, x_target)
            std = torch.exp(0.5 * logvar)
            x_cls = mu + torch.randn_like(mu) * std
        else:
            mu, logvar = None, None
            x_cls = torch.zeros((x_state.shape[0], 1, 32), device=x_state.device)

        actions = self.cvae_decoder(x_cam_tokens, x_state, x_cls)

        return PolicyOutput(
            actions=actions,
            aux={"mu": mu, "logvar": logvar},
        )

    def compute_action(self, obs: dict[str, torch.Tensor]) -> PolicyOutput:
        # At inference, no ground truth actions — CVAE encoder is bypassed,
        # x_cls is zeroed out inside forward when actions key is absent
        with torch.no_grad():
            return self.forward(obs)

    def compute_loss(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
        policy_output: PolicyOutput,
    ) -> torch.Tensor:
        mu = policy_output.aux["mu"]
        logvar = policy_output.aux["logvar"]

        reconstruction_loss = F.l1_loss(policy_output.actions, actions)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction_loss + kl_loss

    @property
    def action_dim(self) -> int:
        return self.cvae_decoder.action_reproj_layer.out_features

    @property
    def action_horizon(self) -> int:
        return self.cvae_decoder.chunk_size
