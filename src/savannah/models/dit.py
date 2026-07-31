import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from savannah.models.encoders import StateEncoder, VisionEncoder
from savannah.models.encoders.language_encoder import LanguageEncoder
from savannah.models.policy import Policy
from savannah.nn.positional_embeddings import get_sinusoidal_position_embedding
from savannah.nn.self_attention import SelfAttention
from savannah.nn.time_embedding import TimeEmbedding
from savannah.objectives import PolicyObjective
from savannah.utils.debug import debug_stat
from savannah.utils.log import logger
from savannah.utils.observation import ObservationKey
from savannah.utils.policy import PolicyOutput


class MultiChoiceAttention(nn.Module):
    def __init__(self, num_encoders: int):
        super().__init__()
        weights = torch.zeros(num_encoders)
        weights[-1] = 10.0
        self.weight_matrix = nn.Parameter(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_weights = F.softmax(self.weight_matrix, dim=0)
        x_out = torch.einsum("bsnd, d -> bsn", x, attn_weights)

        return x_out


class DiTEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_attn_heads: int,
        feedforward_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_attn_heads = num_attn_heads
        self.feedforward_dim = feedforward_dim
        self.dropout = dropout

        self.attn_block = SelfAttention(embed_dim, num_attn_heads)
        self.ffn_block = nn.Sequential(
            nn.Linear(self.embed_dim, self.feedforward_dim),
            nn.GELU(),
            nn.Linear(self.feedforward_dim, self.embed_dim),
        )

        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.layer_norm1(x)
        x_attn = self.attn_block(x_norm)

        # Residual Connection
        x = x + x_attn

        # FFN Block
        x_norm = self.layer_norm2(x)
        x_ffn = self.ffn_block(x_norm)

        # Residual Connection
        x = x + x_ffn

        # Dropout
        x = self.dropout_layer(x)
        return x


class DiTDecoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_attn_heads: int,
        feedforward_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_attn_heads = num_attn_heads
        self.feedforward_dim = feedforward_dim
        self.dropout = dropout

        self.attn_block = SelfAttention(embed_dim, num_attn_heads)

        self.ffn_block = nn.Sequential(
            nn.Linear(self.embed_dim, self.feedforward_dim),
            nn.GELU(),
            nn.Linear(self.feedforward_dim, self.embed_dim),
        )

        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

        self.dropout_layer = nn.Dropout(self.dropout)

        self.adaln_block = nn.Linear(self.embed_dim, 6 * self.embed_dim)
        self._init_adaln_zero()

    def _init_adaln_zero(self):
        """Initializes the alpha scaling projections to 0 so the network

        initially behaves like an identity skip connection.
        """
        nn.init.constant_(self.adaln_block.weight, 0.0)
        nn.init.constant_(self.adaln_block.bias, 0.0)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, x_time: torch.Tensor
    ) -> torch.Tensor:
        # x_time: tokens matching to timestep of denoising
        # x: is the tokens that are input to the decoder and the condition tokens

        # AdaLN stuff
        beta1, gamma1, alpha1, beta2, gamma2, alpha2 = torch.split(
            self.adaln_block(x_time),
            self.embed_dim,
            dim=-1,
        )

        # Masked-Self Attention Block
        x_norm = self.layer_norm1(x)
        x_pre_attn = gamma1 * x_norm + beta1
        x_attn = self.attn_block(x_pre_attn, mask)
        x_post_attn = alpha1 * x_attn

        # Residual Connection
        x = x + x_post_attn

        # FFN Block
        x_norm = self.layer_norm2(x)
        x_pre_ffn = gamma2 * x_norm + beta2
        x_ffn = self.ffn_block(x_pre_ffn)
        x_post_ffn = alpha2 * x_ffn

        # Residual Connection
        x = x + x_post_ffn

        # Dropout
        x = self.dropout_layer(x)

        return x


class DiTPolicy(Policy):
    def __init__(
        self,
        embed_dim: int,
        encoder_num_blocks: int,
        encoder_num_attn_heads: int,
        encoder_feedforward_dim: int,
        encoder_dropout: float,
        decoder_num_blocks: int,
        decoder_num_attn_heads: int,
        decoder_feedforward_dim: int,
        decoder_dropout: float,
        state_dim: int,
        action_dim: int,
        action_horizon: int,
        num_cameras: int,
        vision_encoder: VisionEncoder,
        language_encoder: LanguageEncoder | None,
        objective: PolicyObjective,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._action_horizon = action_horizon

        self.vision_encoder = vision_encoder
        self.num_cameras = num_cameras
        self.objective = objective

        self.encoder = nn.ModuleList(
            [
                DiTEncoderBlock(
                    embed_dim,
                    encoder_num_attn_heads,
                    encoder_feedforward_dim,
                    encoder_dropout,
                )
                for _ in range(encoder_num_blocks)
            ]
        )

        self.decoder = nn.ModuleList(
            [
                DiTDecoderBlock(
                    embed_dim,
                    decoder_num_attn_heads,
                    decoder_feedforward_dim,
                    decoder_dropout,
                )
                for _ in range(decoder_num_blocks)
            ]
        )

        self.time_embedding = nn.Sequential(
            TimeEmbedding(embed_dim),
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

        self.camera_embedding = nn.Embedding(num_cameras, embed_dim)

        self.state_encoder = StateEncoder(self._state_dim, self.embed_dim)
        self.language_encoder = language_encoder
        if language_encoder is not None:
            logger.debug("Initialised Language Encoder into the model")

        sinusoidal_position_embeddings = get_sinusoidal_position_embedding(
            self._action_horizon, self.embed_dim
        )
        self.register_buffer("action_time_embedding", sinusoidal_position_embeddings)

        # Action Embedding
        self.action_embedding = nn.Sequential(
            nn.Linear(self._action_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        # MultiChoice Attention computation
        self.multi_choice_attn = MultiChoiceAttention(encoder_num_blocks)
        self.mask = None

        # Action reprojection
        self.action_reprojection = nn.Linear(self.embed_dim, self._action_dim)

    def forward(self, obs: dict[str, torch.Tensor], *args, **kwargs) -> PolicyOutput:
        x_img = obs[ObservationKey.images]
        x_state = obs[ObservationKey.state]
        x_time = obs[ObservationKey.time]
        x_language = obs.get(ObservationKey.language, None)

        for cam_idx, img in enumerate(x_img):
            debug_stat(f"x_img[{cam_idx}] (raw)", img)
        debug_stat("x_state (raw)", x_state)
        debug_stat("x_time (raw)", x_time)
        if x_language is not None:
            logger.debug("x_language (raw) {}", x_language)

        # Get noisy actions from kwargs
        noisy_actions = kwargs.get("noisy_actions", None)
        if noisy_actions is None:
            raise AssertionError("Pass noisy action for the model to run")
        debug_stat("noisy_actions (raw)", noisy_actions)

        # (B, T, embed_dim)
        x_cam_tokens = self._encode_cameras(x_img)
        debug_stat("x_cam_tokens", x_cam_tokens)

        # (B, N, state_dim) -> (B, N, embedding_dim)
        x_state = self.state_encoder(x_state)
        debug_stat("x_state (embedded)", x_state)

        if x_language is not None:
            if self.language_encoder is None:
                with warnings.catch_warnings():
                    warnings.simplefilter("once")
                    warnings.warn(
                        "Language instruction is provided but LanguageEncoder is not specified. Not using language instruction",
                        RuntimeWarning,
                    )
            else:
                # (B, xx, embed_dim)
                x_language = self.language_encoder(x_language)
                debug_stat("x_lang", x_language)

        # (B,) -> (B, 1) optionally
        if len(x_time.shape) == 1:
            x_time = x_time.unsqueeze(1)

        # (B, 1) -> (B, 1, embed_dim)
        x_time = self.time_embedding(x_time).unsqueeze(1)
        debug_stat("x_time (embedded)", x_time)

        # Project the noisy actions to embeding space
        # (B, N_obs, action_dim) -> (B, N_obs, embed_dim)
        x_noisy_actions = self.action_embedding(noisy_actions)
        debug_stat("x_noisy_actions (embedded)", x_noisy_actions)

        x_noisy_actions = x_noisy_actions + self.action_time_embedding
        debug_stat("x_noisy_actions (+ pos embed)", x_noisy_actions)

        if x_language is None:
            x_cond_tokens = torch.cat([x_cam_tokens, x_state], dim=1)
        else:
            x_cond_tokens = torch.cat([x_language, x_cam_tokens, x_state], dim=1)

        debug_stat("x_cond", x_cond_tokens)

        # Pass the condition tokens through the encoders
        x_cond_tokens_pooled = []
        x_cond = x_cond_tokens
        for enc in self.encoder:
            x_cond = enc(x_cond)
            x_cond_tokens_pooled.append(x_cond.unsqueeze(-1))

        # Get the condition tokens from MultiChoice attention
        # [(B, n_seq, embed_dim, 1)] -> (B, n_seq, embed_dim,  n_enc)
        x_cond_tokens_pooled = torch.cat(x_cond_tokens_pooled, dim=-1)
        # (B, n_seq, embed_dim, n_enc) -> (B, n_seq, embed_dim)
        x_cond_tokens_pooled = self.multi_choice_attn(x_cond_tokens_pooled)
        debug_stat("x_cond_tokens_pooled", x_cond_tokens_pooled)

        # concat cond tokens with noisy state
        x_dec_tokens = torch.cat([x_cond_tokens_pooled, x_noisy_actions], dim=1)
        debug_stat("x_dec_tokens", x_dec_tokens)

        # Create the mask for x_dec_tokens to act on
        # Basically x_dec_tokens is [x_cond_tokens, x_noisy_actions]
        n_cond = x_cond_tokens.shape[1]
        self.mask = torch.full(
            (x_dec_tokens.shape[1], x_dec_tokens.shape[1]),
            False,
            dtype=torch.bool,
            device=x_dec_tokens.device,
        ).to(x_state.device)
        self.mask[:n_cond, :n_cond] = True
        self.mask[n_cond:, :n_cond] = True
        self.mask[n_cond:, n_cond:] = True

        x_dec_out = x_dec_tokens
        for dec in self.decoder:
            x_dec_out = dec(x_dec_out, self.mask, x_time)
        debug_stat("x_dec_out", x_dec_out)

        x_action_tokens = x_dec_out[:, n_cond:, ...]
        debug_stat("x_action_tokens", x_action_tokens)

        x_action = self.action_reprojection(x_action_tokens)
        debug_stat("x_action", x_action)
        return PolicyOutput(actions=x_action)
