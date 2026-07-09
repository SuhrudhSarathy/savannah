from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from savannah.models.policy import Policy
from savannah.models.encoders import VisionEncoder, LanguageEncoder, StateEncoder
from savannah.nn.self_attention import SelfAttention
from savannah.nn.time_embedding import TimeEmbedding
from savannah.objectives import PolicyObjective
from savannah.utils.log import logger
from savannah.utils.observation import ObservationKey
from savannah.utils.policy import PolicyOutput
from savannah.nn.positional_embeddings import get_sinusoidal_position_embedding
import warnings


class EncoderBlock(nn.Module):
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

        self.attn_block = SelfAttention(self.embed_dim, self.num_attn_heads)

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
        x = x + x_attn

        x_norm = self.layer_norm2(x)
        x_ffn = self.ffn_block(x_norm)
        x = x + x_ffn

        x = self.dropout_layer(x)

        return x


class DecoderBlock(nn.Module):
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

        self.adaln_block = nn.Linear(2 * self.embed_dim, 6 * self.embed_dim)
        self._init_adaln_zero()

    def _init_adaln_zero(self):
        """Initializes the alpha scaling projections to 0 so the network

        initially behaves like an identity skip connection.
        """
        nn.init.constant_(self.adaln_block.weight, 0.0)
        nn.init.constant_(self.adaln_block.bias, 0.0)

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # KV is essentially mean of tokens from cat of encoder and Timestep
        # x: is the tokens that are input to the decoder

        # AdaLN stuff
        beta1, gamma1, alpha1, beta2, gamma2, alpha2 = torch.split(
            self.adaln_block(kv),
            self.embed_dim,
            dim=-1,
        )

        # Self Attention Block
        x_norm = self.layer_norm1(x)
        x_pre_attn = gamma1 * x_norm + beta1
        x_attn = self.attn_block(x_pre_attn)
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


class DiTAdaLNPolicy(Policy):
    def __init__(
        self,
        embed_dim: int,
        num_blocks: int,
        encoder_num_attn_heads: int,
        encoder_feedforward_dim: int,
        encoder_dropout: float,
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

        self.encoders = nn.ModuleList(
            [
                EncoderBlock(
                    embed_dim,
                    encoder_num_attn_heads,
                    encoder_feedforward_dim,
                    encoder_dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        self.decoders = nn.ModuleList(
            [
                DecoderBlock(
                    embed_dim,
                    decoder_num_attn_heads,
                    decoder_feedforward_dim,
                    decoder_dropout,
                )
                for _ in range(num_blocks)
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
        if language_encoder is not None:
            self.language_encoder = language_encoder
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

        # Action reprojection
        self.action_reprojection = nn.Linear(self.embed_dim, self._action_dim)

    @staticmethod
    def _debug_stat(name: str, t: torch.Tensor) -> None:
        logger.debug(
            "{}: shape={}, min={:.6f}, max={:.6f}, mean={:.6f}, nan={}, inf={}",
            name,
            tuple(t.shape),
            t.min().item(),
            t.max().item(),
            t.mean().item(),
            torch.isnan(t).any().item(),
            torch.isinf(t).any().item(),
        )

    def forward(self, obs: dict[str, torch.Tensor], *args, **kwargs) -> PolicyOutput:
        x_img = obs[ObservationKey.images]
        x_state = obs[ObservationKey.state]
        x_time = obs[ObservationKey.time]
        x_language = obs.get(ObservationKey.language, None)

        for cam_idx, img in enumerate(x_img):
            self._debug_stat(f"x_img[{cam_idx}] (raw)", img)
        self._debug_stat("x_state (raw)", x_state)
        self._debug_stat("x_time (raw)", x_time)
        if x_language is not None:
            logger.debug("x_language (raw) {}", x_language)

        # Get noisy actions from kwargs
        noisy_actions = kwargs.get("noisy_actions", None)
        if noisy_actions is None:
            raise AssertionError("Pass noisy action for the model to run")
        self._debug_stat("noisy_actions (raw)", noisy_actions)

        # (B, T, embed_dim)
        x_cam_tokens = self._encode_cameras(x_img)
        self._debug_stat("x_cam_tokens", x_cam_tokens)

        # (B, N, state_dim) -> (B, N, embedding_dim)
        x_state = self.state_encoder(x_state)
        self._debug_stat("x_state (embedded)", x_state)

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
                self._debug_stat("x_lang", x_language)

        # (B,) -> (B, 1) optionally
        if len(x_time.shape) == 1:
            x_time = x_time.unsqueeze(1)

        # (B, 1) -> (B, 1, embed_dim)
        x_time = self.time_embedding(x_time).unsqueeze(1)
        self._debug_stat("x_time (embedded)", x_time)

        # Project the noisy actions to embeding space
        # (B, N_obs, action_dim) -> (B, N_obs, embed_dim)
        x_noisy_actions = self.action_embedding(noisy_actions)
        self._debug_stat("x_noisy_actions (embedded)", x_noisy_actions)

        x_noisy_actions = x_noisy_actions + self.action_time_embedding
        self._debug_stat("x_noisy_actions (+ pos embed)", x_noisy_actions)

        if x_language is None:
            x_cond = torch.cat([x_cam_tokens, x_state], dim=1)
        else:
            x_cond = torch.cat([x_language, x_cam_tokens, x_state], dim=1)

        self._debug_stat("x_cond", x_cond)

        encoder_layer_outputs = []
        x_enc_out = x_cond
        for i, enc in enumerate(self.encoders):
            x_enc_out = enc(x_enc_out)
            self._debug_stat(f"encoder[{i}] out", x_enc_out)
            encoder_layer_outputs.append(x_enc_out)

        x_out = x_noisy_actions
        for i, dec in enumerate(self.decoders):
            # (B, TC, embed_dim) -> (B, 1, embed_dim)
            x_enc_out = encoder_layer_outputs[i]
            x_enc_mean = torch.mean(x_enc_out, dim=1, keepdim=True)
            self._debug_stat(f"decoder[{i}] x_enc_mean", x_enc_mean)

            x_kv = torch.cat([x_enc_mean, x_time], dim=-1)
            self._debug_stat(f"decoder[{i}] x_kv", x_kv)

            x_out = dec(x_out, x_kv)
            self._debug_stat(f"decoder[{i}] out", x_out)

        x_out_action = self.action_reprojection(x_out)
        self._debug_stat("x_out_action (final)", x_out_action)

        output = PolicyOutput(actions=x_out_action)

        return output
