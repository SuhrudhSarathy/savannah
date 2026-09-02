import torch
import torch.nn as nn
from einops import rearrange

from savannah.models.encoders import StateEncoder, VisionEncoder
from savannah.models.encoders.language_encoder import LanguageEncoder
from savannah.models.policy import Policy
from savannah.nn.cross_attention import CrossAttention
from savannah.nn.positional_embeddings import get_sinusoidal_position_embedding
from savannah.nn.rope import RoPESelfAttention
from savannah.nn.self_attention import SelfAttention
from savannah.nn.time_embedding import TimeEmbedding
from savannah.objectives import PolicyObjective
from savannah.utils.debug import debug_stat
from savannah.utils.log import logger
from savannah.utils.observation import ObservationKey
from savannah.utils.policy import PolicyOutput


class FFNBlock(nn.Module):
    # Refer: https://sebastianraschka.com/faq/docs/swiglu-modern-llms.html
    def __init__(self, embed_dim: int, feedforward_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(embed_dim, feedforward_dim)
        self.w_up = nn.Linear(embed_dim, feedforward_dim)

        self.w_down = nn.Linear(feedforward_dim, embed_dim)

        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_up = self.w_up(x)
        x_gate = self.w_gate(x)

        x_out = self.silu(x_gate) * x_up

        return self.w_down(x_out)


class DiTCrossAttnBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        cond_dim: int,
        num_attn_heads: int,
        feedforward_dim: int,
        use_rope: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.cond_dim = cond_dim
        self.num_attn_heads = num_attn_heads
        self.feedforward_dim = feedforward_dim
        self.dropout = dropout

        if use_rope:
            self.self_attn_block = RoPESelfAttention(embed_dim, num_attn_heads)
        else:
            self.self_attn_block = SelfAttention(embed_dim, num_attn_heads)

        self.cross_attn_block = CrossAttention(embed_dim, num_attn_heads)
        self.ffn_block = FFNBlock(embed_dim, feedforward_dim)

        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)

        self.dropout_layer = nn.Dropout(self.dropout)

        self.adaln_block = nn.Sequential(
            nn.SiLU(), nn.Linear(self.cond_dim, 9 * self.embed_dim)
        )
        self._init_adaln_zero()

    def _init_adaln_zero(self):
        """Initializes the alpha scaling projections to 0 so the network

        initially behaves like an identity skip connection.
        """
        nn.init.constant_(self.adaln_block[-1].weight, 0.0)
        nn.init.constant_(self.adaln_block[-1].bias, 0.0)

    def forward(
        self, x: torch.Tensor, x_kv: torch.Tensor, x_cond: torch.Tensor
    ) -> torch.Tensor:
        # x_time: tokens matching to timestep of denoising
        # x: is the tokens that are input to the decoder and the condition tokens

        # AdaLN stuff
        beta1, gamma1, alpha1, beta2, gamma2, alpha2, beta3, gamma3, alpha3 = (
            torch.split(
                self.adaln_block(x_cond),
                self.embed_dim,
                dim=-1,
            )
        )

        # Masked-Self Attention Block
        x_norm = self.layer_norm1(x)
        x_pre_attn = gamma1 * x_norm + beta1
        x_attn = self.self_attn_block(x_pre_attn)
        x_post_attn = alpha1 * x_attn

        # Residual Connection
        x_post_attn = self.dropout_layer(x_post_attn)
        x = x + x_post_attn

        x_norm = self.layer_norm2(x)
        x_pre_attn = gamma2 * x_norm + beta2
        x_attn = self.cross_attn_block(x_pre_attn, x_kv)
        x_post_attn = alpha2 * x_attn

        x_post_attn = self.dropout_layer(x_post_attn)
        x = x + x_post_attn

        # FFN Block
        x_norm = self.layer_norm3(x)
        x_pre_ffn = gamma3 * x_norm + beta3
        x_ffn = self.ffn_block(x_pre_ffn)
        x_post_ffn = alpha3 * x_ffn

        # Residual Connection
        x_post_ffn = self.dropout_layer(x_post_ffn)
        x = x + x_post_ffn

        return x


class DITCrossAttnPolicy(Policy):
    def __init__(
        self,
        embed_dim: int,
        time_embed_dim: int,
        state_embed_dim: int,
        decoder_num_blocks: int,
        decoder_num_attn_heads: int,
        decoder_feedforward_dim: int,
        decoder_dropout: float,
        state_dim: int,
        num_obs: int,
        action_dim: int,
        action_horizon: int,
        num_cameras: int,
        use_rope: bool,
        vision_encoder: VisionEncoder,
        language_encoder: LanguageEncoder | None,
        objective: PolicyObjective,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.time_embed_dim = time_embed_dim
        self.state_embed_dim = state_embed_dim
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._action_horizon = action_horizon

        self.vision_encoder = vision_encoder
        self.num_cameras = num_cameras
        self.objective = objective

        self.use_rope = use_rope
        self.use_spe = not self.use_rope

        self.state_encoder = StateEncoder(self._state_dim, self.state_embed_dim)
        self.language_encoder = language_encoder
        if language_encoder is not None:
            logger.debug("Initialised Language Encoder into the model")

        self.n_state_tokens = num_obs
        self.n_time_tokens = 1

        self.condition_dim = (
            self.n_state_tokens * self.state_embed_dim
            + self.n_time_tokens * self.time_embed_dim
        )

        self.decoder = nn.ModuleList(
            [
                DiTCrossAttnBlock(
                    embed_dim,
                    self.condition_dim,
                    decoder_num_attn_heads,
                    decoder_feedforward_dim,
                    use_rope,
                    decoder_dropout,
                )
                for _ in range(decoder_num_blocks)
            ]
        )

        self.timestep_embedding = nn.Sequential(
            TimeEmbedding(self.time_embed_dim),
            nn.Linear(self.time_embed_dim, 2 * self.time_embed_dim),
            nn.GELU(),
            nn.Linear(2 * self.time_embed_dim, self.time_embed_dim),
        )

        # Action Embedding
        self.action_embedding = nn.Sequential(
            nn.Linear(self._action_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        # Time encoding for actions. This can be skipped, if RoPE is being used internally
        if self.use_spe:
            action_time_embedding = get_sinusoidal_position_embedding(
                self._action_horizon, self.embed_dim
            )
            self.register_buffer("action_time_embedding", action_time_embedding)

        # Action reprojection
        self.action_reprojection = nn.Linear(self.embed_dim, self._action_dim)

        # Modality Embedding (for vision and language)
        self.modality_embedding = nn.Embedding(2, self.embed_dim)

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
        x_cam_tokens = self.vision_encoder(x_img)
        debug_stat("x_cam_tokens", x_cam_tokens)

        # Apply Modality Embedding
        x_cam_modality = self.modality_embedding(
            torch.zeros(1, dtype=torch.long, device=x_cam_tokens.device)
        )  # (1, embed_dim)
        x_cam_tokens = x_cam_tokens + x_cam_modality
        x_cam_tokens = x_cam_tokens + x_cam_modality

        # (B, N, state_dim) -> (B, N, embedding_dim)
        x_state = self.state_encoder(x_state)
        debug_stat("x_state (embedded)", x_state)

        if x_language is not None:
            if self.language_encoder is None:
                raise RuntimeError(
                    "Language instruction is provided but LanguageEncoder is not specified. Not using language instruction"
                )
            else:
                # (B, xx, embed_dim)
                x_language = self.language_encoder(x_language)
                debug_stat("x_lang", x_language)

                # Apply Modality Embedding
                x_language_modality = self.modality_embedding(
                    torch.tensor([1 for _ in range(x_language.shape[0])])
                )
                x_language = x_language + x_language_modality

        # (B,) -> (B, 1) optionally
        if len(x_time.shape) == 1:
            x_time = x_time.unsqueeze(1)

        # (B, 1) -> (B, 1, embed_dim)
        x_time = self.timestep_embedding(x_time).unsqueeze(1)
        debug_stat("x_time (embedded)", x_time)

        # Project the noisy actions to embeding space
        # (B, N_obs, action_dim) -> (B, N_obs, embed_dim)
        x_noisy_actions = self.action_embedding(noisy_actions)
        debug_stat("x_noisy_actions (embedded)", x_noisy_actions)

        if self.use_spe:
            x_noisy_actions = x_noisy_actions + self.action_time_embedding
            debug_stat("x_noisy_actions (+ pos embed)", x_noisy_actions)

        x_vision_language_tokens = []
        if x_language is None:
            x_vision_language_tokens = torch.cat([x_cam_tokens], dim=1)
        else:
            x_vision_language_tokens = torch.cat([x_language, x_cam_tokens], dim=1)

        x_state_tokens = rearrange(x_state, "b t d -> b (t d)")
        x_time_tokens = rearrange(x_time, "b t d -> b (t d)")

        x_cond_tokens_pooled = torch.cat([x_state_tokens, x_time_tokens], dim=-1)
        x_cond_tokens_pooled = x_cond_tokens_pooled.unsqueeze(1)
        debug_stat("x_cond_pooled", x_cond_tokens_pooled)

        x_dec_out = x_noisy_actions
        for dec in self.decoder:
            x_dec_out = dec(x_dec_out, x_vision_language_tokens, x_cond_tokens_pooled)
        debug_stat("x_dec_out", x_dec_out)

        x_action = self.action_reprojection(x_dec_out)
        debug_stat("x_action", x_action)
        return PolicyOutput(actions=x_action)
