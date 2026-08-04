import warnings

import torch
import torch.nn as nn
from einops import rearrange

from savannah.models.encoders import StateEncoder, VisionEncoder
from savannah.models.encoders.language_encoder import LanguageEncoder
from savannah.models.policy import Policy
from savannah.nn.positional_embeddings import get_sinusoidal_position_embedding
from savannah.nn.rope import RoPESelfAttention
from savannah.nn.self_attention import SelfAttention
from savannah.nn.time_embedding import TimeEmbedding
from savannah.objectives import PolicyObjective
from savannah.utils.debug import debug_stat
from savannah.utils.log import logger
from savannah.utils.observation import ObservationKey
from savannah.utils.policy import PolicyOutput


class DiTBlock(nn.Module):
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
            self.attn_block = RoPESelfAttention(embed_dim, num_attn_heads)
        else:
            self.attn_block = SelfAttention(embed_dim, num_attn_heads)

        self.ffn_block = nn.Sequential(
            nn.Linear(self.embed_dim, self.feedforward_dim),
            nn.GELU(),
            nn.Linear(self.feedforward_dim, self.embed_dim),
        )

        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

        self.dropout_layer = nn.Dropout(self.dropout)

        self.adaln_block = nn.Linear(self.cond_dim, 6 * self.embed_dim)
        self._init_adaln_zero()

    def _init_adaln_zero(self):
        """Initializes the alpha scaling projections to 0 so the network

        initially behaves like an identity skip connection.
        """
        nn.init.constant_(self.adaln_block.weight, 0.0)
        nn.init.constant_(self.adaln_block.bias, 0.0)

    def forward(self, x: torch.Tensor, x_cond: torch.Tensor) -> torch.Tensor:
        # x_time: tokens matching to timestep of denoising
        # x: is the tokens that are input to the decoder and the condition tokens

        # AdaLN stuff
        beta1, gamma1, alpha1, beta2, gamma2, alpha2 = torch.split(
            self.adaln_block(x_cond),
            self.embed_dim,
            dim=-1,
        )

        # Masked-Self Attention Block
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


class LBMPolicy(Policy):
    def __init__(
        self,
        embed_dim: int,
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
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._action_horizon = action_horizon

        self.vision_encoder = vision_encoder
        self.num_cameras = num_cameras
        self.objective = objective

        self.use_rope = use_rope

        self.camera_embedding = nn.Embedding(num_cameras, embed_dim)

        self.state_encoder = StateEncoder(self._state_dim, self.embed_dim)
        self.language_encoder = language_encoder
        if language_encoder is not None:
            logger.debug("Initialised Language Encoder into the model")

        sinusoidal_position_embeddings = get_sinusoidal_position_embedding(
            self._action_horizon, self.embed_dim
        )
        self.register_buffer("action_time_embedding", sinusoidal_position_embeddings)

        self.n_language_tokens = 1 if language_encoder is not None else 0
        self.n_vision_tokens = num_cameras * num_obs * vision_encoder.tokens_per_image
        self.n_state_tokens = num_obs
        self.n_time_tokens = 1

        self.condition_dim = (
            self.n_language_tokens
            + self.n_vision_tokens
            + self.n_state_tokens
            + self.n_time_tokens
        ) * self.embed_dim

        self.decoder = nn.ModuleList(
            [
                DiTBlock(
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
            TimeEmbedding(embed_dim),
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

        # Action Embedding
        self.action_embedding = nn.Sequential(
            nn.Linear(self._action_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        # Action reprojection
        self.action_reprojection = nn.Linear(self.embed_dim, self._action_dim)

        # Modality Embedding
        self.modality_embed = nn.Embedding(4, self.embed_dim)

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
        x_cam_tokens = x_cam_tokens + self.modality_embed(
            torch.tensor(1, device=x_cam_tokens.device)
        )
        debug_stat("x_cam_tokens", x_cam_tokens)

        # (B, N, state_dim) -> (B, N, embedding_dim)
        x_state = self.state_encoder(x_state) + self.modality_embed(
            torch.tensor(2, device=x_state.device)
        )
        debug_stat("x_state (embedded)", x_state)

        if x_language is not None:
            if self.language_encoder is None:
                raise RuntimeError(
                    "Language instruction is provided but LanguageEncoder is not specified. Not using language instruction"
                )
            else:
                # (B, xx, embed_dim)
                x_language = self.language_encoder(x_language)
                x_language = x_language + self.modality_embed(
                    torch.tensor(0, device=x_language.device)
                )
                debug_stat("x_lang", x_language)

        # (B,) -> (B, 1) optionally
        if len(x_time.shape) == 1:
            x_time = x_time.unsqueeze(1)

        # (B, 1) -> (B, 1, embed_dim)
        x_time = self.timestep_embedding(x_time).unsqueeze(1)
        x_time = x_time + self.modality_embed(torch.tensor(3, device=x_time.device))
        debug_stat("x_time (embedded)", x_time)

        # Project the noisy actions to embeding space
        # (B, N_obs, action_dim) -> (B, N_obs, embed_dim)
        x_noisy_actions = self.action_embedding(noisy_actions)
        debug_stat("x_noisy_actions (embedded)", x_noisy_actions)

        x_noisy_actions = x_noisy_actions + self.action_time_embedding
        debug_stat("x_noisy_actions (+ pos embed)", x_noisy_actions)

        if x_language is None:
            x_cond_tokens = torch.cat([x_cam_tokens, x_state, x_time], dim=1)
        else:
            x_cond_tokens = torch.cat(
                [x_language, x_cam_tokens, x_state, x_time], dim=1
            )

        debug_stat("x_cond", x_cond_tokens)

        x_cond_tokens_pooled = rearrange(x_cond_tokens, "b n e -> b (n e)")
        x_cond_tokens_pooled = x_cond_tokens_pooled.unsqueeze(1)
        debug_stat("x_cond_pooled", x_cond_tokens_pooled)

        x_dec_out = x_noisy_actions
        for dec in self.decoder:
            x_dec_out = dec(x_dec_out, x_cond_tokens_pooled)
        debug_stat("x_dec_out", x_dec_out)

        x_action = self.action_reprojection(x_dec_out)
        debug_stat("x_action", x_action)
        return PolicyOutput(actions=x_action)


if __name__ == "__main__":
    from savannah.models.backbones.resnet_backbone import ResNetBackbone
    from savannah.models.encoders import VisionEncoder
    from savannah.models.lbm import DiTBlock, DiTPolicy
    from savannah.objectives import FlowMatchingObjective, OverfitObjective
    from savannah.utils.observation import ObservationKey

    B, H, W = 2, 64, 64
    STATE_DIM, ACTION_DIM, ACTION_HORIZON, EMBED_DIM = 7, 6, 8, 64
    NUM_CAMERAS = 2
    NUM_OBS = 1

    class StubLanguageEncoder(nn.Module):
        """Lightweight stand-in for LanguageEncoder (avoids downloading CLIP weights)."""

        def __init__(self, embed_dim: int):
            super().__init__()
            self.embed_dim = embed_dim
            self.proj = nn.Linear(1, embed_dim)

        def forward(self, text_inputs: list[str]) -> torch.Tensor:
            batch = len(text_inputs)
            dummy = torch.ones(batch, 1, 1)
            return self.proj(dummy)

    def _make_vision_encoder():
        backbone = ResNetBackbone(out_channels=32)
        return VisionEncoder(backbone, embed_dim=EMBED_DIM)

    def _make_policy(num_obs=NUM_OBS, objective=None, language_encoder=None):
        policy = DiTPolicy(
            embed_dim=EMBED_DIM,
            decoder_num_blocks=2,
            decoder_num_attn_heads=4,
            decoder_feedforward_dim=128,
            decoder_dropout=0.0,
            state_dim=STATE_DIM,
            num_obs=num_obs,
            action_dim=ACTION_DIM,
            action_horizon=ACTION_HORIZON,
            num_cameras=NUM_CAMERAS,
            vision_encoder=_make_vision_encoder(),
            language_encoder=language_encoder,
            objective=objective or OverfitObjective(),
        )
        policy.eval()
        return policy

    def make_obs(num_cameras=NUM_CAMERAS, num_obs=NUM_OBS, with_language=False):
        obs = {
            ObservationKey.images: [
                torch.randn(B, num_obs, 3, H, W) for _ in range(num_cameras)
            ],
            ObservationKey.state: torch.randn(B, num_obs, STATE_DIM),
            ObservationKey.time: torch.rand(B) * 100.0,
            ObservationKey.actions: torch.randn(B, ACTION_HORIZON, ACTION_DIM),
            ObservationKey.gt_actions: torch.randn(B, ACTION_HORIZON, ACTION_DIM),
        }
        if with_language:
            obs[ObservationKey.language] = [f"instruction {i}" for i in range(B)]
        return obs

    def policy():
        return _make_policy()

    def policy_flow():
        return _make_policy(objective=FlowMatchingObjective())

    def policy_multi_obs():
        return _make_policy(num_obs=2)

    def policy_with_language():
        return _make_policy(language_encoder=StubLanguageEncoder(EMBED_DIM))

    policy = policy()
    obs = make_obs()
    out = policy.compute_action(obs)
    print(out.actions.shape)
