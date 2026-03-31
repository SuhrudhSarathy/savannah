from lerobot.processor import PolicyAction
from savannah.utils.device import get_device
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional

from savannah.nn import SelfAttention
from savannah.nn.positional_embeddings import SinusoidalPositionalEncoding
from savannah.nn.time_embedding import TimeEmbedding
from savannah.utils.policy import PolicyOutput
from savannah.models.vision_encoder import VisionEncoder
from savannah.models.policy import Policy
from savannah.utils.observation import ObservationKey


class FMTransformerBlock(nn.Module):
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

        # Self Attn
        self.attn_block = SelfAttention(self.embed_dim, self.num_attn_heads)

        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

        # MLP to process timestep
        self.timestep_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, 6 * self.embed_dim),
            nn.GELU(),
            nn.Linear(6 * self.embed_dim, 6 * self.embed_dim),
        )

        # FFN MLP
        self.ffn_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.feedforward_dim),
            nn.GELU(),
            nn.Linear(self.feedforward_dim, self.embed_dim),
        )

        # Droupout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x (B, T, n_embed)
        # t (B, 1, n_embed)

        # (B, 1, n_embed) -> (B, 1, 6 * n_embed)
        timestep = self.timestep_mlp(t)

        # AdaLN
        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = torch.split(
            timestep, self.embed_dim, dim=-1
        )

        # ----- Attention
        x1 = self.layer_norm1(x)
        x1 = beta1 * x1 + gamma1
        x1 = self.attn_block(x1)

        x1 = alpha1 * x1

        # Residual connection
        x = x + x1
        # ----------

        # ------- FFN
        x2 = self.layer_norm2(x)
        x2 = beta2 * x2 + gamma2
        x2 = self.ffn_mlp(x2)

        x2 = alpha2 * x2

        # Residual connection
        x2 = x + x2

        x = self.dropout(x)

        return x


class AdaLN(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        self.mlp = nn.Linear(embed_dim, 2 * embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        ln = self.mlp(x)
        gamma, beta = torch.split(ln, self.embed_dim, dim=-1)

        return gamma * x + beta


class FlowMatchingPolicy(Policy):
    def __init__(
        self,
        embed_dim: int,
        num_attn_heads: int,
        feedforward_dim: int,
        num_blocks: int,
        state_dim: int,
        action_dim: int,
        action_horizon: int,
        vision_encoder: VisionEncoder,
        num_cameras: int = 1,
        flowmatching_scale: int = 1000,
        flowmatching_inference_steps: int = 10,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_attn_heads = num_attn_heads
        self.feedforward_dim = feedforward_dim

        self.num_blocks = num_blocks

        self.vision_encoder = vision_encoder
        self.num_cameras = num_cameras

        self._state_dim = state_dim
        self._action_dim = action_dim
        self._action_horizon = action_horizon

        self.flowmatching_scale = flowmatching_scale
        self.flowmatching_inference_steps = flowmatching_inference_steps

        # TimeEmbedding (This is generic, no learnable parameters, can be used at all places, timestep is needed)
        self.time_embedding = TimeEmbedding(self.embed_dim)

        # Camera Embedding
        self.camera_embedding = nn.Embedding(num_cameras, self.embed_dim)

        # State embedding
        self.state_embedding = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        # Action Embedding
        self.action_embedding = nn.Sequential(
            nn.Linear(self.action_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        # Action reprojection
        self.action_reprojection = nn.Sequential(
            nn.Linear(self.embed_dim, self.action_dim),
            nn.GELU(),
            nn.Linear(self.action_dim, self.action_dim),
        )

        # FMBlocks
        self.fm_blocks = nn.ModuleList(
            [
                FMTransformerBlock(embed_dim, num_attn_heads, feedforward_dim)
                for _ in range(num_blocks)
            ]
        )

        # Layer Norm
        self.layer_norm = AdaLN(embed_dim)

    def _encode_cameras(self, images: list[torch.Tensor]) -> torch.Tensor:
        """
        images: list of (B, N, 3, H, W), one per camera
        returns: (B, N_cams * N_tokens, embed_dim)
        """
        cam_tokens = []

        for cam_idx, img in enumerate(images):
            # Number of temporal observations
            n = img.shape[1]

            # Rearrange the image and pass it through the vision encoder to get tokens
            img_r = rearrange(img, "b n c h w -> (b n) c h w")
            tokens = self.vision_encoder(img_r)  # (B*T, N_tokens, embed_dim)

            # Rearrange tokens to fallback into the correct temporal order
            tokens_r = rearrange(tokens, "(b n) t embed_dim -> b n t embed_dim", n=n)

            # Project informal about the obs time to embedding space
            t = torch.arange(n, device=img.device).unsqueeze(1)  # (N, 1)
            # (N, 1) -> (N, embed_dim)
            time_embedding = self.time_embedding(t)
            # (N, embed_dim) -> (1, N, 1, batch_size)
            time_embedding = time_embedding.view(1, n, 1, -1)

            # Compute the camera embedding
            # (1, ) -> (embed_dim,)
            camera_embedding = self.camera_embedding(
                torch.tensor(cam_idx, device=img.device)
            )
            # (embed_dim, ) -> (1, 1, 1, embed_dim)
            camera_embedding = camera_embedding.view(1, 1, 1, -1)

            # Adds the vision tokens with the time_embedding and the camera embedding
            # Final shape = (B, N, T, embed_dim)
            added_tokens = tokens_r + time_embedding + camera_embedding

            # Flatten spatial and temporal dimensions into a single sequence
            tokens = rearrange(added_tokens, "b n t embed_dim -> b (n t) embed_dim")

            cam_tokens.append(tokens)
        return torch.cat(cam_tokens, dim=1)  # (B, N_cams * N_tokens, embed_dim)

    def forward(self, obs: dict[str, torch.Tensor], *args, **kwargs) -> PolicyOutput:
        x_img = obs[ObservationKey.images]  # list[(B, M, 3, H, W)]
        x_state = obs[ObservationKey.state]  # (B, N, state_dim)
        x_time = obs[ObservationKey.time]  # (B,) # This is between 0 -> 1

        # Get noisy actions from kwargs
        noisy_actions = kwargs.get("noisy_actions", None)
        if noisy_actions is None:
            raise AssertionError("Pass noisy action for the model to run")

        # (B, T, embed_dim)
        x_cam_tokens = self._encode_cameras(x_img)

        # (B, N, state_dim) -> (B, N, embedding_dim)
        x_state = self.state_embedding(x_state)
        # Add time embeddings for states also
        x_state_time_embed = self.time_embedding(
            torch.arange(x_state.shape[1], device=x_state.device).unsqueeze(1)
        )

        x_state = x_state + x_state_time_embed

        # Scale the time embeddings for better activation
        x_time = x_time * self.flowmatching_scale

        # (B,) -> (B, 1) optionally
        if len(x_time.shape) == 1:
            x_time = x_time.unsqueeze(1)

        # (B, 1) -> (B, 1, embed_dim)
        x_time = self.time_embedding(x_time).unsqueeze(1)

        # Project the noisy actions to embeding space
        # (B, N_obs, action_dim) -> (B, N_obs, embed_dim)
        x_noisy_actions = self.action_embedding(noisy_actions)
        # Add time embeddings for noisy_actions
        x_noisy_actions_time_embed = self.time_embedding(
            torch.arange(
                x_noisy_actions.shape[1], device=x_noisy_actions.device
            ).unsqueeze(1)
        )

        x_noisy_actions = x_noisy_actions + x_noisy_actions_time_embed

        # Concatenate image tokens and proprio tokens
        x_input_tokens = torch.cat([x_cam_tokens, x_state, x_noisy_actions], dim=1)

        for block in self.fm_blocks:
            # (B, T, n_embed) -> (B, T, n_embed)
            x_input_tokens = block(x_input_tokens, x_time)

        # (B, T, embed_dim) -> (B, T, embed_dim)
        x_out = self.layer_norm(x_input_tokens)

        # T = cam_tokens + N_state + N_obs
        # (B, T, embed_dim) -> (B, N_obs, embed_dim)
        x_out_action = x_out[:, -self.action_horizon :, :]

        # (B, N_obs, embed_dim) -> (B, N_obs, action_dim)
        x_out_action = self.action_reprojection(x_out_action)

        output = PolicyOutput(actions=x_out_action)

        return output

    def compute_action(self, obs: dict[str, torch.Tensor]) -> PolicyOutput:
        with torch.no_grad():
            B = obs[ObservationKey.state].shape[0]
            device = obs[ObservationKey.state].device

            # Starting point, pure noise
            x_t = torch.randn((B, self.action_horizon, self.action_dim), device=device)

            # Simple ODE solver
            dt = 1.0 / self.flowmatching_inference_steps

            # Run the inference
            for i in range(self.flowmatching_inference_steps):
                t_val = i * dt
                t_tensor = torch.full((B,), t_val, device=device)

                obs[ObservationKey.time] = t_tensor

                # Get the predictions from model
                v_pred = self.forward(obs, noisy_actions=x_t)

                # Integrate the predictions
                x_t = x_t + v_pred.actions * dt

            return PolicyOutput(actions=x_t)

    def compute_loss(
        self,
        obs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        B = obs[ObservationKey.state].shape[0]
        device = obs[ObservationKey.state].device

        # Get ground truth actions
        x_1 = obs[ObservationKey.gt_actions]

        # Sample noise like x_1
        x_0 = torch.randn_like(x_1)

        # Random sample time
        t = torch.rand((B,), device=device)

        # Get the x_t
        x_t = (1 - t.view(B, 1, 1)) * x_0 + t.view(B, 1, 1) * x_1

        # Compute target
        target = x_1 - x_0

        # Populate the Observation with time for model
        obs[ObservationKey.time] = t

        # Compute the predicted velocity from the model
        pred_vel = self.forward(obs, noisy_actions=x_t).actions

        # Compute loss
        loss = F.mse_loss(pred_vel, target)

        return loss


if __name__ == "__main__":
    from savannah.models.backbones import DummyVisionBackbone

    embed_dim = 128
    num_attn_heads = 4
    num_blocks = 6
    feedforward_dim = 4 * embed_dim
    num_cameras = 3

    state_dim = 6
    action_dim = 6
    action_horizon = 12

    device = get_device()

    feature_extractor = DummyVisionBackbone(downsample_factor=16).to(device)
    vision_encoder = VisionEncoder(backbone=feature_extractor, embed_dim=embed_dim).to(
        device
    )

    fm = FlowMatchingPolicy(
        embed_dim,
        num_attn_heads,
        feedforward_dim,
        num_blocks,
        state_dim,
        action_dim,
        action_horizon,
        vision_encoder,
        num_cameras,
    ).to(device)

    obs = {}
    obs[ObservationKey.images] = [torch.randn(8, 3, 3, 224, 224).to(device)]
    obs[ObservationKey.state] = torch.randn(8, 3, state_dim).to(device)
    obs[ObservationKey.gt_actions] = torch.randn(8, action_horizon, action_dim).to(
        device
    )

    # noisy_action = torch.randn((8, action_horizon, action_dim)).to(device)
    print(fm.num_params())

    out = fm.compute_action(obs)

    print(out.actions.shape)

    loss = fm.compute_loss(obs)

    print(loss)
