"""
Visualizes the DiT decoder's attention:
(1) Cross-attention — which key/value tokens each action-query token attends to,
    broken down by modality (language, vision, state).
(2) Composed Spatial Attention — if an attention pooler / resampler is used,
    multiplies the DiT cross-attention by the resampler cross-attention:
        A_spatial = A_DiT (Tq, K) @ A_pool (K, N_patches)
    to trace attention back to exact 2D image coordinates.
(3) Self-attention — how each action-horizon step attends to other action-horizon steps.
"""

import math
import os
import sys
from typing import NamedTuple

try:
    import imp
except ImportError:
    from types import ModuleType

    imp = ModuleType("imp")
    sys.modules["imp"] = imp

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from savannah.factory import build_policy, build_task
from savannah.utils.checkpoint import resolve_checkpoint_path
from savannah.utils.device import get_device
from savannah.utils.log import logger, setup_logging
from savannah.utils.observation import ObservationKey


def _select_sample(obs: dict, idx: int = 0) -> dict:
    out = {}
    for k, v in obs.items():
        if isinstance(v, list):
            out[k] = [t[idx : idx + 1] for t in v]
        elif isinstance(v, torch.Tensor):
            out[k] = v[idx : idx + 1]
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------- #
# Hooks: Resampler / Attention Pooler
# --------------------------------------------------------------------------- #


def _register_resampler_hooks(
    vision_encoder,
) -> tuple[
    list[tuple[torch.Tensor, torch.Tensor]], list[torch.utils.hooks.RemovableHandle]
]:
    """Captures the (query, kv) inputs into VisionEncoder.attn_pooling's last block.

    VisionEncoder shares a single AttentionPooling module across all cameras --
    VisionEncoder.forward calls it once per camera (in camera order), and the whole
    policy forward runs once per denoising step -- so the returned list interleaves
    as [step0_cam0, step0_cam1, ..., step1_cam0, ...].
    """
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []
    handles = []

    attn_pooling = getattr(vision_encoder, "attn_pooling", None)
    if attn_pooling is not None:
        # AttentionPoolingBlock.forward calls self.attn(q, kv) positionally.
        attn_module = attn_pooling.layers[-1].attn

        def hook(module, inputs, output):
            q, kv = inputs[0], inputs[1]
            captured.append((q.detach(), kv.detach()))

        handles.append(attn_module.register_forward_hook(hook))

    return captured, handles


# --------------------------------------------------------------------------- #
# Hooks: DiT Decoder
# --------------------------------------------------------------------------- #


def _register_cross_attn_hooks(
    policy,
) -> tuple[
    dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
    list[torch.utils.hooks.RemovableHandle],
]:
    captured: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    handles = []

    def make_hook(block_idx):
        def hook(module, inputs, output):
            x, kv = inputs[0], inputs[1]
            captured.setdefault(block_idx, []).append((x.detach(), kv.detach()))

        return hook

    for block_idx, block in enumerate(policy.decoder):
        handles.append(
            block.cross_attn_block.register_forward_hook(make_hook(block_idx))
        )

    return captured, handles


@torch.no_grad()
def _cross_attn_weights(module, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
    nh = module.num_attn_heads if hasattr(module, "num_attn_heads") else module.n_heads
    hd = module.embed_dim // nh

    q = module.q_projection(x)
    k, _ = torch.split(module.kv_projection(kv), module.embed_dim, dim=-1)

    q = q.view(q.shape[0], q.shape[1], nh, hd).transpose(1, 2)
    k = k.view(k.shape[0], k.shape[1], nh, hd).transpose(1, 2)

    attn = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
    return attn.softmax(dim=-1)  # (B, nh, Tq, Tkv)


def _register_self_attn_hooks(
    policy,
) -> tuple[dict[int, list[torch.Tensor]], list[torch.utils.hooks.RemovableHandle]]:
    captured: dict[int, list[torch.Tensor]] = {}
    handles = []

    def make_hook(block_idx):
        def hook(module, inputs, output):
            captured.setdefault(block_idx, []).append(inputs[0].detach())

        return hook

    for block_idx, block in enumerate(policy.decoder):
        handles.append(
            block.self_attn_block.register_forward_hook(make_hook(block_idx))
        )

    return captured, handles


@torch.no_grad()
def _self_attn_weights(module, x: torch.Tensor) -> torch.Tensor:
    nh = module.num_attn_heads if hasattr(module, "num_attn_heads") else module.n_heads
    hd = module.embed_dim // nh
    B, T, _ = x.shape

    q, k, _ = torch.split(module.attn_projection(x), module.embed_dim, dim=-1)
    q = q.view(B, T, nh, hd).transpose(1, 2)
    k = k.view(B, T, nh, hd).transpose(1, 2)

    if hasattr(module, "rope"):
        q, k = module.rope(q, k)

    attn = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
    return attn.softmax(dim=-1)  # (B, nh, T, T)


# --------------------------------------------------------------------------- #
# Layout & Visualizations
# --------------------------------------------------------------------------- #


class VisionTokenLayout(NamedTuple):
    tokens_per_image: int
    grid_shape: tuple[int, int]
    has_cls_token: bool
    is_resampled: bool


def _vision_token_layout(vision_encoder) -> VisionTokenLayout:
    backbone = getattr(vision_encoder, "backbone", vision_encoder)
    patches_per_side = getattr(backbone, "patches_per_side", 14)
    has_cls = getattr(backbone, "use_eos_only", False) is False

    is_resampled = getattr(vision_encoder, "attn_pooling", None) is not None

    tokens_per_image = vision_encoder.tokens_per_image
    return VisionTokenLayout(
        tokens_per_image=tokens_per_image,
        grid_shape=(patches_per_side, patches_per_side),
        has_cls_token=has_cls,
        is_resampled=is_resampled,
    )


def _kv_token_layout(
    policy, total_kv_tokens: int, has_language: bool
) -> dict[str, tuple[int, int]]:
    n_vision = (
        policy.vision_encoder.num_tokens
        if hasattr(policy.vision_encoder, "num_tokens")
        else (
            policy.vision_encoder.tokens_per_image
            * policy.num_cameras
            * policy.n_state_tokens
        )
    )
    n_state = policy.n_state_tokens
    n_language = total_kv_tokens - n_vision - n_state if has_language else 0

    layout = {}
    offset = 0
    if n_language > 0:
        layout["language"] = (offset, offset + n_language)
        offset += n_language
    layout["vision"] = (offset, offset + n_vision)
    offset += n_vision
    layout["state"] = (offset, offset + n_state)
    return layout


def _normalized_entropy(p: torch.Tensor) -> float:
    p = p.clamp_min(1e-12)
    ent = -(p * p.log()).sum().item()
    return ent / math.log(max(p.numel(), 2))


def _overlay_heatmap(image_hwc: np.ndarray, heat: np.ndarray, alpha: float = 0.45):
    cmap = plt.get_cmap("jet")
    heat_rgb = cmap(heat)[..., :3]
    return (1 - alpha) * image_hwc + alpha * heat_rgb


def _plot_block_spatial(
    block_idx: int,
    composed_spatial_weights: torch.Tensor,  # (num_cams, num_obs, H_patches, W_patches)
    images: dict,  # cam_name -> (num_obs, H, W, 3)
    cameras: list[str],
    num_obs: int,
    output_dir: str,
    step: int,
):
    """Renders pixel overlays of the traced spatial attention."""
    fig, axes = plt.subplots(
        len(cameras), num_obs, figsize=(4 * num_obs, 4 * len(cameras)), squeeze=False
    )
    global_max = composed_spatial_weights.max().item()

    for cam_i, cam in enumerate(cameras):
        for obs_i in range(num_obs):
            ax = axes[cam_i][obs_i]
            img = images[cam][obs_i]
            grid = composed_spatial_weights[cam_i, obs_i]

            heat = grid[None, None]  # (1, 1, H, W)
            heat = F.interpolate(
                heat, size=img.shape[:2], mode="bilinear", align_corners=False
            )[0, 0].numpy()
            heat = heat / max(global_max, 1e-12)

            blended = _overlay_heatmap(img, heat)
            ax.imshow(np.clip(blended, 0, 1))
            ax.set_title(
                f"{cam} | obs t-{num_obs - 1 - obs_i} | peak={grid.max().item():.4f}",
                fontsize=9,
            )
            ax.axis("off")

    fig.suptitle(
        f"Decoder Block {block_idx} — Composed Spatial Attention (Step {step})"
    )
    fig.tight_layout()
    out_path = os.path.join(output_dir, f"block_{block_idx:02d}_spatial_attn.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_query_modality_matrix(
    block_idx: int,
    weights: torch.Tensor,  # (1, nh, Tq, Tkv)
    layout: dict[str, tuple[int, int]],
    cameras: list[str],
    num_obs: int,
    output_dir: str,
):
    w = weights[0].mean(dim=0)  # (Tq, Tkv)
    tq = w.shape[0]

    vision_start, vision_end = layout["vision"]
    vision = w[:, vision_start:vision_end]
    per_cam_obs = vision.reshape(tq, len(cameras) * num_obs, -1).sum(dim=-1)
    labels = [f"{cam}\nt-{num_obs - 1 - o}" for cam in cameras for o in range(num_obs)]
    columns = [per_cam_obs]

    state_start, state_end = layout["state"]
    state = w[:, state_start:state_end]
    columns.append(state)
    labels += [f"state\nt-{num_obs - 1 - o}" for o in range(num_obs)]

    if "language" in layout:
        lang_start, lang_end = layout["language"]
        language = w[:, lang_start:lang_end].sum(dim=-1, keepdim=True)
        columns.append(language)
        labels.append("language")

    matrix = torch.cat(columns, dim=-1)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 6))
    im = ax.imshow(matrix.numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("action-horizon query index")
    ax.set_title(f"Decoder block {block_idx} — Attention Mass per Modality")
    fig.colorbar(im, ax=ax, label="summed weight")
    fig.tight_layout()
    out_path = os.path.join(
        output_dir, f"block_{block_idx:02d}_query_modality_matrix.png"
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_self_attention(
    block_idx: int,
    weights: torch.Tensor,
    head_idx: int | None,
    output_dir: str,
    step: int,
) -> tuple[str, float]:
    w = weights[0]
    w = w[head_idx].unsqueeze(0) if head_idx is not None else w
    w = w.mean(dim=0).cpu()

    mean_entropy = float(
        np.mean([_normalized_entropy(w[i]) for i in range(w.shape[0])])
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(w.numpy(), cmap="viridis")
    ax.set_xlabel("Key Action-Horizon Step")
    ax.set_ylabel("Query Action-Horizon Step")
    ax.set_title(
        f"Decoder Block {block_idx} — Self-Attn (Step {step})\n"
        f"Mean Entropy = {mean_entropy:.3f}"
    )
    fig.colorbar(im, ax=ax, label="Attention Weight")
    fig.tight_layout()
    out_path = os.path.join(output_dir, f"block_{block_idx:02d}_self_attn.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path, mean_entropy


# --------------------------------------------------------------------------- #
# Main Entry Point
# --------------------------------------------------------------------------- #


def run_attention_viz(cfg: DictConfig) -> None:
    setup_logging(log_dir="logs", level=cfg.log_level)
    device = get_device()
    logger.info("Using device: {}", device)

    task = build_task(cfg, device=device)
    policy = build_policy(cfg).to(device)

    checkpoint_path = resolve_checkpoint_path(cfg)
    logger.info("Loading checkpoint: {}", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_key = "ema_state_dict" if cfg.use_ema else "model_state_dict"
    policy.load_state_dict(checkpoint[state_key])
    policy.eval()

    episode_index = cfg.episode_index or checkpoint.get("episode_index", 0)
    frame_index = cfg.frame_index or checkpoint.get("frame_index", 0)
    logger.info("Visualizing episode={} frame={}", episode_index, frame_index)

    task.config.num_workers = 0
    loader = task.get_episode_loader(episode_index, frame_index)
    raw_batch = next(iter(loader))
    formatted = task.format_batch(raw_batch)
    obs = _select_sample(formatted, idx=0)

    cameras = list(task.config.cameras)
    num_obs = task.config.obs_horizon
    vision_layout = _vision_token_layout(policy.vision_encoder)

    images = {
        cam: obs[ObservationKey.images][i][0].permute(0, 2, 3, 1).cpu().numpy()
        for i, cam in enumerate(cameras)
    }
    has_language = (
        obs.get(ObservationKey.language) is not None
        and getattr(policy, "language_encoder", None) is not None
    )

    # Attach forward hooks
    captured_cross, cross_handles = _register_cross_attn_hooks(policy)
    captured_self, self_handles = _register_self_attn_hooks(policy)
    captured_pool, pool_handles = _register_resampler_hooks(policy.vision_encoder)

    with torch.no_grad():
        policy.compute_action(obs)

    # Clean up hooks
    for h in cross_handles + self_handles + pool_handles:
        h.remove()

    os.makedirs(cfg.output_dir, exist_ok=True)
    block_indices = (
        range(len(policy.decoder)) if cfg.block_idx is None else [cfg.block_idx]
    )

    # Precompute pooler cross-attention weights A_pool (K_latents -> Patches).
    # Regroup the interleaved [step0_cam0, step0_cam1, ...] capture list by step
    # so we can pick out the same denoise step the DiT decoder hooks use.
    pool_attn_per_cam = {}
    pool_module = getattr(policy.vision_encoder, "attn_pooling", None)
    if vision_layout.is_resampled and pool_module is not None and captured_pool:
        attn_module = pool_module.layers[-1].attn
        steps = [
            captured_pool[i : i + len(cameras)]
            for i in range(0, len(captured_pool), len(cameras))
        ]
        step_calls = steps[cfg.denoise_step]
        for cam_idx, cam in enumerate(cameras):
            q, kv = step_calls[cam_idx]
            # (B, nh, K, N_patches_with_cls) -> (K, N_patches_with_cls)
            pool_weights = _cross_attn_weights(attn_module, q, kv)[0].mean(dim=0)
            if vision_layout.has_cls_token:
                # Strip CLS attention for spatial heatmap
                pool_weights = pool_weights[:, 1:]
            pool_attn_per_cam[cam] = pool_weights.cpu()

    for block_idx in block_indices:
        x, kv = captured_cross[block_idx][cfg.denoise_step]
        weights = _cross_attn_weights(
            policy.decoder[block_idx].cross_attn_block, x, kv
        ).cpu()

        layout = _kv_token_layout(policy, weights.shape[-1], has_language)
        vision_start, vision_end = layout["vision"]

        # 1. Reduce over heads and (optionally) query horizon steps
        w_avg = weights[0].mean(dim=0)  # (Tq, Tkv)
        if cfg.query_idx is not None:
            w_avg = w_avg[cfg.query_idx : cfg.query_idx + 1]  # (1, Tkv)

        # Vision tokens: (Tq, Num_Cameras * Num_Obs * Tokens_Per_Image)
        vision_weights = w_avg[:, vision_start:vision_end]
        tq = vision_weights.shape[0]

        # Reshape to (Tq, Num_Cameras, Num_Obs, Tokens_Per_Image)
        v_reshaped = vision_weights.view(
            tq, len(cameras), num_obs, vision_layout.tokens_per_image
        )

        h_grid, w_grid = vision_layout.grid_shape
        composed_grid = torch.zeros(len(cameras), num_obs, h_grid, w_grid)

        # 2. Compute Composed Spatial Attention
        for cam_i, cam in enumerate(cameras):
            for obs_i in range(num_obs):
                a_dit = v_reshaped[:, cam_i, obs_i].mean(dim=0)  # (Tokens_Per_Image,)

                if vision_layout.is_resampled and cam in pool_attn_per_cam:
                    # Chain Rule: A_composed = A_DiT @ A_pool
                    a_pool = pool_attn_per_cam[cam]  # (K, H*W)
                    a_composed = a_dit @ a_pool  # (H*W,)
                    grid = a_composed.view(h_grid, w_grid)
                else:
                    # Direct spatial ViT patches
                    patches = a_dit[1:] if vision_layout.has_cls_token else a_dit
                    grid = patches.view(h_grid, w_grid)

                composed_grid[cam_i, obs_i] = grid

        spatial_path = _plot_block_spatial(
            block_idx,
            composed_grid,
            images,
            cameras,
            num_obs,
            cfg.output_dir,
            step=cfg.denoise_step,
        )

        matrix_path = _plot_query_modality_matrix(
            block_idx, weights, layout, cameras, num_obs, cfg.output_dir
        )

        x_self = captured_self[block_idx][cfg.denoise_step]
        self_weights = _self_attn_weights(
            policy.decoder[block_idx].self_attn_block, x_self
        ).cpu()
        self_path, _ = _plot_self_attention(
            block_idx, self_weights, cfg.head_idx, cfg.output_dir, step=cfg.denoise_step
        )

        logger.info("Wrote {}, {} and {}", spatial_path, matrix_path, self_path)

    logger.info("Done. Outputs saved to {}", cfg.output_dir)


@hydra.main(version_base=None, config_path="../../configs", config_name="attention_viz")
def main(cfg: DictConfig) -> None:
    run_attention_viz(cfg)


if __name__ == "__main__":
    main()
