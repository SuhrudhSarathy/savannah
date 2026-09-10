"""
Visualizes the DiT decoder's attention: for one fixed episode/frame from the
dataset (by default, the exact one an `overfit_dataset_batch.py` checkpoint
was overfit on — see `episode_index`/`frame_index` below), shows
(1) cross-attention — which key/value tokens each action-query token attends
to, broken down by modality (language, vision: camera/obs-history
step/patch, and state: obs-history step), and (2) self-attention — how each
action-horizon step attends to other action-horizon steps. The cross-attn
kv sequence is `[language tokens?] + vision tokens + state tokens`
(see DITCrossAttnPolicy.forward's `x_kv_tokens` concatenation) — state now
passes through cross-attention like any other modality. Only *time* skips
attention entirely and feeds the AdaLN pooled conditioning instead (see
DiTCrossAttnBlock.forward). There is also no self-attention over
language/vision/state: self-attention only runs over the noisy
action-horizon tokens.

Why: train/val (denoising) loss can look fine while the cross-attention
never actually learns to localize the gripper/object — the loss doesn't
distinguish "attends to the right patch" from "attends everywhere equally".
This script recomputes the *exact* softmax attention weights that
`CrossAttention`/`SelfAttention`/`RoPESelfAttention`.forward use internally
(same q/k projections, same scale, same RoPE rotation when applicable) by
hooking each block's `cross_attn_block`/`self_attn_block` — no changes to
the model code, since SDPA never exposes the weights it computes.

Usage:
    # episode/frame auto-read from the overfit checkpoint's recorded values
    python scripts/debug/visualize_cross_attention.py \
        checkpoint_path=checkpoints/overfit/overfit_final.ckpt \
        task=metaworld_button_press

    # a checkpoint from a full training run has no recorded episode/frame,
    # so pick one explicitly:
    python scripts/debug/visualize_cross_attention.py \
        checkpoint_path=checkpoints/model_best.ckpt \
        task=metaworld_drawer_open episode_index=3 frame_index=0

    # a specific block/query/head instead of the block-averaged view:
    python scripts/debug/visualize_cross_attention.py \
        checkpoint_path=... block_idx=2 query_idx=0 head_idx=3
"""

import sys

try:
    import imp
except ImportError:
    from types import ModuleType

    imp = ModuleType("imp")
    sys.modules["imp"] = imp

import math
import os

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
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
    """Slices a formatted batch down to a single sample, keeping batch dim=1."""
    out = {}
    for k, v in obs.items():
        if isinstance(v, list):
            out[k] = [t[idx : idx + 1] for t in v]
        elif isinstance(v, torch.Tensor):
            out[k] = v[idx : idx + 1]
        else:
            out[k] = v
    return out


def _register_cross_attn_hooks(
    policy,
) -> dict[int, list[tuple[torch.Tensor, torch.Tensor]]]:
    """Captures every (x, kv) pair passed into each block's CrossAttention.forward.

    One entry is appended per sampler step (compute_action calls model.forward
    once per denoising step, and every block's cross_attn_block fires once
    per forward), in step order — so captured[block_idx][-1] is the attention
    input for the *cleanest* x_t, and [0] is for the noisiest.
    """
    captured: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}

    def make_hook(block_idx):
        def hook(module, inputs, output):
            x, kv = inputs
            captured.setdefault(block_idx, []).append((x.detach(), kv.detach()))

        return hook

    for block_idx, block in enumerate(policy.decoder):
        block.cross_attn_block.register_forward_hook(make_hook(block_idx))

    return captured


@torch.no_grad()
def _cross_attn_weights(module, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
    """Recomputes softmax(QK^T / sqrt(d)) exactly as CrossAttention.forward does
    internally, but returns the weights instead of the pooled output — SDPA
    (used at train/inference time) never exposes them."""
    nh = module.n_heads
    hd = module.embed_dim // nh

    q = module.q_projection(x)
    k, v = torch.split(module.kv_projection(kv), module.embed_dim, dim=-1)

    q = q.view(q.shape[0], q.shape[1], nh, hd).transpose(1, 2)  # (B, nh, Tq, hd)
    k = k.view(k.shape[0], k.shape[1], nh, hd).transpose(1, 2)  # (B, nh, Tkv, hd)

    attn = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
    return attn.softmax(dim=-1)  # (B, nh, Tq, Tkv)


def _register_self_attn_hooks(policy) -> dict[int, list[torch.Tensor]]:
    """Captures the `x` input to each block's self_attn_block (the DiT decoder's
    only self-attention — it runs over the noisy action-horizon tokens; states
    never pass through attention, they only feed AdaLN via pooled conditioning).
    One entry per sampler step, in step order, same convention as cross-attn."""
    captured: dict[int, list[torch.Tensor]] = {}

    def make_hook(block_idx):
        def hook(module, inputs, output):
            captured.setdefault(block_idx, []).append(inputs[0].detach())

        return hook

    for block_idx, block in enumerate(policy.decoder):
        block.self_attn_block.register_forward_hook(make_hook(block_idx))

    return captured


@torch.no_grad()
def _self_attn_weights(module, x: torch.Tensor) -> torch.Tensor:
    """Recomputes softmax(QK^T / sqrt(d)) exactly as SelfAttention/RoPESelfAttention
    do internally (including the RoPE rotation when present), returning the
    weights instead of the pooled output."""
    nh = module.n_heads
    hd = module.embed_dim // nh
    B, T, _ = x.shape

    q, k, v = torch.split(module.attn_projection(x), module.embed_dim, dim=-1)
    q = q.view(B, T, nh, hd).transpose(1, 2)  # (B, nh, T, hd)
    k = k.view(B, T, nh, hd).transpose(1, 2)

    if hasattr(module, "rope"):
        q, k = module.rope(q, k)

    attn = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
    return attn.softmax(dim=-1)  # (B, nh, T, T)


def _plot_self_attention(
    block_idx: int,
    weights: torch.Tensor,  # (1, nh, T, T)
    head_idx: int | None,
    output_dir: str,
    step: int,
) -> tuple[str, float]:
    """(query action-step) x (key action-step) heatmap — shows whether each
    denoised action attends locally (banded near the diagonal, i.e. temporal
    smoothness) or diffusely across the whole chunk."""
    w = weights[0]  # (nh, T, T)
    w = w[head_idx].unsqueeze(0) if head_idx is not None else w
    w = w.mean(dim=0).cpu()  # (T, T)

    row_entropy = torch.stack([_row_entropy(w[i]) for i in range(w.shape[0])])
    mean_entropy = row_entropy.mean().item()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(w.numpy(), cmap="viridis")
    ax.set_xlabel("key action-horizon index")
    ax.set_ylabel("query action-horizon index")
    ax.set_title(
        f"Decoder block {block_idx} — self-attn (step {step})\n"
        f"mean per-query entropy = {mean_entropy:.3f}"
    )
    fig.colorbar(im, ax=ax, label="attention weight")
    fig.tight_layout()
    out_path = os.path.join(output_dir, f"block_{block_idx:02d}_self_attn.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path, mean_entropy


def _row_entropy(row: torch.Tensor) -> torch.Tensor:
    return torch.tensor(_normalized_entropy(row))


def _vision_token_grid_shape(vision_encoder, num_cameras: int, num_obs: int):
    """Returns (h, w) patches per image, or None if the backbone pools to a
    single token per image (e.g. spatial-softmax) — no spatial map to draw."""
    tokens_per_image = vision_encoder.backbone.tokens_per_image
    if tokens_per_image == 1:
        return None
    side = round(math.sqrt(tokens_per_image))
    if side * side != tokens_per_image:
        logger.warning(
            "tokens_per_image={} isn't a perfect square; can't lay it out as an "
            "image-shaped grid, falling back to a 1D strip",
            tokens_per_image,
        )
        return (1, tokens_per_image)
    return (side, side)


def _reduce_weights(
    weights: torch.Tensor, query_idx: int | None, head_idx: int | None
) -> torch.Tensor:
    """weights: (1, nh, Tq, Tkv) -> (Tkv,), averaging over query/head unless
    a specific index is requested."""
    w = weights[0]  # (nh, Tq, Tkv)
    w = w[head_idx].unsqueeze(0) if head_idx is not None else w
    w = w.mean(dim=0)  # (Tq, Tkv)
    w = w[query_idx].unsqueeze(0) if query_idx is not None else w
    w = w.mean(dim=0)  # (Tkv,)
    return w


def _kv_token_layout(
    policy, total_kv_tokens: int, has_language: bool
) -> dict[str, tuple[int, int]]:
    """The cross-attn kv sequence is built by DITCrossAttnPolicy.forward as
    `[language tokens?] + vision tokens + state tokens`. Returns each
    modality's (start, end) slice into that Tkv axis, inferring the language
    token count (it's a runtime-dependent number of CLIP tokens, not a fixed
    config value) as whatever's left over after vision + state."""
    n_vision = policy.vision_encoder.num_tokens
    n_state = policy.n_state_tokens
    n_language = total_kv_tokens - n_vision - n_state if has_language else 0
    assert n_language >= 0, (
        f"kv token count ({total_kv_tokens}) is smaller than vision+state "
        f"({n_vision}+{n_state}) — layout assumption is wrong"
    )

    layout = {}
    offset = 0
    if n_language > 0:
        layout["language"] = (offset, offset + n_language)
        offset += n_language
    layout["vision"] = (offset, offset + n_vision)
    offset += n_vision
    layout["state"] = (offset, offset + n_state)
    offset += n_state
    return layout


def _normalized_entropy(p: torch.Tensor) -> float:
    """0 = fully concentrated on one token, 1 = uniform over all Tkv tokens."""
    p = p.clamp_min(1e-12)
    ent = -(p * p.log()).sum().item()
    return ent / math.log(p.numel())


def _overlay_heatmap(image_hwc: np.ndarray, heat: np.ndarray, alpha: float = 0.45):
    """image_hwc in [0,1], heat already resized to image resolution and in [0,1]."""
    cmap = plt.get_cmap("jet")
    heat_rgb = cmap(heat)[..., :3]
    return (1 - alpha) * image_hwc + alpha * heat_rgb


def _plot_block(
    block_idx: int,
    weights_vec: torch.Tensor,  # (T_vision,) — vision tokens only, kv's language/state slices excluded
    images: dict,  # cam_name -> (num_obs, H, W, 3) numpy in [0,1]
    grid_shape,  # (h, w) or None
    cameras: list[str],
    num_obs: int,
    output_dir: str,
    step: int,
):
    tokens_per_cam_obs = 1 if grid_shape is None else grid_shape[0] * grid_shape[1]
    per_cam_obs = weights_vec.reshape(len(cameras), num_obs, tokens_per_cam_obs)
    global_max = weights_vec.max().item()

    fig, axes = plt.subplots(
        len(cameras), num_obs, figsize=(4 * num_obs, 4 * len(cameras)), squeeze=False
    )
    for cam_i, cam in enumerate(cameras):
        for obs_i in range(num_obs):
            ax = axes[cam_i][obs_i]
            img = images[cam][obs_i]
            token_weights = per_cam_obs[cam_i, obs_i]

            if grid_shape is None:
                ax.imshow(img)
                ax.set_title(
                    f"{cam} | obs t-{num_obs - 1 - obs_i} | w={token_weights.item():.3f}",
                    fontsize=9,
                )
            else:
                h, w = grid_shape
                heat = token_weights.view(h, w)[None, None]
                heat = F.interpolate(
                    heat, size=img.shape[:2], mode="bilinear", align_corners=False
                )[0, 0].numpy()
                heat = heat / max(global_max, 1e-12)
                blended = _overlay_heatmap(img, heat)
                ax.imshow(np.clip(blended, 0, 1))
                ax.set_title(
                    f"{cam} | obs t-{num_obs - 1 - obs_i} | peak={token_weights.max().item():.3f}",
                    fontsize=9,
                )
            ax.axis("off")

    fig.suptitle(f"Decoder block {block_idx} — cross-attn (denoise step {step})")
    fig.tight_layout()
    out_path = os.path.join(output_dir, f"block_{block_idx:02d}_attn.png")
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
    """(action-horizon step) x (camera/obs-step, state obs-step, language)
    heatmap, spatial vision tokens summed out per camera/obs-step — shows
    *when* in the action chunk the policy looks at *which* modality/token
    group, independent of head/spatial layout."""
    w = weights[0].mean(dim=0)  # (Tq, Tkv)
    tq = w.shape[0]

    vision_start, vision_end = layout["vision"]
    vision = w[:, vision_start:vision_end]
    per_cam_obs = vision.reshape(tq, len(cameras) * num_obs, -1).sum(
        dim=-1
    )  # (Tq, cams*obs)
    labels = [f"{cam}\nt-{num_obs - 1 - o}" for cam in cameras for o in range(num_obs)]
    columns = [per_cam_obs]

    state_start, state_end = layout["state"]
    state = w[:, state_start:state_end]  # (Tq, num_obs) — one token per obs-step
    columns.append(state)
    labels += [f"state\nt-{num_obs - 1 - o}" for o in range(num_obs)]

    if "language" in layout:
        lang_start, lang_end = layout["language"]
        language = w[:, lang_start:lang_end].sum(dim=-1, keepdim=True)  # (Tq, 1)
        columns.append(language)
        labels.append("language")

    matrix = torch.cat(columns, dim=-1)  # (Tq, cams*obs + num_obs [+ 1])

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 6))
    im = ax.imshow(matrix.numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("action-horizon query index")
    ax.set_title(f"Decoder block {block_idx} — attention mass per modality/token group")
    fig.colorbar(im, ax=ax, label="summed attention weight")
    fig.tight_layout()
    out_path = os.path.join(
        output_dir, f"block_{block_idx:02d}_query_modality_matrix.png"
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


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

    # Default to the exact episode/frame the checkpoint was overfit on (see
    # overfit_on_batch), so attention here is inspected on a sample the model
    # actually trained on rather than an arbitrary/unseen one — checkpoints
    # not produced by the overfit script (e.g. a full training run) won't
    # have these keys, so an explicit episode_index is required in that case.
    episode_index = cfg.episode_index
    if episode_index is None:
        episode_index = checkpoint.get("episode_index")
    if episode_index is None:
        raise ValueError(
            "episode_index not set and checkpoint has no recorded episode_index "
            "(only overfit checkpoints do) — pass episode_index=... explicitly."
        )
    frame_index = cfg.frame_index
    if frame_index is None:
        frame_index = checkpoint.get("frame_index") or 0
    logger.info("Visualizing episode={} frame={}", episode_index, frame_index)

    # Force single-process loading: we only need one sample, and multi-worker
    # DataLoader + torchcodec video decoding turns any in-worker failure into
    # a garbled/recursive-looking traceback when it's re-raised across the
    # process boundary (see the "Keep at 0 for clean error stack traces!"
    # comment in savannah.data.dataset's own __main__ smoke tests).
    task.config.num_workers = 0
    loader = task.get_episode_loader(episode_index, frame_index)
    raw_batch = next(iter(loader))
    formatted = task.format_batch(raw_batch)  # step=None -> no augmentation
    obs = _select_sample(formatted, idx=0)

    cameras = list(task.config.cameras)
    num_obs = task.config.obs_horizon
    grid_shape = _vision_token_grid_shape(policy.vision_encoder, len(cameras), num_obs)

    # Raw [0,1] pixel images for the overlay, per camera: (num_obs, H, W, 3)
    images = {
        cam: obs[ObservationKey.images][i][0].permute(0, 2, 3, 1).cpu().numpy()
        for i, cam in enumerate(cameras)
    }

    has_language = (
        obs.get(ObservationKey.language) is not None
        and policy.language_encoder is not None
    )

    captured = _register_cross_attn_hooks(policy)
    captured_self = _register_self_attn_hooks(policy)

    with torch.no_grad():
        policy.compute_action(obs)

    os.makedirs(cfg.output_dir, exist_ok=True)

    block_indices = (
        range(len(policy.decoder)) if cfg.block_idx is None else [cfg.block_idx]
    )
    for block_idx in block_indices:
        x, kv = captured[block_idx][cfg.denoise_step]
        weights = _cross_attn_weights(policy.decoder[block_idx].cross_attn_block, x, kv)
        weights = weights.cpu()

        layout = _kv_token_layout(policy, weights.shape[-1], has_language)

        reduced = _reduce_weights(weights, cfg.query_idx, cfg.head_idx)
        entropy = _normalized_entropy(reduced)

        vision_start, vision_end = layout["vision"]
        reduced_vision = reduced[vision_start:vision_end]
        vision_mass = reduced_vision.sum().item()
        vision_entropy = _normalized_entropy(reduced_vision / vision_mass)

        mass_report = ", ".join(
            f"{modality}={reduced[start:end].sum().item():.3f}"
            for modality, (start, end) in layout.items()
        )
        logger.info(
            "block {}: attention mass by modality — {} "
            "(normalized entropy over all {} kv tokens = {:.3f}; "
            "over the {} vision tokens alone = {:.3f})",
            block_idx,
            mass_report,
            reduced.numel(),
            entropy,
            reduced_vision.numel(),
            vision_entropy,
        )

        attn_path = _plot_block(
            block_idx,
            reduced_vision,
            images,
            grid_shape,
            cameras,
            num_obs,
            cfg.output_dir,
            step=len(captured[block_idx]) + cfg.denoise_step
            if cfg.denoise_step < 0
            else cfg.denoise_step,
        )
        matrix_path = _plot_query_modality_matrix(
            block_idx, weights, layout, cameras, num_obs, cfg.output_dir
        )

        x_self = captured_self[block_idx][cfg.denoise_step]
        self_weights = _self_attn_weights(
            policy.decoder[block_idx].self_attn_block, x_self
        )
        self_attn_path, self_entropy = _plot_self_attention(
            block_idx,
            self_weights.cpu(),
            cfg.head_idx,
            cfg.output_dir,
            step=len(captured_self[block_idx]) + cfg.denoise_step
            if cfg.denoise_step < 0
            else cfg.denoise_step,
        )
        logger.info(
            "block {}: self-attn mean per-query entropy = {:.3f} "
            "(0 = each action-step attends to a specific horizon position, "
            "1 = uniform over all {} positions)",
            block_idx,
            self_entropy,
            self_weights.shape[-1],
        )
        logger.info("Wrote {}, {} and {}", attn_path, matrix_path, self_attn_path)

    logger.info("Done. Outputs in {}", cfg.output_dir)


@hydra.main(version_base=None, config_path="../../configs", config_name="attention_viz")
def main(cfg: DictConfig) -> None:
    run_attention_viz(cfg)


if __name__ == "__main__":
    main()
