import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _decay(initial: float, final: float, step: int, decay_rate: float) -> float:
    """Exponential decay from `initial` toward `final` as `step` grows.

    `decay_rate` is the characteristic timescale in steps — after
    `decay_rate` steps the gap to `final` has shrunk by ~63%.
    """
    if decay_rate <= 0:
        return final
    weight = math.exp(-step / decay_rate)
    return final + (initial - final) * weight


@dataclass
class AugmentationConfig:
    enabled: bool = False

    # Shared exponential decay timescale (in training steps) for all knobs below.
    decay_rate: float = 5000.0

    # Random crop: each sample is cropped to a random `scale` fraction of its
    # original H/W (uniform in [scale_min(step), 1.0]) then resized back to
    # the original size, so output shape always matches input shape.
    crop_scale_min_initial: float = 0.8
    crop_scale_min_final: float = 1.0

    # Per-sample brightness/contrast jitter strength (fractional).
    color_jitter_initial: float = 0.3
    color_jitter_final: float = 0.0

    # Multiplicative (speckle) pixel noise std, applied as x + x * N(0, std).
    speckle_std_initial: float = 0.05
    speckle_std_final: float = 0.0

    # Additive uniform noise on (normalized) state vectors, magnitude +/- eps.
    state_noise_initial: float = 0.01
    state_noise_final: float = 0.0


class ObservationAugmenter:
    """Train-time image/state augmentation with a shared decay schedule.

    Aggressiveness is highest at step 0 and exponentially decays toward the
    `*_final` bounds, so early training (when the model most needs robustness
    to limited/noisy data) sees the strongest augmentation, while late
    training converges on clean observations.

    Meant to be called only for training batches — pass `step=None` (or don't
    call at all) for validation/eval, where these methods are no-ops.
    """

    def __init__(self, config: AugmentationConfig):
        self.config = config

    def augment_images(
        self, images: list[torch.Tensor], step: int
    ) -> list[torch.Tensor]:
        """images: list of (B, N, C, H, W) tensors, one per camera, float in [0, 1]."""
        if not self.config.enabled:
            return images
        return [self._augment_camera(img, step) for img in images]

    def augment_state(self, state: torch.Tensor, step: int) -> torch.Tensor:
        if not self.config.enabled:
            return state
        cfg = self.config
        eps = _decay(
            cfg.state_noise_initial, cfg.state_noise_final, step, cfg.decay_rate
        )
        if eps <= 0:
            return state
        noise = (torch.rand_like(state) * 2 - 1) * eps
        return state + noise

    def _augment_camera(self, img: torch.Tensor, step: int) -> torch.Tensor:
        B, N, C, H, W = img.shape
        x = img.reshape(B * N, C, H, W)

        x = self._random_crop_resize(x, B, N, step)
        x = self._color_jitter(x, B, N, step)
        x = self._speckle_noise(x, step)

        return x.reshape(B, N, C, H, W).clamp(0.0, 1.0)

    def _random_crop_resize(
        self, x: torch.Tensor, B: int, N: int, step: int
    ) -> torch.Tensor:
        """Crops each of the B samples to a random `scale` fraction of H/W (same
        crop reused across that sample's N obs-horizon frames for temporal
        consistency, independent across samples/cameras), then resizes back
        to (H, W) via a single vectorized grid_sample — output shape == input
        shape.
        """
        cfg = self.config
        scale_min = _decay(
            cfg.crop_scale_min_initial, cfg.crop_scale_min_final, step, cfg.decay_rate
        )
        if scale_min >= 1.0:
            return x

        device, dtype = x.device, x.dtype
        scales = torch.empty(B, device=device, dtype=dtype).uniform_(scale_min, 1.0)

        # Crop center may range over +/-(1 - scale) in normalized [-1, 1]
        # coords so the crop box stays fully inside the image.
        max_offset = 1.0 - scales
        offset_x = (torch.rand(B, device=device, dtype=dtype) * 2 - 1) * max_offset
        offset_y = (torch.rand(B, device=device, dtype=dtype) * 2 - 1) * max_offset

        theta = torch.zeros(B, 2, 3, device=device, dtype=dtype)
        theta[:, 0, 0] = scales
        theta[:, 1, 1] = scales
        theta[:, 0, 2] = offset_x
        theta[:, 1, 2] = offset_y

        theta = theta.repeat_interleave(N, dim=0)
        grid = F.affine_grid(theta, size=x.shape, align_corners=False)
        return F.grid_sample(x, grid, align_corners=False, padding_mode="reflection")

    def _color_jitter(self, x: torch.Tensor, B: int, N: int, step: int) -> torch.Tensor:
        """Per-sample brightness + contrast jitter (same across a sample's N
        frames, independent across samples/cameras)."""
        cfg = self.config
        strength = _decay(
            cfg.color_jitter_initial, cfg.color_jitter_final, step, cfg.decay_rate
        )
        if strength <= 0:
            return x

        device, dtype = x.device, x.dtype
        brightness = torch.empty(B, device=device, dtype=dtype).uniform_(
            1 - strength, 1 + strength
        )
        contrast = torch.empty(B, device=device, dtype=dtype).uniform_(
            1 - strength, 1 + strength
        )
        brightness = brightness.repeat_interleave(N).view(-1, 1, 1, 1)
        contrast = contrast.repeat_interleave(N).view(-1, 1, 1, 1)

        x = x * brightness
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        return (x - mean) * contrast + mean

    def _speckle_noise(self, x: torch.Tensor, step: int) -> torch.Tensor:
        """Simulates camera sensor grain: multiplicative per-pixel Gaussian noise."""
        cfg = self.config
        std = _decay(
            cfg.speckle_std_initial, cfg.speckle_std_final, step, cfg.decay_rate
        )
        if std <= 0:
            return x
        noise = torch.randn_like(x) * std
        return x + x * noise
