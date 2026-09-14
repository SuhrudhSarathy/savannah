import torch
import torch.nn as nn
from transformers import AutoProcessor


class ImageNormalizer(nn.Module):
    """GPU-native, differentiable image normalization layer.

    Extracts mean and std stats from a Hugging Face processor or config in __init__,
    preserving autograd history on the input tensor during forward().
    """

    def __init__(
        self,
        base_model_name_or_path: str,
        fallback_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        fallback_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        mean, std = None, None
        try:
            processor = AutoProcessor.from_pretrained(base_model_name_or_path)
            img_proc = getattr(processor, "image_processor", processor)
            mean = getattr(img_proc, "image_mean", None)
            std = getattr(img_proc, "image_std", None)
        except Exception:
            pass

        if mean is None:
            mean = list(fallback_mean)
        if std is None:
            std = list(fallback_std)

        self.register_buffer(
            "mean",
            torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes input image tensor in-place on the computation graph.

        Args:
            x: Image tensor of shape (B, 3, H, W), expected in range [0.0, 1.0].

        Returns:
            Normalized tensor with unbroken autograd graph.
        """
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std
