import pytest
import torch
import torch.nn as nn

from savannah.models.encoders.vision_encoder import VisionEncoder

B, H, W = 2, 32, 32
BACKBONE_OUT, EMBED_DIM = 16, 32
PROJ_EMBED_DIM = 24
NUM_CAMERAS, NUM_OBS = 2, 2
SPATIAL_TOKENS = 4


class PoolingStubBackbone(nn.Module):
    """Lightweight pooling backbone stand-in (tokens_per_image=1, 2D output)."""

    def __init__(self, out_channels: int = BACKBONE_OUT):
        super().__init__()
        self._out_channels = out_channels
        self.conv = nn.Conv2d(3, out_channels, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def tokens_per_image(self) -> int:
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 3, H, W) -> (B, out_channels)
        return self.pool(self.conv(x)).flatten(1)


class SpatialStubBackbone(nn.Module):
    """Lightweight multi-token backbone stand-in (tokens_per_image>1, 3D output)."""

    def __init__(
        self, out_channels: int = BACKBONE_OUT, tokens_per_image: int = SPATIAL_TOKENS
    ):
        super().__init__()
        self._out_channels = out_channels
        self._tokens_per_image = tokens_per_image
        self.conv = nn.Conv2d(3, out_channels, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(int(tokens_per_image**0.5))

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def tokens_per_image(self) -> int:
        return self._tokens_per_image

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 3, H, W) -> (B, tokens_per_image, out_channels)
        features = self.pool(self.conv(x))
        b, c, h, w = features.shape
        return features.view(b, c, h * w).transpose(1, 2)


def make_images(num_cameras=NUM_CAMERAS, num_obs=NUM_OBS, batch=B):
    return [torch.randn(batch, num_obs, 3, H, W) for _ in range(num_cameras)]


@pytest.fixture
def pooling_backbone():
    return PoolingStubBackbone()


@pytest.fixture
def spatial_backbone():
    return SpatialStubBackbone()


@pytest.fixture
def vision_encoder_pooling(pooling_backbone):
    return VisionEncoder(
        pooling_backbone,
        embed_dim=EMBED_DIM,
        num_cameras=NUM_CAMERAS,
        num_obs=NUM_OBS,
    )


@pytest.fixture
def vision_encoder_spatial(spatial_backbone):
    return VisionEncoder(
        spatial_backbone,
        embed_dim=EMBED_DIM,
        num_cameras=NUM_CAMERAS,
        num_obs=NUM_OBS,
    )


@pytest.fixture
def vision_encoder_with_proj(pooling_backbone):
    return VisionEncoder(
        pooling_backbone,
        embed_dim=PROJ_EMBED_DIM,
        num_cameras=NUM_CAMERAS,
        num_obs=NUM_OBS,
    )


@pytest.fixture
def vision_encoder_identity_proj(pooling_backbone):
    return VisionEncoder(
        pooling_backbone,
        embed_dim=BACKBONE_OUT,
        num_cameras=NUM_CAMERAS,
        num_obs=NUM_OBS,
    )


def test_num_tokens_property_pooling(vision_encoder_pooling):
    assert vision_encoder_pooling.num_tokens == NUM_CAMERAS * NUM_OBS * 1


def test_num_tokens_property_spatial(vision_encoder_spatial):
    assert vision_encoder_spatial.num_tokens == NUM_CAMERAS * NUM_OBS * SPATIAL_TOKENS


def test_proj_is_identity_when_dims_match(vision_encoder_identity_proj):
    assert isinstance(vision_encoder_identity_proj.proj, nn.Identity)


def test_proj_is_linear_when_dims_differ(vision_encoder_with_proj):
    assert isinstance(vision_encoder_with_proj.proj, nn.Linear)
    assert vision_encoder_with_proj.proj.in_features == BACKBONE_OUT
    assert vision_encoder_with_proj.proj.out_features == PROJ_EMBED_DIM


def test_extract_tokens_from_image_unsqueezes_pooled_features(
    vision_encoder_pooling,
):
    x = torch.randn(B, 3, H, W)
    tokens = vision_encoder_pooling.extract_tokens_from_image(x)
    assert tokens.shape == (B, 1, BACKBONE_OUT)


def test_forward_output_shape_pooling_backbone(vision_encoder_pooling):
    images = make_images()
    out = vision_encoder_pooling(images)
    assert out.shape == (B, vision_encoder_pooling.num_tokens, EMBED_DIM)


def test_forward_output_shape_spatial_backbone(vision_encoder_spatial):
    images = make_images()
    out = vision_encoder_spatial(images)
    assert out.shape == (B, vision_encoder_spatial.num_tokens, EMBED_DIM)


def test_forward_output_shape_with_projection(vision_encoder_with_proj):
    images = make_images()
    out = vision_encoder_with_proj(images)
    assert out.shape[-1] == PROJ_EMBED_DIM


def test_forward_wrong_num_cameras_raises(vision_encoder_pooling):
    images = make_images(num_cameras=NUM_CAMERAS + 1)
    with pytest.raises(AssertionError):
        vision_encoder_pooling(images)


def test_forward_wrong_num_obs_raises(vision_encoder_pooling):
    images = make_images(num_obs=NUM_OBS + 1)
    with pytest.raises(AssertionError):
        vision_encoder_pooling(images)


def test_forward_no_nan_or_inf(vision_encoder_pooling):
    images = make_images()
    out = vision_encoder_pooling(images)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_forward_gradient_flow(vision_encoder_pooling):
    vision_encoder_pooling.train()
    images = make_images()
    out = vision_encoder_pooling(images)
    out.sum().backward()

    missing_grad = [
        name
        for name, p in vision_encoder_pooling.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not missing_grad, f"parameters with no gradient: {missing_grad}"


def test_camera_embedding_differs_per_camera(vision_encoder_pooling):
    vision_encoder_pooling.eval()
    image = torch.randn(B, NUM_OBS, 3, H, W)
    images = [image, image.clone()]
    out = vision_encoder_pooling(images)

    tokens_per_cam = vision_encoder_pooling.num_tokens // NUM_CAMERAS
    cam0_tokens = out[:, :tokens_per_cam, :]
    cam1_tokens = out[:, tokens_per_cam:, :]
    assert not torch.allclose(cam0_tokens, cam1_tokens)


def test_history_embedding_differs_per_obs(vision_encoder_pooling):
    vision_encoder_pooling.eval()
    frame = torch.randn(B, 1, 3, H, W)
    image = frame.repeat(1, NUM_OBS, 1, 1, 1)
    images = [image, image.clone()]
    out = vision_encoder_pooling(images)

    tokens_per_cam = vision_encoder_pooling.num_tokens // NUM_CAMERAS
    cam0_tokens = out[:, :tokens_per_cam, :]
    obs0_token = cam0_tokens[:, 0, :]
    obs1_token = cam0_tokens[:, 1, :]
    assert not torch.allclose(obs0_token, obs1_token)


def test_eval_mode_deterministic(vision_encoder_pooling):
    vision_encoder_pooling.eval()
    images = make_images()
    out1 = vision_encoder_pooling(images)
    out2 = vision_encoder_pooling(images)
    assert torch.equal(out1, out2)
