import pytest
import torch

from savannah.models.backbones.dinov3_backbone import DinoV3Backbone

B = 2
IMAGE_SIZE = 224
BASE_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"


@pytest.fixture
def backbone():
    torch.manual_seed(0)
    return DinoV3Backbone(
        image_size=IMAGE_SIZE,
        base_model=BASE_MODEL,
        trainable=False,
    )


def test_output_shape(backbone):
    x = torch.randn(B, 3, IMAGE_SIZE, IMAGE_SIZE)
    out = backbone(x)
    assert out.shape == (B, 1, backbone.out_channels)


def test_distinguishes_solid_color_images(backbone):
    # Regression test: dataset images are float32 in [0, 1] (see
    # savannah/data/dataset.py), but the image processor defaults to
    # do_rescale=True (divide by 255), which assumes raw [0, 255] input.
    # Without do_rescale=False in forward(), pixel values get rescaled twice
    # and collapse into a narrow band near one corner of the model's
    # normalized input range, making clearly different images nearly
    # indistinguishable.
    backbone.eval()
    red = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    red[:, 0] = 1.0
    blue = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    blue[:, 2] = 1.0

    with torch.no_grad():
        red_features = backbone(red)
        blue_features = backbone(blue)

    cos_sim = torch.nn.functional.cosine_similarity(
        red_features.flatten(1), blue_features.flatten(1)
    ).item()
    assert cos_sim < 0.98
