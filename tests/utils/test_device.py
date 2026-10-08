import pytest
import torch

from savannah.utils import device as device_utils
from savannah.utils.device import configure_mujoco_gl, get_device


@pytest.fixture
def no_mujoco_gl(monkeypatch):
    monkeypatch.delenv("MUJOCO_GL", raising=False)


@pytest.mark.parametrize("system", ["Darwin", "Windows"])
def test_mujoco_gl_left_unset_off_linux(monkeypatch, no_mujoco_gl, system):
    # EGL/osmesa are Linux-only; MuJoCo raises on import if they're set on
    # macOS/Windows, so nothing may be set there.
    monkeypatch.setattr(device_utils.platform, "system", lambda: system)
    configure_mujoco_gl()
    assert "MUJOCO_GL" not in device_utils.os.environ


@pytest.mark.parametrize(("cuda", "expected"), [(True, "egl"), (False, "osmesa")])
def test_mujoco_gl_on_linux(monkeypatch, no_mujoco_gl, cuda, expected):
    monkeypatch.setattr(device_utils.platform, "system", lambda: "Linux")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    configure_mujoco_gl()
    assert device_utils.os.environ["MUJOCO_GL"] == expected


def test_mujoco_gl_respects_existing_value(monkeypatch):
    monkeypatch.setenv("MUJOCO_GL", "glfw")
    monkeypatch.setattr(device_utils.platform, "system", lambda: "Linux")
    configure_mujoco_gl()
    assert device_utils.os.environ["MUJOCO_GL"] == "glfw"


def test_savannah_device_overrides_autodetect(monkeypatch):
    monkeypatch.setenv("SAVANNAH_DEVICE", "cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert get_device() == torch.device("cpu")
