import os
import platform

import torch

from savannah.utils.log import logger


def get_device():
    if "SAVANNAH_DEVICE" in os.environ:
        device = torch.device(os.environ["SAVANNAH_DEVICE"])
        logger.info("Using device: {} (from SAVANNAH_DEVICE)", device)
        return device

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
        torch.mps.empty_cache()
        torch.mps.synchronize()

    logger.info("Using device: {}", device)
    return device


def configure_mujoco_gl():
    """Pick a MuJoCo offscreen rendering backend before `mujoco` is imported.

    GLFW (mujoco's default) needs a real X11 display and fails with
    `gladLoadGL error` on headless GPU servers. EGL renders offscreen via
    the GPU without a display; osmesa is the software fallback for
    CPU-only machines. Must run before `import mujoco`, since the backend
    is selected from MUJOCO_GL at import time.

    EGL and osmesa only exist on Linux (MuJoCo rejects them elsewhere); on
    macOS/Windows MuJoCo's default already renders offscreen without a
    display, so MUJOCO_GL is left unset there.
    """
    if "MUJOCO_GL" in os.environ or platform.system() != "Linux":
        return
    os.environ["MUJOCO_GL"] = "egl" if torch.cuda.is_available() else "osmesa"
    logger.info("Set MUJOCO_GL={}", os.environ["MUJOCO_GL"])
