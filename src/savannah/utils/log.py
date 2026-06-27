import logging
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

_CONSOLE_FMT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>"
)

_FILE_FMT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}"


class _InterceptHandler(logging.Handler):
    """Routes stdlib logging (hydra, torch, etc.) into loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(
    log_dir: str | Path | None = None, level: str = "INFO"
) -> Path | None:
    """Call once at process startup.

    - Colored stderr sink at `level` (default INFO).
    - Rotating file sink at DEBUG in `log_dir` (captures everything including
      tensor debug stats, hydra config dumps, torch warnings).
    - Stdlib logging interceptor so hydra/torch/lerobot logs land in the same
      stream rather than writing directly to stderr.

    Returns the log file path, or None if log_dir was not provided.
    """
    logger.remove()

    logger.add(
        sys.stderr, format=_CONSOLE_FMT, level=level, colorize=True, enqueue=True
    )

    log_path: Path | None = None
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_path = Path(log_dir) / f"run_{ts}.log"
        logger.add(
            str(log_path),
            format=_FILE_FMT,
            level="DEBUG",
            rotation="50 MB",
            retention=10,
            enqueue=True,
            colorize=False,
        )

    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for name in list(logging.root.manager.loggerDict):
        log = logging.getLogger(name)
        log.handlers = [_InterceptHandler()]
        log.propagate = False

    return log_path


__all__ = ["logger", "setup_logging"]
