"""Pretrained policy checkpoints: saving, loading, and resolving them locally or
on the Hugging Face Hub.

A checkpoint is a directory, laid out the same way on local disk and inside a
Hub model repo (one repo per run, one subfolder per checkpoint kind:
`latest/`, `best_model/`, `best_model_success/`):

    config.yaml         fully resolved run config (model + task + name)
    model.safetensors   EMA weights
    manifest.json       format version, savannah version, git hash, step,
                        metrics, obs/action specs, normalization stats
    training_state.pt   LOCAL ONLY -- raw weights, optimizer, scheduler, step,
                        ... (needed to resume training or for use_ema=false;
                        never pushed to the Hub)

Loading follows the `transformers` `from_pretrained` conventions:

    policy = build_policy_from_pretrained(
        "suhrudhsarathy/unet_pusht",      # Hub repo id, or a local directory
        subfolder="best_model",           # which checkpoint in the run
        revision="best_model-step-15000", # optional branch / tag / commit
        device="cuda",
    )

Saves are atomic: the checkpoint is written to a temporary sibling directory
and swapped in with renames, so a crash mid-save leaves the previous
checkpoint untouched.
"""

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id
from omegaconf import DictConfig, OmegaConf, open_dict
from safetensors.torch import load_file, save_file

from savannah.factory import build_policy
from savannah.models.policy import Policy
from savannah.tasks import BaseRobotTask
from savannah.utils.log import logger

FORMAT_VERSION = 2

CONFIG_FILE = "config.yaml"
WEIGHTS_FILE = "model.safetensors"
MANIFEST_FILE = "manifest.json"
TRAINING_STATE_FILE = "training_state.pt"

# Written next to the checkpoint locally but never uploaded to the Hub.
LOCAL_ONLY_FILES = frozenset({TRAINING_STATE_FILE})

DEFAULT_SUBFOLDER = "best_model"


@dataclass
class CheckpointContents:
    """Everything in a checkpoint directory, without the policy built."""

    path: Path
    manifest: dict[str, Any]
    config: DictConfig
    ema_weights: dict[str, torch.Tensor]
    training_state: dict[str, Any] | None


@dataclass
class PretrainedPolicy:
    policy: Policy
    config: DictConfig
    manifest: dict[str, Any]
    training_state: dict[str, Any] | None
    path: Path
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


def _savannah_version() -> str | None:
    try:
        return metadata.version("savannah")
    except metadata.PackageNotFoundError:
        return None


def _git_hash() -> str | None:
    """HEAD of the checkout this code runs from, suffixed "-dirty" if it has
    uncommitted changes. None when not running from a git checkout."""
    repo = Path(__file__).resolve().parent

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        return f"{commit}-dirty" if git("status", "--porcelain") else commit
    except (OSError, subprocess.CalledProcessError):
        return None


def _build_manifest(
    config: dict,
    task: BaseRobotTask,
    step: int | None,
    metrics: dict[str, float] | None,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    model, data = config["model"], config["task"]["config"]
    state_keys = ["observation.state"]
    if data.get("use_language", False):
        state_keys.append("language")
    low_high = task.action_limits()
    return {
        "format_version": FORMAT_VERSION,
        "savannah_version": _savannah_version(),
        "git_hash": _git_hash(),
        "step": step,
        "metrics": metrics or {},
        "extra": extra or {},
        "obs_spec": {
            "cameras": list(data["cameras"]),
            "image_size": data.get("image_size"),
            "obs_horizon": data["obs_horizon"],
            "state_keys": state_keys,
            "state_dim": model["state_dim"],
        },
        "action_spec": {
            "dim": model["action_dim"],
            "space": task.action_space,
            "rate_hz": data["fps"],
            "horizon": model["action_horizon"],
            "limits": None
            if low_high is None
            else {"low": low_high[0], "high": low_high[1]},
        },
        "normalization": task.normalization_stats(),
    }


# --------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------


def _fsync_file(path: Path) -> None:
    with open(path, "rb") as f:
        os.fsync(f.fileno())


def _fsync_dir(path: Path) -> None:
    dir_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _swap_in(tmp_dir: Path, path: Path) -> None:
    """Moves the finished `tmp_dir` to `path`, replacing any existing
    checkpoint there. If the swap fails part-way, the old checkpoint is put
    back."""
    old_dir = None
    if path.exists():
        old_dir = Path(
            tempfile.mkdtemp(dir=path.parent, prefix=f".{path.name}.", suffix=".old")
        )
        old_dir.rmdir()  # only reserve the name; os.replace needs it absent
        os.replace(path, old_dir)
    try:
        os.replace(tmp_dir, path)
    except BaseException:
        if old_dir is not None:
            os.replace(old_dir, path)
        raise
    if old_dir is not None:
        shutil.rmtree(old_dir, ignore_errors=True)


def save_pretrained(
    save_directory: str | Path,
    cfg: DictConfig,
    task: BaseRobotTask,
    ema_state_dict: dict[str, torch.Tensor],
    *,
    training_state: dict[str, Any] | None = None,
    step: int | None = None,
    metrics: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Atomically writes a checkpoint directory to `save_directory`,
    replacing any existing one. `cfg` is the composed run config (must contain
    `model` and `task`) and is stored fully resolved. `training_state` stays
    local: Hub uploads skip it."""
    path = Path(save_directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = OmegaConf.to_container(cfg, resolve=True)
    manifest = _build_manifest(config, task, step, metrics, extra)

    tmp_dir = Path(
        tempfile.mkdtemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    )
    try:
        (tmp_dir / CONFIG_FILE).write_text(OmegaConf.to_yaml(OmegaConf.create(config)))
        # copy=True: safetensors refuses tensors that share storage (tied
        # weights, or CPU tensors still aliasing the live model).
        save_file(
            {
                k: v.detach().to("cpu", copy=True).contiguous()
                for k, v in ema_state_dict.items()
            },
            tmp_dir / WEIGHTS_FILE,
            metadata={"format": "pt"},
        )
        if training_state is not None:
            torch.save(training_state, tmp_dir / TRAINING_STATE_FILE)
        (tmp_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n")

        for f in tmp_dir.iterdir():
            _fsync_file(f)
        os.chmod(tmp_dir, 0o755)
        _swap_in(tmp_dir, path)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    # Persist the renames themselves across a power loss.
    _fsync_dir(path.parent)
    return path


# --------------------------------------------------------------------------
# Resolving & reading
# --------------------------------------------------------------------------


def _is_checkpoint_dir(path: Path) -> bool:
    return (path / CONFIG_FILE).is_file() and (path / MANIFEST_FILE).is_file()


def resolve_pretrained_dir(
    pretrained_name_or_path: str | Path,
    *,
    subfolder: str | None = DEFAULT_SUBFOLDER,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
) -> Path:
    """The local checkpoint directory for `pretrained_name_or_path`.

    - A local checkpoint directory is used as is (`subfolder` is ignored).
    - A local run directory (e.g. `checkpoints/<run>`) resolves to
      `<run>/<subfolder>`.
    - Anything else is treated as a Hub repo id: only `<subfolder>/` at
      `revision` is downloaded (cached by huggingface_hub).
    """
    local = Path(pretrained_name_or_path).expanduser()
    if local.exists():
        if _is_checkpoint_dir(local):
            return local
        candidate = local / subfolder if subfolder else local
        if _is_checkpoint_dir(candidate):
            return candidate
        available = sorted(p.name for p in local.iterdir() if _is_checkpoint_dir(p))
        raise FileNotFoundError(
            f"No checkpoint at {candidate}. "
            + (
                f"Checkpoints in {local}: {available}"
                if available
                else f"{local} contains no checkpoints"
            )
        )

    repo_id = str(pretrained_name_or_path)
    try:
        validate_repo_id(repo_id)
    except HFValidationError as e:
        raise FileNotFoundError(
            f"{repo_id} is neither an existing local path nor a valid Hub repo id"
        ) from e

    logger.info(
        "Fetching {}{} from the Hub (revision={})",
        repo_id,
        f"/{subfolder}" if subfolder else "",
        revision or "main",
    )
    snapshot = Path(
        snapshot_download(
            repo_id,
            revision=revision,
            allow_patterns=[f"{subfolder}/*"] if subfolder else None,
            cache_dir=cache_dir,
            token=token,
            library_name="savannah",
            library_version=_savannah_version(),
        )
    )
    resolved = snapshot / subfolder if subfolder else snapshot
    if not _is_checkpoint_dir(resolved):
        raise FileNotFoundError(
            f"{repo_id} (revision={revision or 'main'}) has no checkpoint "
            f"in {subfolder or 'the repo root'!r}"
        )
    return resolved


def read_checkpoint(path: str | Path) -> CheckpointContents:
    """Reads a local checkpoint directory without building the policy.
    Tensors are loaded onto the CPU."""
    path = Path(path)
    if not _is_checkpoint_dir(path):
        raise FileNotFoundError(f"No checkpoint at {path}")

    manifest = json.loads((path / MANIFEST_FILE).read_text())
    version = manifest.get("format_version")
    if not isinstance(version, int):
        raise ValueError(f"{path / MANIFEST_FILE} has no valid format_version")
    if version > FORMAT_VERSION:
        raise ValueError(
            f"Checkpoint format {version} is newer than this savannah supports "
            f"({FORMAT_VERSION}); upgrade savannah to load {path}"
        )

    training_state_path = path / TRAINING_STATE_FILE
    return CheckpointContents(
        path=path,
        manifest=manifest,
        config=OmegaConf.load(path / CONFIG_FILE),
        ema_weights=load_file(path / WEIGHTS_FILE, device="cpu"),
        training_state=torch.load(
            training_state_path, map_location="cpu", weights_only=True
        )
        if training_state_path.is_file()
        else None,
    )


# --------------------------------------------------------------------------
# Loading weights / building policies
# --------------------------------------------------------------------------


def _select_weights(
    contents: CheckpointContents, use_ema: bool
) -> dict[str, torch.Tensor]:
    if use_ema:
        return contents.ema_weights
    if contents.training_state is None:
        raise ValueError(
            f"{contents.path} has only EMA weights (training state is never "
            "pushed to the Hub); pass use_ema=true or load a local checkpoint"
        )
    return contents.training_state["model_state_dict"]


def _load_weights(
    policy: torch.nn.Module,
    weights: dict[str, torch.Tensor],
    *,
    ignore_mismatched_sizes: bool,
    source: Path,
) -> None:
    if not ignore_mismatched_sizes:
        policy.load_state_dict(weights)
        return

    own = policy.state_dict()
    mismatched = {
        k: (tuple(v.shape), tuple(own[k].shape))
        for k, v in weights.items()
        if k in own and v.shape != own[k].shape
    }
    weights = {k: v for k, v in weights.items() if k not in mismatched}
    result = policy.load_state_dict(weights, strict=False)

    for k, (ckpt_shape, model_shape) in mismatched.items():
        logger.warning(
            "{}: skipped {} (checkpoint {} vs model {}) -- newly initialized",
            source,
            k,
            ckpt_shape,
            model_shape,
        )
    if result.missing_keys:
        logger.warning(
            "{}: {} keys not in checkpoint, newly initialized: {}",
            source,
            len(result.missing_keys),
            result.missing_keys,
        )
    if result.unexpected_keys:
        logger.warning(
            "{}: {} checkpoint keys unused by the model: {}",
            source,
            len(result.unexpected_keys),
            result.unexpected_keys,
        )


def load_pretrained_weights(
    policy: torch.nn.Module,
    pretrained_name_or_path: str | Path,
    *,
    subfolder: str | None = DEFAULT_SUBFOLDER,
    revision: str | None = None,
    use_ema: bool = True,
    ignore_mismatched_sizes: bool = True,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
) -> torch.nn.Module:
    """Loads pretrained weights into an already-built `policy`, e.g. to
    fine-tune a policy whose config differs from the pretrained one (a new
    action head, extra cameras, ...). Keys whose shapes don't match are
    skipped and logged unless `ignore_mismatched_sizes=False`."""
    path = resolve_pretrained_dir(
        pretrained_name_or_path,
        subfolder=subfolder,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
    )
    contents = read_checkpoint(path)
    _load_weights(
        policy,
        _select_weights(contents, use_ema),
        ignore_mismatched_sizes=ignore_mismatched_sizes,
        source=path,
    )
    return policy


def load_pretrained(
    pretrained_name_or_path: str | Path,
    *,
    subfolder: str | None = DEFAULT_SUBFOLDER,
    revision: str | None = None,
    device: torch.device | str = "cpu",
    use_ema: bool = True,
    ignore_mismatched_sizes: bool = False,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
    **model_kwargs: Any,
) -> PretrainedPolicy:
    """Rebuilds a policy, in eval mode on `device`, from a checkpoint on the
    Hub or on local disk, along with the config and manifest it was saved
    with.

    `model_kwargs` override fields of the stored `model` config before the
    policy is built (nested dicts merge), like config kwargs in
    `transformers`' `from_pretrained`. Combine with
    `ignore_mismatched_sizes=True` when an override changes parameter shapes.
    """
    path = resolve_pretrained_dir(
        pretrained_name_or_path,
        subfolder=subfolder,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
    )
    contents = read_checkpoint(path)
    config = contents.config
    if model_kwargs:
        with open_dict(config):
            config.model = OmegaConf.merge(config.model, model_kwargs)

    policy = build_policy(config)
    _load_weights(
        policy,
        _select_weights(contents, use_ema),
        ignore_mismatched_sizes=ignore_mismatched_sizes,
        source=path,
    )
    policy.to(device).eval()
    return PretrainedPolicy(
        policy=policy,
        config=config,
        manifest=contents.manifest,
        training_state=contents.training_state,
        path=path,
        extra=contents.manifest.get("extra", {}),
    )


def build_policy_from_pretrained(
    pretrained_name_or_path: str | Path, **kwargs: Any
) -> Policy:
    """A policy, in eval mode, from a Hub repo id or local checkpoint. Takes
    the same keyword arguments as `load_pretrained`."""
    return load_pretrained(pretrained_name_or_path, **kwargs).policy


# --------------------------------------------------------------------------
# Hydra script helpers
# --------------------------------------------------------------------------


def apply_pretrained_config(cfg: DictConfig, pretrained: PretrainedPolicy) -> None:
    """Fills a script's `cfg` with the `model`, `task` and `name` the
    checkpoint was trained with."""
    with open_dict(cfg):
        cfg.model = pretrained.config.model
        cfg.task = pretrained.config.task
        cfg.name = pretrained.config.get("name", "eval")


def load_pretrained_from_cfg(
    cfg: DictConfig, device: torch.device | str
) -> PretrainedPolicy:
    """Loads `cfg.pretrained` (at `cfg.subfolder` / `cfg.revision`) for an
    eval/debug script and fills `cfg` with its model and task config."""
    if not cfg.pretrained:
        raise ValueError(
            "Set pretrained=<Hub repo id or local checkpoint dir>, "
            "e.g. pretrained=suhrudhsarathy/unet_pusht"
        )
    pretrained = load_pretrained(
        cfg.pretrained,
        subfolder=cfg.subfolder,
        revision=cfg.revision,
        device=device,
        use_ema=cfg.use_ema,
    )
    logger.info("Loaded checkpoint from {}", pretrained.path)
    apply_pretrained_config(cfg, pretrained)
    return pretrained
