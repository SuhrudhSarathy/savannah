"""A policy bundle is the one on-disk format for trained policies: a single
`.bundle` file, written during training and read by eval and resume alike.

It is a zip archive holding:

    bundle.json        format version, savannah version, git hash, obs/action
                       specs, normalization stats, SHA-256 of the other members
    config.yaml        fully resolved run config
    weights.pt         EMA weights
    training_state.pt  raw weights, optimizer, scheduler, step, ... (optional,
                       needed only to resume training or use_ema=false)

Saves are atomic: the file is fully written and flushed to disk under a
temporary name, then renamed over `path`, so a crash mid-save leaves any
previous bundle untouched. Every load verifies the checksums and raises
BundleCorruptError rather than returning damaged weights.
"""

import glob
import hashlib
import io
import json
import os
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf, open_dict

import wandb
from savannah.factory import build_policy
from savannah.models.policy import Policy
from savannah.tasks import BaseRobotTask
from savannah.utils.log import logger

BUNDLE_FORMAT_VERSION = 1
BUNDLE_SUFFIX = ".bundle"

MANIFEST_FILE = "bundle.json"
CONFIG_FILE = "config.yaml"
WEIGHTS_FILE = "weights.pt"
TRAINING_STATE_FILE = "training_state.pt"


class BundleCorruptError(ValueError):
    """The file is truncated, damaged, or not a bundle at all."""


@dataclass
class BundleContents:
    manifest: dict[str, Any]
    config: DictConfig
    ema_weights: dict[str, torch.Tensor]
    training_state: dict[str, Any] | None


@dataclass
class PolicyBundle:
    policy: Policy
    config: DictConfig
    manifest: dict[str, Any]
    training_state: dict[str, Any] | None


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


def _build_manifest(config: dict, task: BaseRobotTask) -> dict[str, Any]:
    model, data = config["model"], config["task"]["config"]
    state_keys = ["observation.state"]
    if data.get("use_language", False):
        state_keys.append("language")
    low_high = task.action_limits()
    return {
        "format_version": BUNDLE_FORMAT_VERSION,
        "savannah_version": _savannah_version(),
        "git_hash": _git_hash(),
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


def _torch_bytes(obj: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.getvalue()


def _write_atomic(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as f:
            # Stored, not deflated: weights barely compress, and saves happen
            # inside the training loop.
            with zipfile.ZipFile(f, "w", zipfile.ZIP_STORED) as archive:
                for name, data in members.items():
                    archive.writestr(name, data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise

    # Persist the rename itself across a power loss.
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def save_bundle(
    path: str | Path,
    cfg: DictConfig,
    task: BaseRobotTask,
    ema_state_dict: dict[str, torch.Tensor],
    training_state: dict[str, Any] | None = None,
) -> Path:
    """Atomically writes a bundle to `path`, replacing any existing one. `cfg`
    is the composed run config (must contain `model` and `task`) and is stored
    fully resolved."""
    path = Path(path)
    config = OmegaConf.to_container(cfg, resolve=True)

    members = {
        CONFIG_FILE: OmegaConf.to_yaml(OmegaConf.create(config)).encode(),
        WEIGHTS_FILE: _torch_bytes(
            {k: v.detach().cpu() for k, v in ema_state_dict.items()}
        ),
    }
    if training_state is not None:
        members[TRAINING_STATE_FILE] = _torch_bytes(training_state)

    manifest = _build_manifest(config, task)
    manifest["sha256"] = {
        name: hashlib.sha256(data).hexdigest() for name, data in members.items()
    }
    members[MANIFEST_FILE] = (json.dumps(manifest, indent=2) + "\n").encode()

    _write_atomic(path, members)
    return path


def _read_verified_members(path: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    if not path.is_file():
        raise FileNotFoundError(f"No bundle at {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            if MANIFEST_FILE not in names:
                raise BundleCorruptError(f"{path} has no {MANIFEST_FILE}")
            manifest = json.loads(archive.read(MANIFEST_FILE))

            version = manifest.get("format_version")
            if not isinstance(version, int):
                raise BundleCorruptError(f"{path} has no valid format_version")
            if version > BUNDLE_FORMAT_VERSION:
                raise ValueError(
                    f"Bundle format {version} is newer than this savannah supports "
                    f"({BUNDLE_FORMAT_VERSION}); upgrade savannah to load {path}"
                )

            checksums = manifest["sha256"]
            missing = checksums.keys() - names
            if missing:
                raise BundleCorruptError(f"{path} is missing {sorted(missing)}")
            members = {name: archive.read(name) for name in checksums}
    # A damaged member fails zip's own CRC check as BadZipFile; a damaged
    # manifest can also surface as a decode error or a missing key.
    except (zipfile.BadZipFile, json.JSONDecodeError, KeyError, EOFError) as e:
        raise BundleCorruptError(f"{path} is corrupt: {e}") from e

    for name, expected in checksums.items():
        if hashlib.sha256(members[name]).hexdigest() != expected:
            raise BundleCorruptError(f"{path} is corrupt: bad checksum for {name}")
    return manifest, members


def read_bundle(path: str | Path) -> BundleContents:
    """Reads and verifies a bundle without building the policy. Tensors are
    loaded onto the CPU."""
    manifest, members = _read_verified_members(Path(path))

    def load(name: str) -> Any:
        return torch.load(
            io.BytesIO(members[name]), map_location="cpu", weights_only=True
        )

    return BundleContents(
        manifest=manifest,
        config=OmegaConf.create(members[CONFIG_FILE].decode()),
        ema_weights=load(WEIGHTS_FILE),
        training_state=load(TRAINING_STATE_FILE)
        if TRAINING_STATE_FILE in members
        else None,
    )


def load_bundle(
    path: str | Path, device: torch.device | str = "cpu", use_ema: bool = True
) -> PolicyBundle:
    """Rebuilds a policy, in eval mode on `device`, from a bundle file."""
    contents = read_bundle(path)
    if use_ema:
        weights = contents.ema_weights
    else:
        if contents.training_state is None:
            raise ValueError(f"{path} has only EMA weights; pass use_ema=true")
        weights = contents.training_state["model_state_dict"]

    policy = build_policy(contents.config)
    policy.load_state_dict(weights)
    policy.to(device).eval()
    return PolicyBundle(
        policy=policy,
        config=contents.config,
        manifest=contents.manifest,
        training_state=contents.training_state,
    )


def load_policy(path: str | Path, device: torch.device | str = "cpu") -> Policy:
    return load_bundle(path, device).policy


def resolve_bundle_path(cfg: DictConfig) -> str:
    """The bundle a script should load: `bundle_path`, or `wandb_artifact`
    after downloading it from W&B."""
    if cfg.wandb_artifact:
        logger.info("Fetching bundle from W&B artifact: {}", cfg.wandb_artifact)
        artifact = wandb.Api().artifact(cfg.wandb_artifact, type="model")
        artifact_dir = artifact.download()
        bundles = glob.glob(os.path.join(artifact_dir, f"*{BUNDLE_SUFFIX}"))
        if not bundles:
            raise FileNotFoundError(
                f"No {BUNDLE_SUFFIX} file found in artifact dir: {artifact_dir}"
            )
        return bundles[0]

    if cfg.bundle_path:
        return cfg.bundle_path

    raise ValueError("Either bundle_path or wandb_artifact must be set")


def apply_bundle_config(cfg: DictConfig, bundle: PolicyBundle) -> None:
    """Fills an eval-time `cfg` with the `model`, `task` and `name` the bundle
    was trained with."""
    with open_dict(cfg):
        cfg.model = bundle.config.model
        cfg.task = bundle.config.task
        cfg.name = bundle.config.get("name", "eval")
