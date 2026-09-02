"""Task for reading ABC's put-bottles-in-bin dataset (https://abc.bot).

ABC episodes (produced by `export_mcap.py` / `prepare.py` in the abc repo)
live in a flat per-episode layout, not a LeRobot dataset:

    <root>/episode_<uuid>/
        states_actions.bin              # (T, 28) float64: 14 states + 14 actions
        combined_camera-images-rgb.mp4  # cameras stacked vertically, 30 fps
        episode_metadata.json           # {"cameras": [...], "task_name": ..., "instruction": ...}

`norm_stats.json` (z-score mean/std for state and actions) is expected one
level above `root`, e.g. `<root>/../norm_stats.json` — this is where abc's
`prepare.py` places it relative to `train_real/`, `val_real/`, etc.

Because this isn't a LeRobot dataset, ABCPutBottlesTask reads episodes
directly (mirroring abc's own `abc_minimal/train_loop.py`) instead of going
through LerobotDatasetWrapper.

Eval runs against `abc_bottle_gym` (https://github.com/SuhrudhSarathy/abc_bottle_gym),
a standalone Gymnasium port of abc's put-bottles-in-bin MuJoCo task
("PutBottlesInBin-v0") with no CUDA/torch/mujoco-warp dependency. Its 14-dim
state/action layout (6 arm joints + 1 gripper, per arm) matches
`states_actions.bin` exactly, so training and eval share the same
normalization stats (`norm_stats.json`, z-scored mean/std).
"""

import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import abc_bottle_gym  # noqa: F401 -- registers "PutBottlesInBin-v0"
import gymnasium as gym
import numpy as np
import torch
from abc_bottle_gym.env import ACTION_HIGH, ACTION_LOW, DEFAULT_CAMERA_KEYS
from torch.utils.data import DataLoader, Dataset

from savannah.tasks import BaseRobotTask
from savannah.utils.observation import ObservationKey

STATE_DIM = 14
ACTION_DIM = 14
ROW_DIM = STATE_DIM + ACTION_DIM


def _task_name_to_prompt(task_name: str) -> str:
    """Falls back to abc's own `task_name_to_prompt` convention (underscores
    -> spaces) when an episode's metadata has no `instruction` field."""
    return " ".join(task_name.replace("-", " ").replace("_", " ").split())


def _load_norm_stats(root: Path) -> Dict[str, Dict[str, np.ndarray]]:
    raw = json.loads((root.parent / "norm_stats.json").read_text())
    stats = raw.get("norm_stats", raw)
    if "state" not in stats and "actions" not in stats:
        key = "xdof" if "xdof" in stats else next(iter(stats))
        stats = stats[key]
    return {
        split: {k: np.asarray(v, dtype=np.float32) for k, v in stats[split].items()}
        for split in ("state", "actions")
    }


def _normalize(x: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return (x - stats["mean"]) / (stats["std"] + 1e-6)


def _unnormalize(x: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return x * (stats["std"] + 1e-6) + stats["mean"]


def _scan_episodes(
    root: Path, chunk_length: int
) -> List[Tuple[Path, int, int, Tuple[str, ...], str]]:
    """Returns (episode_dir, length, usable_starts, source_cameras, prompt).

    `prompt` is each episode's natural-language instruction: metadata's own
    `instruction` field if present, else `task_name` prettified the same way
    abc's `task_name_to_prompt` does. Training (`ABCEpisodeDataset`) and eval
    (`ABCPutBottlesTask._get_prompt`) both read it from here, so they always
    agree on what language the policy was conditioned on.
    """
    episodes = []
    for ep_dir in sorted(root.iterdir()):
        bin_path = ep_dir / "states_actions.bin"
        if not bin_path.exists():
            continue
        length = bin_path.stat().st_size // (ROW_DIM * 8)
        usable = length - (chunk_length - 1)
        if usable <= 0:
            continue
        meta = {}
        meta_path = ep_dir / "episode_metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        cameras = tuple(meta.get("cameras") or ())
        prompt = meta.get("instruction") or _task_name_to_prompt(
            meta.get("task_name", "")
        )
        episodes.append((ep_dir, length, usable, cameras, prompt))
    if not episodes:
        raise ValueError(f"no ABC episodes found under {root}")
    return episodes


def _read_state_action_rows(ep_dir: Path, start: int, end: int) -> np.ndarray:
    row_bytes = ROW_DIM * 8
    with open(ep_dir / "states_actions.bin", "rb") as f:
        f.seek(start * row_bytes)
        raw = f.read((end - start) * row_bytes)
    return np.frombuffer(raw, dtype=np.float64).reshape(-1, ROW_DIM)


def _decode_frame(
    ep_dir: Path,
    idx: int,
    episode_length: int,
    source_cameras: Tuple[str, ...],
    cameras: List[str],
) -> Dict[str, torch.Tensor]:
    """Decodes frame `idx` from the per-episode combined mp4 via torchcodec.

    Uses a synthesized CFR frame map (pts = 512*k, 1/15360 timebase) instead
    of per-file probing, matching abc's `train_loop.decode_frame`.
    """
    from torchcodec.decoders import VideoDecoder

    frames = [
        {"pts": 512 * i, "duration": 512, "key_frame": 1 if i % 30 == 0 else 0}
        for i in range(episode_length)
    ]
    mapping = json.dumps({"frames": frames})
    decoder = VideoDecoder(
        str(ep_dir / "combined_camera-images-rgb.mp4"), custom_frame_mappings=mapping
    )
    frame = decoder[idx]  # (C, n_cams * H, W) uint8
    n_cams = len(source_cameras)
    h = frame.shape[1] // n_cams
    stacked = {
        name: frame[:, i * h : (i + 1) * h, :].float() / 255.0
        for i, name in enumerate(source_cameras)
    }
    return {cam: stacked[cam] for cam in cameras}


class ABCEpisodeDataset(Dataset):
    """Map-style dataset over all usable (episode, frame) pairs in `root`.

    Mirrors abc's own `EpisodeDataset` but returns LeRobot-style keys
    ("observation.state", "action", "observation.images.<cam>") so
    `ABCPutBottlesTask.format_batch` stays consistent with the other tasks
    in this package.
    """

    def __init__(
        self,
        root: str,
        cameras: List[str],
        obs_horizon: int,
        action_horizon: int,
        max_episodes: Optional[int] = None,
    ):
        self.root = Path(root)
        self.cameras = list(cameras)
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.episodes = _scan_episodes(self.root, action_horizon)
        if max_episodes is not None:
            self.episodes = self.episodes[:max_episodes]
        self.norm_stats = _load_norm_stats(self.root)
        self.cum = np.cumsum([usable for _, _, usable, _, _ in self.episodes])

    def __len__(self) -> int:
        return int(self.cum[-1])

    def __getitem__(self, global_idx: int) -> dict:
        ep_idx = int(np.searchsorted(self.cum, global_idx, side="right"))
        offset = int(self.cum[ep_idx - 1]) if ep_idx > 0 else 0
        k = global_idx - offset
        ep_dir, length, _, source_cameras, prompt = self.episodes[ep_idx]

        rows = _read_state_action_rows(ep_dir, k, k + self.action_horizon)
        actions = _normalize(rows[:, STATE_DIM:], self.norm_stats["actions"]).astype(
            np.float32
        )

        # Observation window of obs_horizon frames ending at k, oldest first.
        # Left-pad by repeating frame 0 when k is near the episode start --
        # same convention as preprocess_observation_history at eval time.
        obs_indices = [
            max(0, k - self.obs_horizon + 1 + i) for i in range(self.obs_horizon)
        ]
        state_rows = np.stack(
            [
                rows[0, :STATE_DIM]
                if idx == k
                else _read_state_action_rows(ep_dir, idx, idx + 1)[0, :STATE_DIM]
                for idx in obs_indices
            ]
        )
        state = _normalize(state_rows, self.norm_stats["state"]).astype(
            np.float32
        )  # (T, 14)
        frames = [
            _decode_frame(ep_dir, idx, length, source_cameras, self.cameras)
            for idx in obs_indices
        ]

        item = {
            "observation.state": torch.from_numpy(state),  # (T, 14)
            "action": torch.from_numpy(actions),  # (action_horizon, 14)
            "task": prompt,
        }
        for cam in self.cameras:
            item[f"observation.images.{cam}"] = torch.stack(
                [frame[cam] for frame in frames]
            )  # (T, C, H, W)
        return item


class ABCPutBottlesTask(BaseRobotTask):
    """Reads ABC's put-bottles-in-bin data (real or sim split) for training.

    `config.root` must point at a split directory produced by abc's
    `prepare.py` / `export_hf_task.py`, e.g. `<ABC_CACHE>/train_real`. The
    matching validation split is assumed to sit alongside it with the first
    `train` replaced by `val` (`train_real` -> `val_real`, `train_sim` ->
    `val_sim`), matching abc's cache layout. `config.root` is also used at
    eval time to locate `norm_stats.json` (one level up) and to derive the
    language prompt from the training episodes' own metadata, so both
    state/action normalization and language conditioning are identical
    between training and rollout.
    """

    def __init__(self, config, device: torch.device):
        super().__init__(config, device)
        self._norm_stats: Optional[Dict[str, Dict[str, np.ndarray]]] = None
        self._prompt: Optional[str] = None

    def _get_norm_stats(self) -> Dict[str, Dict[str, np.ndarray]]:
        if self._norm_stats is None:
            if not self.config.root:
                raise ValueError(
                    "ABCPutBottlesTask requires config.root to be set (used to "
                    "locate norm_stats.json for state/action normalization)"
                )
            self._norm_stats = _load_norm_stats(Path(self.config.root))
        return self._norm_stats

    def _get_prompt(self) -> str:
        """The language instruction used for eval-time conditioning.

        Read from the training split's own episode metadata (same field
        `ABCEpisodeDataset` uses for `item["task"]`), so eval always matches
        whatever language the policy was actually trained on -- no separate
        hardcoded prompt to drift out of sync.
        """
        if self._prompt is None:
            if not self.config.root:
                raise ValueError(
                    "ABCPutBottlesTask requires config.root to be set (used to "
                    "derive the eval-time language prompt from training episodes)"
                )
            episodes = _scan_episodes(
                Path(self.config.root), self.config.action_horizon
            )
            self._prompt = episodes[0][-1]
        return self._prompt

    def get_train_loader(self) -> DataLoader:
        if self._train_loader is None:
            self._train_loader, self._val_loader = self._create_loaders()
        return self._train_loader

    def get_val_loader(self) -> DataLoader:
        if self._val_loader is None:
            self.get_train_loader()  # Forces creation of both
        return self._val_loader

    def _create_loaders(self) -> Tuple[DataLoader, DataLoader]:
        if not self.config.root:
            raise ValueError("ABCPutBottlesTask requires config.root to be set")

        train_root = Path(self.config.root)
        val_root = train_root.with_name(train_root.name.replace("train", "val", 1))

        train_dataset = ABCEpisodeDataset(
            str(train_root),
            self.config.cameras,
            self.config.obs_horizon,
            self.config.action_horizon,
        )
        val_dataset = ABCEpisodeDataset(
            str(val_root),
            self.config.cameras,
            self.config.obs_horizon,
            self.config.action_horizon,
            max_episodes=self.config.val_episodes,
        )

        loader_kwargs = {}
        if self.config.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 4

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            **loader_kwargs,
        )
        return train_loader, val_loader

    def format_batch(
        self, batch: Dict[str, Any], step: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        state = batch["observation.state"].to(self.device)  # (B, obs_horizon, D)
        actions = batch["action"].to(self.device)  # (B, action_horizon, D)

        images = [
            batch[f"observation.images.{cam}"].to(
                self.device
            )  # (B, obs_horizon, C, H, W)
            for cam in self.config.cameras
        ]

        if step is not None:
            images = self._augmenter.augment_images(images, step)
            state = self._augmenter.augment_state(state, step)

        result = {
            ObservationKey.images: images,
            ObservationKey.state: state,
            ObservationKey.gt_actions: actions,
        }
        if self.config.use_language:
            result[ObservationKey.language] = batch["task"]
        return result

    def _make_env(self, render_mode: str) -> gym.Env:
        kwargs: Dict[str, Any] = {
            "render_mode": render_mode,
            "camera_keys": tuple(self.config.cameras) or DEFAULT_CAMERA_KEYS,
            # Always decode camera frames, regardless of render_mode, since
            # preprocess_observation(_history) needs obs["images"] for the policy.
            "include_images": True,
        }
        if self.config.image_size is not None:
            kwargs["height"] = self.config.image_size
            kwargs["width"] = self.config.image_size
        return gym.make("PutBottlesInBin-v0", **kwargs)

    def _obs_to_tensors(self, obs_list: List[Dict[str, Any]]) -> dict:
        stats = self._get_norm_stats()
        states = np.stack(
            [_normalize(o["state"], stats["state"]) for o in obs_list]
        ).astype(np.float32)  # (T, 14)
        state_tensor = (
            torch.from_numpy(states).unsqueeze(0).to(self.device)
        )  # (1, T, 14)

        images = []
        for cam in self.config.cameras:
            imgs = np.stack([o["images"][cam] for o in obs_list])  # (T, C, H, W) uint8
            img_tensor = (
                torch.from_numpy(imgs)
                .float()
                .divide(255.0)
                .unsqueeze(0)
                .to(self.device)
            )  # (1, T, C, H, W)
            images.append(img_tensor)

        result = {ObservationKey.images: images, ObservationKey.state: state_tensor}
        if self.config.use_language:
            result[ObservationKey.language] = [self._get_prompt()]
        return result

    def preprocess_observation(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return self._obs_to_tensors([obs])

    def preprocess_observation_history(self, obs_history: deque) -> dict:
        obs_horizon = self.config.obs_horizon
        obs_list = list(obs_history)

        if len(obs_list) < obs_horizon:
            pad = [obs_list[0]] * (obs_horizon - len(obs_list))
            obs_list = pad + obs_list

        return self._obs_to_tensors(obs_list)

    def postprocess_chunk(self, actions: torch.Tensor) -> None:
        # ActionChunkVisualizer expects 2D pixel coords (pusht-only); ABC's
        # actions are 14-dim joint/gripper values, so there's nothing to draw.
        return None

    def postprocess_action(self, action_tensor: torch.Tensor) -> np.ndarray:
        stats = self._get_norm_stats()
        action = action_tensor.detach().cpu().numpy().reshape(-1).astype(np.float32)
        action = _unnormalize(action, stats["actions"])
        return np.clip(action, ACTION_LOW, ACTION_HIGH).astype(np.float32)

    def is_success(self, info: Dict[str, Any]) -> bool:
        return bool(info.get("success", False))

    def get_metrics(self, info: Dict[str, Any]) -> str:
        return (
            f"bottles_in_bin: {info.get('num_bottles_in_bin', 0)}"
            f"/{info.get('num_active_bottles', 0)}"
        )
