#!/usr/bin/env python
"""Convert ABC's put-bottles-in-bin episode dumps to a LeRobot dataset and push it.

ABC episodes (see `savannah.tasks.abc` for the on-disk layout) are a flat
per-episode directory of a raw states/actions binary plus a combined,
multi-camera mp4 -- not a LeRobot dataset. This script re-reads that layout
episode by episode (reusing the same scanning/decoding helpers
`ABCPutBottlesTask` trains from, so the conversion can't drift out of sync
with what training actually reads) and re-emits it as a standard
LeRobotDataset, storing raw (unnormalized) state/action values so consumers
can compute their own stats.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

from savannah.tasks.abc import (
    ACTION_DIM,
    STATE_DIM,
    _open_decoder,
    _read_state_action_rows,
    _scan_episodes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ABC put-bottles-in-bin episodes to a LeRobot dataset and push to HuggingFace Hub."
    )
    parser.add_argument(
        "--root",
        nargs="+",
        required=True,
        help="One or more ABC split directories to convert (e.g. "
        "/mnt/shared/abc/train_sim /mnt/shared/abc/train_real). Episodes "
        "from all roots are merged into a single output dataset.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace Hub repo id, e.g. myuser/abc-put-bottles",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help="Camera names to export (default: whatever the first episode's "
        "metadata declares)",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--robot-type", type=str, default="xdof")
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Limit episodes converted per --root (for a quick test run)",
    )
    parser.add_argument("--push-to-hub", action="store_true", default=False)
    parser.add_argument(
        "--push-only",
        action="store_true",
        default=False,
        help="Skip conversion; push an already-converted local dataset to the Hub",
    )
    parser.add_argument("--private", action="store_true", default=False)
    return parser.parse_args()


def build_features(cameras: list[str], height: int, width: int) -> dict:
    features = {
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": None},
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": None},
    }
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        }
    return features


def _split_cameras_raw(
    frame, source_cameras: tuple[str, ...], cameras: list[str]
) -> dict[str, np.ndarray]:
    """Splits a decoded (C, n_cams * H, W) uint8 stacked frame into
    per-camera HWC uint8 arrays -- LeRobotDataset's video feature
    convention. Unlike `savannah.tasks.abc._split_cameras`, this keeps raw
    uint8 pixels (no /255 float normalization), matching how the other
    collection scripts in this repo store frames.
    """
    n_cams = len(source_cameras)
    h = frame.shape[1] // n_cams
    stacked = {
        name: frame[:, i * h : (i + 1) * h, :].permute(1, 2, 0).numpy()
        for i, name in enumerate(source_cameras)
    }
    return {cam: stacked[cam] for cam in cameras}


def convert_episode(
    dataset: LeRobotDataset,
    ep_dir: Path,
    length: int,
    source_cameras: tuple[str, ...],
    prompt: str,
    cameras: list[str],
) -> None:
    rows = _read_state_action_rows(ep_dir, 0, length)
    states = rows[:, :STATE_DIM].astype(np.float32)
    actions = rows[:, STATE_DIM:].astype(np.float32)

    decoder = _open_decoder(ep_dir, length)
    for t in range(length):
        frame = _split_cameras_raw(decoder[t], source_cameras, cameras)
        item = {
            "observation.state": states[t],
            "action": actions[t],
            "task": prompt,
        }
        for cam in cameras:
            item[f"observation.images.{cam}"] = frame[cam]
        dataset.add_frame(item)
    dataset.save_episode()


def main() -> None:
    args = parse_args()

    if args.push_only:
        from lerobot.utils.constants import HF_LEROBOT_HOME

        local_root = HF_LEROBOT_HOME / args.repo_id
        if not (local_root / "meta" / "info.json").exists():
            raise FileNotFoundError(
                f"No local dataset found at {local_root}. "
                f"Run without --push-only to convert data first."
            )
        tqdm.write(f"Pushing existing local dataset {args.repo_id} to Hub ...")
        LeRobotDataset(args.repo_id, root=local_root).push_to_hub(private=args.private)
        tqdm.write("Pushed.")
        return

    episodes = []
    for root in args.root:
        root_episodes = _scan_episodes(Path(root), chunk_length=1)  # usable == length
        if args.max_episodes is not None:
            root_episodes = root_episodes[: args.max_episodes]
        episodes.extend(root_episodes)
    if not episodes:
        raise ValueError(f"No ABC episodes found under {args.root}")

    ep_dir0, length0, _, source_cameras0, _ = episodes[0]
    cameras = args.cameras or list(source_cameras0)
    for _, _, _, source_cameras, _ in episodes:
        missing = set(cameras) - set(source_cameras)
        if missing:
            raise ValueError(
                f"Camera(s) {sorted(missing)} requested but not present in all "
                f"episodes (some only have {source_cameras})"
            )

    decoder0 = _open_decoder(ep_dir0, length0)
    sample_frame = decoder0[0]
    height = sample_frame.shape[1] // len(source_cameras0)
    width = sample_frame.shape[2]

    features = build_features(cameras, height, width)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        robot_type=args.robot_type,
    )

    for ep_dir, length, _, source_cameras, prompt in tqdm(
        episodes, desc="Converting episodes", unit="ep"
    ):
        convert_episode(dataset, ep_dir, length, source_cameras, prompt, cameras)

    if args.push_to_hub:
        tqdm.write(f"Pushing to {args.repo_id} ...")
        dataset.push_to_hub(private=args.private)
        tqdm.write("Pushed.")


if __name__ == "__main__":
    main()
