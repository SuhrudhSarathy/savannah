#!/usr/bin/env python
"""Collect MetaWorld expert demonstrations and push them as a LeRobot dataset."""

from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")

import metaworld
import metaworld.policies as mw_policies
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

from savannah.tasks.metaworld import STATE_KEEP_INDICES, MultiCameraObsWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect MetaWorld expert demonstrations and push to HuggingFace Hub."
    )
    parser.add_argument("--task", type=str, default="assembly-v3")
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=50,
        help="Number of successful episodes to collect",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Max total rollout attempts (default: 3 * num_episodes)",
    )
    parser.add_argument(
        "--cameras", nargs="+", default=["topview", "gripperPOV", "corner"]
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace Hub repo id, e.g. myuser/assembly-v3-expert",
    )
    parser.add_argument("--push-to-hub", action="store_true", default=False)
    parser.add_argument(
        "--push-only",
        action="store_true",
        default=False,
        help="Skip collection; push an already-recorded local dataset to the Hub",
    )
    parser.add_argument(
        "--keep-failed",
        action="store_true",
        default=False,
        help="Save failed episodes too (no success filtering)",
    )
    parser.add_argument("--private", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_features(cameras: list[str], image_size: int) -> dict:
    H = W = image_size
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["ee_x", "ee_y", "ee_z", "gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["dx", "dy", "dz", "gripper_ctrl"],
        },
    }
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (H, W, 3),
            "names": ["height", "width", "channel"],
        }
    return features


def make_env(
    task_name: str,
    seed: int,
    cameras: list[str],
    image_size: int,
) -> tuple:
    mt1 = metaworld.MT1(task_name, seed=seed)
    raw_env = mt1.train_classes[task_name](render_mode="rgb_array")
    raw_env._freeze_rand_vec = False
    wrapped = MultiCameraObsWrapper(
        raw_env, camera_names=cameras, img_size=(image_size, image_size)
    )
    return wrapped, raw_env, mt1


def collect_episode(
    wrapped_env,
    raw_env,
    mt1,
    expert_policy,
    task_name: str,
    cameras: list[str],
    max_steps: int = 500,
) -> tuple[list[dict], bool]:
    task_idx = np.random.randint(len(mt1.train_tasks))
    raw_env.set_task(mt1.train_tasks[task_idx])
    obs_dict, _ = wrapped_env.reset()

    task_str = f"metaworld {task_name} expert demonstration"
    frames: list[dict] = []
    success = False

    for _ in range(max_steps):
        raw_obs = obs_dict["state"]
        action = np.clip(expert_policy.get_action(raw_obs), -1.0, 1.0).astype(
            np.float32
        )

        frame: dict = {
            "observation.state": raw_obs[STATE_KEEP_INDICES].astype(np.float32),
            "action": action,
            "task": task_str,
        }
        for cam in cameras:
            frame[f"observation.images.{cam}"] = obs_dict[cam]

        frames.append(frame)

        obs_dict, _, terminated, truncated, info = wrapped_env.step(action)

        if info.get("success", 0) >= 0.5:
            success = True
            break
        if truncated or terminated:
            break

    return frames, success


def main() -> None:
    args = parse_args()

    if args.push_only:
        from lerobot.utils.constants import HF_LEROBOT_HOME

        local_root = HF_LEROBOT_HOME / args.repo_id
        if not (local_root / "meta" / "info.json").exists():
            raise FileNotFoundError(
                f"No local dataset found at {local_root}. "
                f"Run without --push-only to collect data first."
            )
        tqdm.write(f"Pushing existing local dataset {args.repo_id} to Hub ...")
        LeRobotDataset(args.repo_id, root=local_root).push_to_hub(private=args.private)
        tqdm.write("Pushed.")
        return

    assert args.image_size % 2 == 0, (
        f"--image-size must be even (got {args.image_size})"
    )

    max_attempts = args.max_attempts or 3 * args.num_episodes

    if args.task not in mw_policies.ENV_POLICY_MAP:
        available = sorted(mw_policies.ENV_POLICY_MAP.keys())
        raise ValueError(f"No expert policy for '{args.task}'. Available: {available}")

    expert = mw_policies.ENV_POLICY_MAP[args.task]()
    features = build_features(args.cameras, args.image_size)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        robot_type="sawyer",
    )

    wrapped_env, raw_env, mt1 = make_env(
        args.task, args.seed, args.cameras, args.image_size
    )

    successes = 0
    attempts = 0
    failed = 0

    pbar = tqdm(total=args.num_episodes, desc="Collecting episodes", unit="ep")

    try:
        while successes < args.num_episodes and attempts < max_attempts:
            attempts += 1
            frames, success = collect_episode(
                wrapped_env, raw_env, mt1, expert, args.task, args.cameras
            )

            if success or args.keep_failed:
                for frame in frames:
                    dataset.add_frame(frame)
                dataset.save_episode()
                if success:
                    successes += 1
                    pbar.update(1)
                else:
                    failed += 1
            else:
                failed += 1

            pbar.set_postfix(failed=failed, attempts=attempts)

    finally:
        pbar.close()
        wrapped_env.close()

    if args.push_to_hub:
        tqdm.write(f"Pushing to {args.repo_id} ...")
        dataset.push_to_hub(private=args.private)
        tqdm.write("Pushed.")


if __name__ == "__main__":
    main()
