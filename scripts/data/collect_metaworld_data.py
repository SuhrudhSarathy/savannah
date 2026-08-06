#!/usr/bin/env python
"""Collect MetaWorld expert demonstrations and push them as a LeRobot dataset.

Works for both standard MT1 tasks (e.g. assembly-v3) and MTN multi-instance
tasks (e.g. hammer-multi-nail-v3) — the benchmark class is picked
automatically based on `metaworld.env_dict.MULTI_INSTANCE_V3_ENVIRONMENTS`.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import metaworld
import metaworld.policies as mw_policies
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from metaworld.env_dict import MULTI_INSTANCE_V3_ENVIRONMENTS
from tqdm import tqdm

from savannah.tasks.metaworld import STATE_KEEP_INDICES, MultiCameraObsWrapper

# Extra per-step `info` fields (currently from hammer-multi-nail-v3 and other
# MTN envs) that can optionally be stored as dataset features via
# --info-fields. Only meaningful for tasks whose `info` dict has these keys.
INFO_FIELD_SPECS = {
    "near_object": {"dtype": "float32", "shape": (1,)},
    "grasp_success": {"dtype": "bool", "shape": (1,)},
    "grasp_reward": {"dtype": "float32", "shape": (1,)},
    "in_place_reward": {"dtype": "float32", "shape": (1,)},
    "unscaled_reward": {"dtype": "float32", "shape": (1,)},
    "nail_positions": {"dtype": "float32", "shape": (3, 3)},
    "nail_pos": {"dtype": "float32", "shape": (3,)},
    "nails_hit": {"dtype": "bool", "shape": (3,)},
    "num_nails_hit": {"dtype": "int64", "shape": (1,)},
    "hammer_dropped_off": {"dtype": "bool", "shape": (1,)},
    "button_pos": {"dtype": "float32", "shape": (3,)},
    "button_positions": {"dtype": "float32", "shape": (3, 3)},
}


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
    parser.add_argument(
        "--info-fields",
        nargs="+",
        default=[],
        choices=sorted(INFO_FIELD_SPECS.keys()),
        help="Extra per-step info fields (e.g. nail_positions, nails_hit) to store alongside each frame.",
    )
    return parser.parse_args()


def build_features(cameras: list[str], image_size: int, info_fields: list[str]) -> dict:
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
    for name in info_fields:
        spec = INFO_FIELD_SPECS[name]
        features[name] = {"dtype": spec["dtype"], "shape": spec["shape"], "names": None}
    return features


def _extract_info_field(name: str, info: dict) -> np.ndarray:
    spec = INFO_FIELD_SPECS[name]
    return np.asarray(info[name], dtype=spec["dtype"]).reshape(spec["shape"])


def load_task_description(task_name: str) -> str:
    config_path = Path(__file__).parent / "metaworld_config.json"
    with open(config_path) as f:
        task_descriptions = json.load(f)["TASK_DESCRIPTIONS"]
    return task_descriptions.get(
        task_name, f"metaworld {task_name} expert demonstration"
    )


def make_env(
    task_name: str,
    seed: int,
    cameras: list[str],
    image_size: int,
) -> tuple:
    benchmark_cls = (
        metaworld.MTN if task_name in MULTI_INSTANCE_V3_ENVIRONMENTS else metaworld.MT1
    )
    benchmark = benchmark_cls(task_name, seed=seed)
    raw_env = benchmark.train_classes[task_name](render_mode="rgb_array")
    raw_env._freeze_rand_vec = False
    wrapped = MultiCameraObsWrapper(
        raw_env, camera_names=cameras, img_size=(image_size, image_size)
    )
    return wrapped, raw_env, benchmark


def collect_episode(
    wrapped_env,
    raw_env,
    benchmark,
    expert_policy,
    task_description: str,
    cameras: list[str],
    info_fields: list[str],
    max_steps: int = 500,
) -> tuple[list[dict], bool]:
    task_idx = np.random.randint(len(benchmark.train_tasks))
    raw_env.set_task(benchmark.train_tasks[task_idx])
    obs_dict, _ = wrapped_env.reset()

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
            "task": task_description,
        }
        for cam in cameras:
            frame[f"observation.images.{cam}"] = obs_dict[cam]

        obs_dict, _, terminated, truncated, info = wrapped_env.step(action)

        # info reflects the outcome of `action` taken from this frame's
        # observation, so it's attached here rather than at the next frame.
        for name in info_fields:
            frame[name] = _extract_info_field(name, info)

        frames.append(frame)

        if info.get("success", False):
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

    expert_cls = mw_policies.ENV_POLICY_MAP[args.task]
    task_description = load_task_description(args.task)
    features = build_features(args.cameras, args.image_size, args.info_fields)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        robot_type="sawyer",
    )

    wrapped_env, raw_env, benchmark = make_env(
        args.task, args.seed, args.cameras, args.image_size
    )

    successes = 0
    attempts = 0
    failed = 0

    pbar = tqdm(total=args.num_episodes, desc="Collecting episodes", unit="ep")

    try:
        while successes < args.num_episodes and attempts < max_attempts:
            attempts += 1
            # Fresh policy instance per attempt: some expert policies (e.g.
            # hammer-multi-nail-v3's) are stateful across get_action() calls
            # and have no reset(), so reusing one across episodes would leak
            # state-machine progress from a prior attempt into this one.
            expert = expert_cls()
            frames, success = collect_episode(
                wrapped_env,
                raw_env,
                benchmark,
                expert,
                task_description,
                args.cameras,
                args.info_fields,
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
