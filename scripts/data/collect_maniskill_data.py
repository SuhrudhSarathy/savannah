#!/usr/bin/env python
"""Collect ColorMatching expert demonstrations and push them as a LeRobot dataset."""

from __future__ import annotations

import argparse
import random
import warnings

warnings.filterwarnings("ignore")

import gymnasium as gym
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

import savannah.envs  # noqa: F401 — triggers @register_env
from savannah.envs import VectorizedScriptedPolicy

COLORS = ["red", "green", "blue"]
CAMERAS = ["hand_camera", "front_cam", "side_cam"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect ColorMatching expert demonstrations and push to HuggingFace Hub."
    )
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
        "--cameras", nargs="+", default=CAMERAS,
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace Hub repo id, e.g. myuser/color-matching-expert",
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
            "shape": (8,),
            "names": [
                "ee_x", "ee_y", "ee_z",
                "ee_qw", "ee_qx", "ee_qy", "ee_qz",
                "gripper",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": [
                "dx", "dy", "dz",
                "drot_x", "drot_y", "drot_z",
                "gripper_ctrl",
            ],
        },
    }
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (H, W, 3),
            "names": ["height", "width", "channel"],
        }
    return features


def collect_episode(
    env,
    raw_env,
    pick_color: str,
    place_color: str,
    cameras: list[str],
    max_steps: int,
) -> tuple[list[dict], bool]:
    obs, _ = env.reset(options={"pick_color": pick_color, "place_color": place_color})

    policy = VectorizedScriptedPolicy(
        num_envs=1,
        device=raw_env.device,
        pick_color=pick_color,
        place_color=place_color,
    )

    task_str = f"Pick the {pick_color} block and place it in the {place_color} bin"
    frames: list[dict] = []
    success = False

    for _ in range(max_steps):
        tcp_pose = raw_env.agent.tcp.pose.raw_pose[0].cpu().numpy()
        qpos = raw_env.agent.robot.get_qpos()[0].cpu().numpy()
        gripper_width = np.array([qpos[-2] + qpos[-1]], dtype=np.float32)
        state = np.concatenate([tcp_pose, gripper_width]).astype(np.float32)

        action = policy.act(raw_env)
        action_np = action[0].cpu().numpy().astype(np.float32)

        frame: dict = {
            "observation.state": state,
            "action": action_np,
            "task": task_str,
        }
        for cam in cameras:
            img = obs["sensor_data"][cam]["rgb"][0].cpu().numpy()
            frame[f"observation.images.{cam}"] = img[..., :3]

        frames.append(frame)

        obs, _, terminated, truncated, info = env.step(action)

        if info["success"].any():
            success = True
            break
        if terminated.any() or truncated.any():
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    max_attempts = args.max_attempts or 3 * args.num_episodes
    features = build_features(args.cameras, args.image_size)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        robot_type="panda",
    )

    env = gym.make(
        "ColorMatching-v1",
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_ee_delta_pose",
    )
    raw_env = env.unwrapped

    successes = 0
    attempts = 0
    failed = 0

    pbar = tqdm(total=args.num_episodes, desc="Collecting episodes", unit="ep")

    try:
        while successes < args.num_episodes and attempts < max_attempts:
            attempts += 1
            pick_color = random.choice(COLORS)
            place_color = random.choice(COLORS)

            frames, success = collect_episode(
                env, raw_env, pick_color, place_color, args.cameras, args.max_steps,
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

            pbar.set_postfix(
                pick=pick_color, place=place_color,
                failed=failed, attempts=attempts,
            )

    finally:
        pbar.close()
        env.close()

    if args.push_to_hub:
        tqdm.write(f"Pushing to {args.repo_id} ...")
        dataset.push_to_hub(private=args.private)
        tqdm.write("Pushed.")


if __name__ == "__main__":
    main()
