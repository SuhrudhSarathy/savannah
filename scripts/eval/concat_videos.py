#!/usr/bin/env python
"""Concatenate eval episode videos into a single MP4 with fade transitions and episode titles.

Usage:
    python scripts/eval/concat_videos.py --folder eval_videos_ddim_assembly --output combined.mp4
    python scripts/eval/concat_videos.py --folder eval_videos --output combined.mp4 --title "Assembly v3 — DiT+DDIM"
    python scripts/eval/concat_videos.py --folder eval_videos --output combined.mp4 --title "Button Press" --fps 30
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np


def find_episode_videos(folder: Path) -> list[Path]:
    pattern = re.compile(r"episode[_-]?(\d+)", re.IGNORECASE)
    videos = []
    for f in sorted(folder.glob("*.mp4")):
        m = pattern.search(f.stem)
        if m:
            videos.append((int(m.group(1)), f))
    videos.sort(key=lambda x: x[0])
    return [path for _, path in videos]


def read_video(path: Path) -> tuple[list[np.ndarray], int, tuple[int, int]]:
    cap = cv2.VideoCapture(str(path))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        return [], fps, (0, 0)
    h, w = frames[0].shape[:2]
    return frames, fps, (w, h)


def make_text_frame(
    text: str,
    size: tuple[int, int],
    duration_s: float,
    fps: int,
    font_scale: float = 1.2,
    bg_color: tuple[int, int, int] = (0, 0, 0),
    fg_color: tuple[int, int, int] = (255, 255, 255),
) -> list[np.ndarray]:
    w, h = size
    frame = np.full((h, w, 3), bg_color, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2

    max_w = int(w * 0.85)
    lines = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = current + " " + word
            tw, _ = cv2.getTextSize(candidate, font, font_scale, thickness)[0]
            if tw > max_w:
                lines.append(current)
                current = word
            else:
                current = candidate
        lines.append(current)

    line_sizes = [
        cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines
    ]
    line_height = max(s[1] for s in line_sizes) + 10
    total_height = line_height * len(lines)
    y_start = (h - total_height) // 2 + line_sizes[0][1]

    for i, (line, (tw, th)) in enumerate(zip(lines, line_sizes)):
        x = (w - tw) // 2
        y = y_start + i * line_height
        cv2.putText(
            frame, line, (x, y), font, font_scale, fg_color, thickness, cv2.LINE_AA
        )

    return [frame.copy()] * int(duration_s * fps)


def make_fade(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    duration_s: float,
    fps: int,
) -> list[np.ndarray]:
    n = max(int(duration_s * fps), 2)
    frames = []
    for i in range(n):
        alpha = i / (n - 1)
        blended = cv2.addWeighted(frame_a, 1.0 - alpha, frame_b, alpha, 0)
        frames.append(blended)
    return frames


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate eval videos with transitions"
    )
    parser.add_argument(
        "--folder", required=True, help="Folder containing episode mp4 files"
    )
    parser.add_argument(
        "--output", default="combined_eval.mp4", help="Output file path"
    )
    parser.add_argument("--title", default=None, help="Title text shown at the start")
    parser.add_argument(
        "--fps", type=int, default=None, help="Output FPS (default: same as source)"
    )
    parser.add_argument(
        "--fade-duration", type=float, default=0.5, help="Fade duration in seconds"
    )
    parser.add_argument(
        "--title-duration",
        type=float,
        default=2.0,
        help="Title/episode card duration in seconds",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    video_paths = find_episode_videos(folder)
    if not video_paths:
        print(f"No episode videos found in {folder}")
        return

    print(f"Found {len(video_paths)} episodes in {folder}")

    first_frames, src_fps, size = read_video(video_paths[0])
    if not first_frames:
        print(f"Could not read {video_paths[0]}")
        return

    fps = args.fps or src_fps
    fade_dur = args.fade_duration
    title_dur = args.title_duration

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, size)

    all_episodes = [(0, first_frames)]
    for path in video_paths[1:]:
        frames, _, _ = read_video(path)
        if frames:
            idx = int(re.search(r"(\d+)", path.stem).group(1))
            all_episodes.append((idx, frames))

    def resize_if_needed(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if (w, h) != size:
            return cv2.resize(frame, size)
        return frame

    def write_frames(frames: list[np.ndarray]):
        for f in frames:
            writer.write(resize_if_needed(f))

    black = np.zeros((size[1], size[0], 3), dtype=np.uint8)

    if args.title:
        title_frames = make_text_frame(args.title, size, title_dur, fps, font_scale=1.5)
        write_frames(title_frames)
        write_frames(make_fade(title_frames[-1], black, fade_dur, fps))

    for i, (ep_idx, ep_frames) in enumerate(all_episodes):
        card_text = f"Episode {ep_idx}"
        card_frames = make_text_frame(card_text, size, title_dur, fps)

        if i == 0 and not args.title:
            write_frames(card_frames)
        else:
            prev_frame = black if i == 0 else all_episodes[i - 1][1][-1]
            write_frames(make_fade(prev_frame, card_frames[0], fade_dur, fps))
            write_frames(card_frames)

        write_frames(make_fade(card_frames[-1], ep_frames[0], fade_dur, fps))
        write_frames(ep_frames)

    write_frames(make_fade(all_episodes[-1][1][-1], black, fade_dur, fps))

    writer.release()
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
