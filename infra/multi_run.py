"""
Run multiple training experiments sequentially or in parallel based on available VRAM.

Usage:
    uv run python scripts/multi_run.py configs/experiments.yaml
    uv run python scripts/multi_run.py configs/experiments.yaml --dry-run
"""

import argparse
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class GPU:
    index: int
    total_mb: int
    free_mb: int


@dataclass
class Experiment:
    name: str
    overrides: list[str] = field(default_factory=list)


def get_gpus() -> list[GPU]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            idx, total, free = (x.strip() for x in line.split(","))
            gpus.append(GPU(index=int(idx), total_mb=int(total), free_mb=int(free)))
        return gpus
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def load_experiments(path: Path) -> tuple[list[Experiment], int]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    vram_per_run_gb = cfg.get("vram_per_run_gb", 8)
    experiments = [
        Experiment(
            name=exp["name"],
            overrides=exp.get("overrides", []),
        )
        for exp in cfg["experiments"]
    ]
    return experiments, vram_per_run_gb


def build_command(
    exp: Experiment,
    cuda_devices: str | None = None,
    memory_fraction: float | None = None,
) -> tuple[list[str], dict]:
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/train.py",
        f"wandb.run_name={exp.name}",
        f"hydra.run.dir=outputs/{exp.name}",
    ]
    cmd.extend(exp.overrides)
    env = os.environ.copy()
    if cuda_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    if memory_fraction is not None:
        env["CUDA_MEMORY_FRACTION"] = str(memory_fraction)
    return cmd, env


def run_batch(
    batch: list[tuple[Experiment, str | None]],
    dry_run: bool,
    memory_fraction: float | None = None,
) -> list[int]:
    procs = []
    for exp, cuda_devices in batch:
        cmd, env = build_command(exp, cuda_devices, memory_fraction)
        frac_str = f"  mem_fraction={memory_fraction:.2f}" if memory_fraction else ""
        print(
            f"  [start] {exp.name}  CUDA_VISIBLE_DEVICES={cuda_devices or 'all'}{frac_str}"
        )
        print(f"          {' '.join(cmd)}")
        if not dry_run:
            procs.append(subprocess.Popen(cmd, env=env))
        else:
            procs.append(None)

    exit_codes = []
    for (exp, _), proc in zip(batch, procs):
        if proc is not None:
            proc.wait()
            exit_codes.append(proc.returncode)
            status = (
                "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
            )
            print(f"  [done]  {exp.name}  {status}")
        else:
            exit_codes.append(0)
    return exit_codes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments_yaml", type=Path)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without running"
    )
    args = parser.parse_args()

    experiments, vram_per_run_gb = load_experiments(args.experiments_yaml)
    gpus = get_gpus()
    vram_per_run_mb = vram_per_run_gb * 1024

    print(f"Experiments: {len(experiments)}")

    memory_fraction: float | None = None

    if not gpus:
        print("No GPUs detected — running sequentially on CPU.")
        max_parallel = 1
        multi_gpu = False
    elif len(gpus) > 1:
        max_parallel = min(len(gpus), len(experiments))
        multi_gpu = True
        print(
            f"GPUs: {len(gpus)} detected — assigning one per experiment (max {max_parallel} parallel)"
        )
        for g in gpus:
            print(f"  GPU {g.index}: {g.free_mb} MB free / {g.total_mb} MB total")
    else:
        g = gpus[0]
        max_parallel = max(1, math.floor(g.free_mb / vram_per_run_mb))
        multi_gpu = False
        print(f"GPU 0: {g.free_mb} MB free / {g.total_mb} MB total")
        print(
            f"VRAM per run estimate: {vram_per_run_gb} GB → {max_parallel} parallel slot(s)"
        )
        if max_parallel > 1:
            # Cap each process to its share, with a 5% buffer for the runtime
            memory_fraction = round((1.0 / max_parallel) * 0.95, 4)
            print(f"Per-process VRAM cap: {memory_fraction:.2%}")

    if max_parallel == 1:
        print("Strategy: sequential")
    else:
        print(f"Strategy: parallel (batch size {max_parallel})")

    print()

    all_exit_codes = []
    for i in range(0, len(experiments), max_parallel):
        chunk = experiments[i : i + max_parallel]
        print(f"--- Batch {i // max_parallel + 1} ({len(chunk)} experiment(s)) ---")

        if multi_gpu:
            batch = [
                (exp, str(gpus[j % len(gpus)].index)) for j, exp in enumerate(chunk)
            ]
        else:
            batch = [(exp, None) for exp in chunk]

        codes = run_batch(batch, dry_run=args.dry_run, memory_fraction=memory_fraction)
        all_exit_codes.extend(codes)
        print()

    failed = sum(1 for c in all_exit_codes if c != 0)
    print(
        f"Done. {len(experiments) - failed}/{len(experiments)} experiments succeeded."
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
