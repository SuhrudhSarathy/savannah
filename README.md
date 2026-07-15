# Savannah

Robot learning policies for manipulation tasks, built on [LeRobot](https://github.com/huggingface/lerobot)
datasets and configured with [Hydra](https://hydra.cc/). Supports three simulators — [PushT](https://github.com/huggingface/gym-pusht),
[MetaWorld](https://github.com/Farama-Foundation/Metaworld), and [ManiSkill](https://github.com/haosulab/ManiSkill) — behind a
common task interface.

## Models

All policies live in `src/savannah/models/` and share a common `Policy` interface
(`src/savannah/models/policy.py`) with a pluggable training/inference `objective`
(`src/savannah/objectives.py`):

| Model | Config | Objective |
| --- | --- | --- |
| `DiTAdaLNPolicy` (default) | `configs/model/dit_adaln.yaml` | DDIM diffusion |
| `DiTAdaLNPolicy` | `configs/model/dit_adaln_flow_matching.yaml` | Flow matching |
| `DiTCrossAttnPolicy` | `configs/model/dit_cross_attn.yaml` | DDIM diffusion |
| `DiTCrossAttnPolicy` | `configs/model/dit_cross_attn_flow_matching.yaml` | Flow matching |
| `DiTPolicy` (LoRA ResNet backbone) | `configs/model/dit_lora_ddim.yaml` | DDIM diffusion |
| `UNetPolicy` (1D conv U-Net) | `configs/model/unet.yaml` | DDIM diffusion |
| `UNetPolicy` (1D conv U-Net) | `configs/model/unet_flow_matching.yaml` | Flow matching |

Vision backbones (`src/savannah/models/backbones/`) are pluggable per model config: `ResNetBackbone`
for full fine-tuning, or `LoRAResNetBackbone` to fine-tune a pretrained ResNet with low-rank adapters
instead of updating all its weights.

`src/savannah/models/act.py` (ACT / CVAE policy) is an in-progress addition, not yet wired to a config.

## Tasks

Each simulator is a `BaseRobotTask` (`src/savannah/tasks/base.py`) that wraps a LeRobot dataset for
training and a gym env for sim-based eval:

| Task | Config | Env |
| --- | --- | --- |
| PushT | `configs/task/pusht.yaml` | `gym_pusht` |
| MetaWorld — assembly | `configs/task/metaworld_assembly.yaml` | `metaworld` |
| MetaWorld — button press | `configs/task/metaworld_button_press.yaml` | `metaworld` |
| ManiSkill — color matching (pick colored cube into matching bin) | `configs/task/maniskill_color_matching.yaml` | `mani_skill` (custom `ColorMatchingEnv` in `src/savannah/envs/color_matching.py`) |

Datasets are collected/pushed to the Hub with the scripts in `scripts/data/`:

```bash
uv run python scripts/data/collect_metaworld_data.py --env-name assembly-v3 --repo-id <hf-repo>
uv run python scripts/data/collect_maniskill_data.py --repo-id <hf-repo>
```

## Installation

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
# install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# install dependencies into a local .venv
uv sync
```

`infra/setup.sh` automates the steps above (including the FFmpeg system libraries required by
`torchcodec`, a LeRobot dependency) and also configures Weights & Biases:

```bash
WANDB_API_KEY=<your-key> bash infra/setup.sh
```

If you skip that, log into W&B manually before training (or set `wandb.enable_logging=false`):

```bash
uv run wandb login <your-key>
```

## Configuration

Configuration is composed with Hydra from `configs/`:

- `configs/config.yaml` — top-level config for `scripts/train.py`. Defaults: `model: dit_adaln`, `task: pusht`, `trainer: default`.
- `configs/model/*.yaml` — policy architecture + objective.
- `configs/task/*.yaml` — dataset/task config (PushT, MetaWorld, ManiSkill).
- `configs/trainer/default.yaml` — optimizer, schedule, eval frequency, checkpoint location.
- `configs/eval.yaml` — config for `scripts/eval/*.py`.
- `configs/overfit.yaml` / `configs/overfit_dit_lora_ddim.yaml` — configs for the overfit sanity-check scripts.
- `configs/experiments.yaml` — batch of runs for `infra/multi_run.py`.

Any field can be overridden from the command line, e.g. `model=dit_cross_attn task=metaworld_assembly trainer.lr=1e-4`.

## Training

```bash
uv run python scripts/train.py
```

Pick a model/task and customize the run:

```bash
uv run python scripts/train.py model=dit_cross_attn_flow_matching
uv run python scripts/train.py model=unet task=metaworld_assembly trainer.training_steps=50000

# disable W&B logging
uv run python scripts/train.py wandb.enable_logging=false
```

Checkpoints are written to `trainer.checkpoint_dir` (default `checkpoints/dit_adaln_pusht`).

`best_model`, `best_model_success`, and `latest` checkpoints can additionally be
uploaded as **W&B artifacts** — controlled by `wandb.enable_model_checkpoint`
(default `false`, requires `wandb.enable_logging=true`).

### Running multiple experiments

`configs/experiments.yaml` defines a set of training runs. Run them all, distributing across
available GPU VRAM:

```bash
uv run python infra/multi_run.py configs/experiments.yaml
uv run python infra/multi_run.py configs/experiments.yaml --dry-run
```

### Cloud / GPU box helpers

`infra/run_train.sh` runs a single training job and can optionally tear down a vast.ai instance
afterwards; `infra/run_multi_train.sh` does the same for a batch (`infra/multi_run.py`) run:

```bash
bash infra/run_train.sh model=unet trainer.training_steps=75000
AUTO_DESTROY=true VAST_INSTANCE_ID=12345 bash infra/run_train.sh model=unet

AUTO_DESTROY=true VAST_INSTANCE_ID=12345 bash infra/run_multi_train.sh configs/experiments.yaml
```

## Evaluation

Evaluate a trained policy in the simulator and record videos to `eval_videos/`:

```bash
# from a local checkpoint
uv run python scripts/eval/eval.py model=unet checkpoint_path=checkpoints/unet_pusht/latest.ckpt

# from a W&B model artifact
uv run python scripts/eval/eval.py model=dit_cross_attn_flow_matching wandb_artifact=savannah/dit_cross_attn_flow_matching_pusht:best

# on a different task
uv run python scripts/eval/eval.py task=metaworld_assembly checkpoint_path=...
```

Useful overrides: `num_episodes`, `execute_steps` (receding-horizon action chunk size), `use_ema`,
`pick_color`/`place_color` (ManiSkill ColorMatching only).

To inspect a single dataset episode (predicted vs. ground-truth actions) instead of running the
simulator:

```bash
uv run python scripts/eval/eval_single_episode.py model=unet checkpoint_path=... episode_index=3
```

To download a checkpoint from a W&B model artifact without running eval:

```bash
uv run python scripts/data/download_artifact.py savannah/dit_adaln_pusht:best
```

To stitch a folder of per-episode eval videos into a single annotated clip:

```bash
uv run python scripts/eval/concat_videos.py --folder eval_videos --output combined.mp4 --title "Assembly — DiT+DDIM"
```

## Sanity checks

Before a full training run, these scripts check that a model/objective combination can learn
at all:

```bash
# 1. Overfit a single batch of random data (architecture + objective only, no dataset)
uv run python scripts/debug/overfit_random.py model=unet

# 2. Overfit a single real batch from the dataset
uv run python scripts/debug/overfit_dataset_batch.py model=unet

# 3. Overfit a single (state, action) pair with no noise (architecture only)
uv run python scripts/debug/overfit_single_noise_learning.py model=unet
```

## Tests

```bash
uv run pytest
```

Covers model architectures (`tests/models/`), dataset loading (`tests/data/`), and sim-based
eval rollouts across all three tasks (`tests/eval/`).

## Logging

Training and eval logs are written under `logs/`. Set `log_level` (`DEBUG | INFO | WARNING | ERROR`)
on any script to control verbosity, e.g. `uv run python scripts/train.py log_level=DEBUG`.
