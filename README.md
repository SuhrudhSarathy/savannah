# Savannah

Robot learning policies for manipulation tasks (currently [PushT](https://github.com/huggingface/gym-pusht)), built on
[LeRobot](https://github.com/huggingface/lerobot) datasets/environments and configured with [Hydra](https://hydra.cc/).

## Models

All policies live in `src/savannah/models/` and share a common `Policy` interface
(`src/savannah/models/policy.py`) with a pluggable training/inference `objective`
(`src/savannah/objectives.py`):

| Model | Config | Objective |
| --- | --- | --- |
| `DiTBlockPolicy` | `configs/model/dit_block.yaml` | DDIM diffusion |
| `FlowMatchingPolicy` | `configs/model/flow_matching.yaml` | Flow matching |
| `UNetPolicy` (1D conv U-Net) | `configs/model/unet.yaml` | DDIM diffusion |
| `UNetPolicy` (1D conv U-Net) | `configs/model/unet_flow_matching.yaml` | Flow matching |
| `DumbPolicy` (sanity check) | `configs/model/dumb_model.yaml` | Overfit |

## Installation

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
# install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# install dependencies into a local .venv
uv sync
```

`infra/setup.sh` automates the steps above and also configures Weights & Biases:

```bash
WANDB_API_KEY=<your-key> bash infra/setup.sh
```

If you skip that, log into W&B manually before training (or set `wandb.enable_logging=false`):

```bash
uv run wandb login <your-key>
```

## Configuration

Configuration is composed with Hydra from `configs/`:

- `configs/config.yaml` — top-level config for `scripts/train.py`. Defaults: `model: dit_block`, `task: pusht`, `trainer: default`.
- `configs/model/*.yaml` — policy architecture + objective.
- `configs/task/pusht.yaml` — dataset/task config (PushT).
- `configs/trainer/default.yaml` — optimizer, schedule, eval frequency, checkpoint location.
- `configs/eval.yaml` — config for `scripts/eval*.py`.
- `configs/overfit.yaml` — config for the overfit sanity-check scripts.

Any field can be overridden from the command line, e.g. `model=flow_matching trainer.lr=1e-4`.

## Training

```bash
uv run python scripts/train.py
```

Pick a model and customize the run:

```bash
uv run python scripts/train.py model=flow_matching
uv run python scripts/train.py model=unet trainer.training_steps=50000

# disable W&B logging
uv run python scripts/train.py wandb.enable_logging=false
```

Checkpoints are written to `trainer.checkpoint_dir` (default `checkpoints/`).

`best_model`, `best_model_success`, and `latest` checkpoints can additionally be
uploaded to:

- **W&B artifacts** — controlled by `wandb.enable_model_checkpoint` (default `true`,
  requires `wandb.enable_logging=true`).
- **Google Cloud Storage** — set `gcp.enable_model_checkpoint=true` and
  `gcp.bucket=<bucket-name>` (requires GCS credentials via Application Default
  Credentials — run `gcloud auth application-default login` or set
  `GOOGLE_APPLICATION_CREDENTIALS`). Checkpoints land at
  `gs://<bucket>/<trainer.artifact_name>_<run-timestamp>/`.

### Running multiple experiments

`configs/experiments.yaml` defines a set of training runs (one per model). Run them all,
distributing across available GPU VRAM:

```bash
uv run python infra/multi_run.py configs/experiments.yaml
uv run python infra/multi_run.py configs/experiments.yaml --dry-run
```

### Cloud / GPU box helper

`infra/run_train.sh` runs training and can optionally tear down a vast.ai instance afterwards:

```bash
bash infra/run_train.sh model=unet trainer.training_steps=75000
AUTO_DESTROY=true VAST_INSTANCE_ID=12345 bash infra/run_train.sh model=unet
```

## Evaluation

Evaluate a trained policy in the PushT simulator and record videos to `eval_videos/`:

```bash
# from a local checkpoint
uv run python scripts/eval.py model=unet checkpoint_path=checkpoints/unet_pusht/latest.ckpt

# from a W&B model artifact
uv run python scripts/eval.py model=flow_matching wandb_artifact=savannah/flow_matching_pusht:best
```

Useful overrides: `num_episodes`, `execute_steps` (receding-horizon action chunk size), `use_ema`.

To inspect a single dataset episode (predicted vs. ground-truth actions) instead of running the
simulator:

```bash
uv run python scripts/eval_single_episode.py model=unet checkpoint_path=... episode_index=3
```

## Sanity checks

Before a full training run, these scripts check that a model/objective combination can learn
at all:

```bash
# 1. Overfit a single batch of random data (architecture + objective only, no dataset)
uv run python scripts/overfit_random.py model=unet

# 2. Overfit a single real batch from the dataset
uv run python scripts/overfit_dataset_batch.py model=unet

# 3. Overfit a single (state, action) pair with no noise (architecture only)
uv run python scripts/overfit_single_noise_learning.py model=unet
```

## Tests

```bash
uv run pytest
```

## Logging

Training and eval logs are written under `logs/`. Set `log_level` (`DEBUG | INFO | WARNING | ERROR`)
on any script to control verbosity, e.g. `uv run python scripts/train.py log_level=DEBUG`.
