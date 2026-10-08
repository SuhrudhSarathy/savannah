# Savannah

Robot learning policies for manipulation tasks, built on [LeRobot](https://github.com/huggingface/lerobot)
datasets and configured with [Hydra](https://hydra.cc/). Supports three simulators — [PushT](https://github.com/huggingface/gym-pusht),
[MetaWorld](https://github.com/Farama-Foundation/Metaworld), and [ManiSkill](https://github.com/haosulab/ManiSkill) — behind a
common task interface.

## Models

All policies live in `src/savannah/models/` and share a common `Policy` interface
(`src/savannah/models/policy.py`) with a pluggable training/inference `objective`
(`src/savannah/objectives.py`: `DDIMObjective`, `FlowMatchingObjective`, and `OverfitObjective` for debugging):

| Model | Config | Vision backbone | Objective |
| --- | --- | --- | --- |
| `DITCrossAttnPolicy` | `configs/model/dit_cross_attn.yaml` | ResNet-18 (spatial tokens) | DDIM |
| `DITCrossAttnPolicy` | `configs/model/dit_cross_attn_flow_matching.yaml` | DINOv3 ViT-S (frozen) | Flow matching |
| `DITCrossAttnPolicy` (default) | `configs/model/dit_cross_attn_with_dino.yaml` | DINOv3 ViT-S (fine-tuned) + attention pooling | DDIM |
| `DITCrossAttnPolicy` | `configs/model/dit_cross_attn_with_dino_l.yaml` | DINOv3 ViT-B (fine-tuned) + attention pooling | DDIM |
| `LBMPolicy` | `configs/model/lbm.yaml` | DINOv3 ViT-S (frozen) | DDIM |
| `LBMPolicy` | `configs/model/lbm_flow_matching.yaml` | DINOv3 ViT-S (frozen) | Flow matching |
| `UNetPolicy` | `configs/model/unet.yaml` | LoRA ResNet-18 | DDIM |
| `UNetPolicy` | `configs/model/unet_flow_matching.yaml` | ResNet-18 | Flow matching |

- **`DITCrossAttnPolicy`** — DiT decoder over the action chunk. Vision, state, and language tokens are
  attended to via cross-attention; the denoising timestep conditions each block through AdaLN.
- **`LBMPolicy`** — DiT decoder where all observation tokens (vision, state, language) and the
  timestep are flattened into a single AdaLN conditioning vector.
- **`UNetPolicy`** — 1D convolutional U-Net over the action chunk with FiLM conditioning.

`src/savannah/models/act.py` (ACT / CVAE policy) is not wired to a config and is not trainable with
`PolicyTrainer` (it has its own `compute_loss` signature).

### Encoders and backbones

Each policy turns camera images into tokens with a `VisionEncoder`
(`src/savannah/models/encoders/vision_encoder.py`). It wraps a vision backbone, adds per-camera
and per-observation-step embeddings, and can optionally compress each image's tokens into a fixed
number with attention pooling (`enable_attn_pooling`, `attn_pooling_num_queries`).

Backbones live in `src/savannah/models/backbones/` and are swapped per model config:

| Backbone | Notes |
| --- | --- |
| `ResNetBackbone` | ImageNet ResNet-18, BatchNorm → GroupNorm, fully fine-tuned. Spatial-softmax keypoints (1 token) or a spatial token grid. |
| `LoRAResNetBackbone` | Same ResNet-18, frozen, with LoRA adapters on every conv. |
| `DinoV3Backbone` | Hugging Face DINOv3. CLS token only (`use_eos_only`) or CLS + patch tokens; frozen or `trainable`. |
| `CLIPBackbone` | Hugging Face CLIP vision tower, same options as DINOv3. |
| `DummyVisionBackbone` | Tiny randomly initialised conv — no downloads. Used by the tests. |

Language conditioning uses `LanguageEncoder` (frozen CLIP text model). No model config enables it by
default (`language_encoder: null`). Tasks that provide instructions (e.g. ManiSkill color matching,
`use_language: true`) need it switched on from the command line:

```bash
uv run python scripts/train.py model=dit_cross_attn task=maniskill_color_matching \
  'model.language_encoder={_target_:savannah.models.encoders.LanguageEncoder,embed_dim:${model.embed_dim}}'
```

## Tasks

Each simulator is a `BaseRobotTask` (`src/savannah/tasks/base.py`) that wraps a LeRobot dataset for
training and a gym env for sim-based eval:

| Task | Config | Dataset | Env |
| --- | --- | --- | --- |
| PushT | `pusht.yaml` | `lerobot/pusht` | `gym_pusht` |
| MetaWorld — assembly | `metaworld_assembly.yaml` | `suhrudhsarathy/metaworld-assembly-v3` | `metaworld` |
| MetaWorld — disassemble | `metaworld_disassemble.yaml` | `suhrudhsarathy/metaworld-disassemble-v3` | `metaworld` |
| MetaWorld — hammer | `metaworld_hammer.yaml` | `suhrudhsarathy/metaworld-hammer-v3` | `metaworld` |
| MetaWorld — drawer open | `metaworld_drawer_open.yaml` | `suhrudhsarathy/metaworld-drawer-open-v3` | `metaworld` |
| MetaWorld — drawer close | `metaworld_drawer_close.yaml` | `suhrudhsarathy/metaworld-drawer-close-v3` | `metaworld` |
| MetaWorld — button press | `metaworld_button_press.yaml` | `suhrudhsarathy/metaworld-button-press-v3` | `metaworld` |
| MetaWorld — button press (multi-camera) | `metaworld_multi_button_press.yaml` | `suhrudhsarathy/metaworld-button-press-multi-v3` | `metaworld` |
| ManiSkill — color matching (pick colored cube into matching bin) | `maniskill_color_matching.yaml` | `suhrudhsarathy/color-pick-and-place` | `mani_skill` (custom `ColorMatchingEnv` in `src/savannah/envs/color_matching.py`) |

All task configs live in `configs/task/`. Train-time image and state augmentation (crop, color
jitter, speckle noise, state noise, random camera/state masking, all decaying over training) is
configured in `configs/augmentation/default.yaml` (`src/savannah/data/augmentation.py`).

Datasets are collected with the scripts in `scripts/data/` and optionally pushed to the Hub:

```bash
uv run python scripts/data/collect_metaworld_data.py --task assembly-v3 --repo-id <hf-repo> --push-to-hub
uv run python scripts/data/collect_maniskill_data.py --repo-id <hf-repo> --push-to-hub
```

Both accept `--num-episodes`, `--cameras`, `--image-size`, `--fps`, `--keep-failed` and
`--push-only` (push an already-recorded local dataset).

## Installation

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
# install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# core dependencies + dev tools (pytest, ruff, pre-commit) into a local .venv
uv sync

# simulators (gym-pusht, MetaWorld, ManiSkill) — needed for any of the tasks above,
# for sim eval during training, and for scripts/eval/
uv sync --extra sim
```

`torchcodec` (a LeRobot dependency) needs the FFmpeg shared libraries (FFmpeg 4–7) on the system,
e.g. `brew install ffmpeg` or `sudo apt-get install ffmpeg`. `torch` comes from PyPI; on Linux its
wheels bundle CUDA 13, which needs an NVIDIA driver >= 580.

`infra/setup.sh` installs uv and FFmpeg, runs `uv sync --extra sim`, and logs into Weights & Biases:

```bash
WANDB_API_KEY=<your-key> bash infra/setup.sh
```

If you skip that, log into W&B manually before training (or set `wandb.enable_logging=false`):

```bash
uv run wandb login <your-key>
```

To run ruff on every commit:

```bash
uv run pre-commit install
```

## Configuration

Configuration is composed with Hydra from `configs/`:

- `configs/config.yaml` — top-level config for `scripts/train.py`. Defaults: `model: dit_cross_attn_with_dino`,
  `task: metaworld_assembly`, `trainer: default`, `augmentation: default`.
- `configs/model/*.yaml` — policy architecture, vision backbone, and objective.
- `configs/task/*.yaml` — dataset/task config (cameras, observation/action horizons, image size, batch size).
- `configs/trainer/default.yaml` — optimizer and LR schedule (`trainer.train.*`), validation and sim-eval
  frequency (`trainer.eval.*`), and checkpointing (`trainer.checkpoint.*`).
- `configs/augmentation/default.yaml` — train-time augmentation.
- `configs/eval.yaml` — config for `scripts/eval/*.py`.
- `configs/overfit.yaml` — config for `scripts/debug/overfit_dataset_batch.py`.
- `configs/attention_viz.yaml` — config for `scripts/debug/visualize_cross_attention.py`.
- `configs/experiments.yaml` — batch of runs for `infra/multi_run.py`.

Any field can be overridden from the command line, e.g.
`model=dit_cross_attn task=metaworld_hammer trainer.train.lr=1e-4`.

## Training

```bash
uv run python scripts/train.py
```

Pick a model/task and customize the run:

```bash
uv run python scripts/train.py model=dit_cross_attn_flow_matching task=pusht
uv run python scripts/train.py model=unet task=metaworld_assembly trainer.train.training_steps=50000

# disable W&B logging
uv run python scripts/train.py wandb.enable_logging=false

# resume from a checkpoint (restores model, EMA, optimizer, scheduler, and step)
uv run python scripts/train.py trainer.checkpoint.resume_from=checkpoints/<run>/latest/latest.ckpt
```

The trainer uses AdamW (with a separate `trainer.train.vision_lr` for the vision encoder), gradient
accumulation, bf16 autocast, gradient clipping, and an EMA copy of the weights. Every
`trainer.eval.eval_freq` steps it computes validation loss and action-reconstruction error. Every
`trainer.eval.sim_eval_freq` steps it rolls out the EMA policy in the simulator (when
`trainer.eval.eval_using_sim=true`) and logs success rate and a video.

The device is picked automatically (CUDA → MPS → CPU). Set `SAVANNAH_DEVICE` (e.g. `SAVANNAH_DEVICE=cpu`)
to force one.

Checkpoints are written to `trainer.checkpoint.checkpoint_dir` (default `checkpoints/<model>_<task>`).
Each save writes to two places: a permanent per-step archive
(`checkpoint_dir/{global_step}/{name}.ckpt`, never overwritten) and a rolling
`checkpoint_dir/latest/{name}.ckpt` mirror of the most recent checkpoint of
each kind (`latest`, `best_model` by validation reconstruction error, `best_model_success` by sim
success rate). Pressing Ctrl-C saves `latest.ckpt` before exiting.

`best_model`, `best_model_success`, and `latest` checkpoints (from the `latest/`
mirror) can additionally be uploaded as **W&B artifacts** — controlled by
`wandb.enable_model_checkpoint` (default `false`, requires
`wandb.enable_logging=true`).

### Running multiple experiments

`configs/experiments.yaml` defines a set of training runs (a name plus Hydra overrides each). Run them
all, distributing across available GPU VRAM (`vram_per_run_gb`):

```bash
uv run python infra/multi_run.py configs/experiments.yaml
uv run python infra/multi_run.py configs/experiments.yaml --dry-run
```

Each experiment's `name` becomes the run's Hydra `name`. Its W&B run name, artifact name, and
checkpoint dir (`checkpoints/<name>`) all follow it. Set `checkpoint_root` in the YAML to put every run's
checkpoints under `<checkpoint_root>/<name>` instead (e.g. on a larger data drive).

### Cloud / GPU box helpers

`infra/run_train.sh` runs a single training job and can optionally tear down a vast.ai instance
afterwards (including when training fails); `infra/run_multi_train.sh` does the same for a batch
(`infra/multi_run.py`) run:

```bash
bash infra/run_train.sh model=unet trainer.train.training_steps=75000
AUTO_DESTROY=true VAST_INSTANCE_ID=12345 bash infra/run_train.sh model=unet

AUTO_DESTROY=true VAST_INSTANCE_ID=12345 bash infra/run_multi_train.sh configs/experiments.yaml
```

## Evaluation

Evaluate a trained policy in the simulator and record videos to `video_folder` (default
`eval_videos/<model>_<task>`). Pass the same `model`/`task` the checkpoint was trained with:

```bash
# from a local checkpoint
uv run python scripts/eval/eval.py model=unet task=pusht checkpoint_path=checkpoints/unet_pusht/latest/latest.ckpt

# from a W&B model artifact
uv run python scripts/eval/eval.py model=dit_cross_attn task=metaworld_assembly wandb_artifact=savannah/dit_cross_attn_metaworld_assembly:best
```

Useful overrides: `num_episodes`, `execute_steps` (receding-horizon action chunk size), `use_ema`
(default `true`), `video_folder`, `pick_color`/`place_color` (ManiSkill color matching only).

To inspect a single dataset episode (predicted vs. ground-truth actions) instead of running the
simulator:

```bash
uv run python scripts/eval/eval_single_episode.py model=unet task=pusht checkpoint_path=... episode_index=3
```

To download a checkpoint from a W&B model artifact without running eval:

```bash
uv run python scripts/data/download_artifact.py savannah/dit_cross_attn_metaworld_assembly:best --output-dir artifacts/
```

To stitch a folder of per-episode eval videos into a single annotated clip:

```bash
uv run python scripts/eval/concat_videos.py --folder eval_videos --output combined.mp4 --title "Assembly — DiT+DDIM"
```

## Debugging

Check that a model/objective combination can overfit one real batch from the dataset before
committing to a full run. With `overfit.frame_index=null` it cycles through every frame of one
episode instead. It reports PASS/FAIL against `overfit.loss_threshold` and
`overfit.action_mse_threshold`:

```bash
uv run python scripts/debug/overfit_dataset_batch.py model=unet task=pusht
uv run python scripts/debug/overfit_dataset_batch.py overfit.episode_index=3 overfit.frame_index=null
```

Visualize what a trained `DITCrossAttnPolicy` attends to: cross-attention by modality, cross-attention
traced back to image patches through the attention pooler, and self-attention over the action chunk.
Plots are written to `output_dir`:

```bash
uv run python scripts/debug/visualize_cross_attention.py checkpoint_path=... episode_index=0 frame_index=10
```

## Tests

Tests are split into tiers by directory (see `tests/conftest.py`):

```bash
uv run pytest                    # unit tests — dummy backbone, no downloads, no simulators (~10s)
uv run pytest -m smoke           # end-to-end training runs through scripts/*.py (~1.5 min)
uv run pytest -m integration     # real pretrained backbones and Hub datasets (downloads on first run)
uv run pytest -m sim             # rollouts in the real simulators (needs the `sim` extra)
uv run pytest -m ""              # everything
```

| Directory | Tier | Covers |
| --- | --- | --- |
| `tests/models/`, `tests/nn/`, `tests/trainer/` | unit | policies, vision encoder, attention modules, checkpoint save/resume |
| `tests/smoke/` | smoke | `scripts/train.py` for every config in `configs/model/` (plus resume and the overfit script) on a synthetic task, with a dummy backbone and shrunk model |
| `tests/backbones/`, `tests/integration/` | integration | each real backbone on its own, and every policy × backbone × language-encoder combination |
| `tests/data/` | integration | LeRobot dataset loading and train/val episode splits |
| `tests/eval/` | sim | the sim-eval rollout loop on each simulator |

Shared download-free stand-ins (stub language encoder, synthetic task and env) live in
`tests/dummies.py`.

## Logging

Training and eval logs are written under `logs/`. Set `log_level` (`DEBUG | INFO | WARNING | ERROR`)
on any script to control verbosity, e.g. `uv run python scripts/train.py log_level=DEBUG`. At `DEBUG`,
models also log shape/min/max/mean/NaN stats for intermediate tensors (`savannah.utils.debug.debug_stat`).
