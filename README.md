# Savannah

Robot learning policies for manipulation tasks, built on [LeRobot](https://github.com/huggingface/lerobot) datasets/environments and configured with [Hydra](https://hydra.cc/).

## Architecture

Policies are composed from three independent, swappable components:

```
ModularPolicy
├── ConditionEncoder   (images + state → condition tokens)
│   ├── VisionEncoder  (backbone → patch tokens, projected to embed_dim)
│   ├── camera_embedding, state_embedding, positional encoding
│   └── optional: LanguageEncoder + FiLM  (language-conditioned vision)
├── ActionHead         (noisy_actions + cond_tokens + timestep → denoised actions)
│   ├── DiTActionHead            (SA encoder + AdaLN decoder, per-layer conditioning)
│   ├── CrossAttentionActionHead (cross-attn to cond tokens, AdaLN timestep)
│   └── UNetActionHead           (1D conv U-Net, FiLM conditioning)
└── PolicyObjective    (noise schedule, loss, inference sampler)
    ├── FlowMatchingObjective    (linear interpolation, iterative ODE)
    ├── DDIMObjective            (diffusion noise schedule, DDIM sampler)
    └── OverfitObjective         (MSE, for sanity checks)
```

Any `ActionHead` works with any `PolicyObjective`. Benchmarking a new combination
requires only a new config file — no code changes.

### Action Heads

| Class | Mechanism | When to use |
|---|---|---|
| `DiTActionHead` | SA encodes cond tokens per layer; AdaLN from `cat(mean(enc_i), t)` | Strong baseline; condition re-encoded internally |
| `CrossAttentionActionHead` | Cross-attends to cond tokens; AdaLN from timestep only | Preserves backbone token structure; preferred with pretrained encoders |
| `UNetActionHead` | 1D conv U-Net; FiLM from `cat(mean(cond_tokens), t)` | Smaller memory footprint, fast inference |

> **Note on `DiTActionHead` vs `CrossAttentionActionHead` with pretrained backbones:**
> The DiT head re-encodes condition tokens through its own internal SA encoder, so a VLM's
> pretrained contextual structure is re-processed from scratch. The CrossAttn head uses
> condition tokens directly as K/V and preserves that structure. This is a meaningful
> architectural distinction when swapping in a pretrained vision backbone.

### Condition Encoder

All heads share the same `ConditionEncoder`, which:
- Encodes each camera stream through a `VisionEncoder` (backbone → linear projection)
- Adds per-frame time embeddings and per-camera embeddings
- Embeds proprioceptive state with positional encoding
- Concatenates to `(B, N_cond, embed_dim)`

Optional language conditioning: provide a `LanguageEncoder` subclass and a `FiLM` module
to apply language-conditioned scale+shift to vision tokens before concatenation.

```python
# LanguageEncoder ABC — implement this for your text encoder
class LanguageEncoder(ABC, nn.Module):
    @property
    @abstractmethod
    def output_dim(self) -> int: ...

    @abstractmethod
    def forward(self, tokens: Tensor) -> Tensor:
        # (B, N_text) → (B, output_dim)
        ...
```

### Objectives and `time_scale`

`FlowMatchingObjective` emits timesteps in `[0, 1]`; `DDIMObjective` emits them in `[0, 100]`.
`TimeEmbedding` is calibrated for the `[0, 100]` range, so set `time_scale` accordingly in
the model config:

| Objective | `time_scale` |
|---|---|
| `FlowMatchingObjective` | `100.0` |
| `DDIMObjective` | `1.0` (default) |

## Configs

Each experiment is a model config under `configs/model/` that fully specifies the backbone,
action head, and objective. The `${model.embed_dim}` interpolation propagates a single
`embed_dim` to all sub-components.

```yaml
# configs/model/cross_attn_fm.yaml
_target_: savannah.models.modular_policy.ModularPolicy
name: cross_attn_fm_resnet
embed_dim: 256
action_dim: 4
action_horizon: 32
state_dim: 4
num_cameras: 1
time_scale: 100.0

condition_encoder:
  _target_: savannah.models.condition_encoder.ConditionEncoder
  embed_dim: ${model.embed_dim}
  state_dim: ${model.state_dim}
  num_cameras: ${model.num_cameras}
  vision_encoder:
    _target_: savannah.models.vision_encoder.VisionEncoder
    embed_dim: ${model.embed_dim}
    backbone:
      _target_: savannah.models.backbones.resnet_backbone.ResNetBackbone
      out_channels: 96

action_head:
  _target_: savannah.models.action_heads.CrossAttentionActionHead
  embed_dim: ${model.embed_dim}
  num_blocks: 4
  num_attn_heads: 4
  feedforward_dim: 512
  action_dim: ${model.action_dim}
  action_horizon: ${model.action_horizon}

objective:
  _target_: savannah.objectives.FlowMatchingObjective
  inference_steps: 10
```

W&B run names are set from `model.name`, so each config produces a labelled, filterable run.
The full `_target_` paths for `action_head`, `objective`, and `backbone` appear as separate
fields in the W&B run config, enabling group-by on any axis.

### Deprecated models

`DiTBlockPolicy`, `FlowMatchingPolicy`, and `UNetPolicy` are kept for checkpoint compatibility
but emit a `DeprecationWarning` on instantiation. They will be removed once existing checkpoints
are migrated to `ModularPolicy`.

## Installation

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

`infra/setup.sh` automates the above and configures Weights & Biases:

```bash
WANDB_API_KEY=<your-key> bash infra/setup.sh
```

If you skip that, log in manually before training:

```bash
uv run wandb login <your-key>
```

## Training

```bash
uv run python scripts/train.py
```

Override any field from the command line:

```bash
uv run python scripts/train.py model=cross_attn_fm
uv run python scripts/train.py model=dit_adaln_ddim trainer.lr=1e-4
uv run python scripts/train.py wandb.enable_logging=false
```

Checkpoints are written to `trainer.checkpoint_dir` (default `checkpoints/`).

`best_model`, `best_model_success`, and `latest` checkpoints can be uploaded to:

- **W&B artifacts** — controlled by `wandb.enable_model_checkpoint` (default `true`).
- **HuggingFace Hub** — controlled by `hf.*` config fields.

### Running multiple experiments

`configs/experiments.yaml` defines a benchmark matrix of training runs. Run them all,
distributed across available GPU VRAM:

```bash
uv run python infra/multi_run.py configs/experiments.yaml
uv run python infra/multi_run.py configs/experiments.yaml --dry-run
```

### Cloud / GPU box helper

```bash
bash infra/run_train.sh model=cross_attn_fm trainer.training_steps=75000
AUTO_DESTROY=true VAST_INSTANCE_ID=12345 bash infra/run_train.sh model=cross_attn_fm
```

## Evaluation

```bash
# from a local checkpoint
uv run python scripts/eval.py model=cross_attn_fm checkpoint_path=checkpoints/cross_attn_fm/latest.ckpt

# from a W&B model artifact
uv run python scripts/eval.py model=cross_attn_fm wandb_artifact=savannah/cross_attn_fm:best
```

Useful overrides: `num_episodes`, `execute_steps` (receding-horizon chunk size), `use_ema`.

To inspect predicted vs. ground-truth actions for a single dataset episode:

```bash
uv run python scripts/eval_single_episode.py model=cross_attn_fm checkpoint_path=... episode_index=3
```

## Sanity checks

```bash
# overfit a single batch of random data (architecture + objective, no dataset)
uv run python scripts/overfit_random.py model=cross_attn_fm

# overfit a single real batch from the dataset
uv run python scripts/overfit_dataset_batch.py model=cross_attn_fm

# overfit a single (state, action) pair with no noise
uv run python scripts/overfit_single_noise_learning.py model=cross_attn_fm
```

## Tests

```bash
uv run pytest
```

## Logging

Training and eval logs are written under `logs/`. Control verbosity with `log_level`
(`DEBUG | INFO | WARNING | ERROR`):

```bash
uv run python scripts/train.py log_level=DEBUG
```
