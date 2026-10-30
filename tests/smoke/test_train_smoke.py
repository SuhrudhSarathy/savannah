"""Smoke tests for the full training pipeline, driven through the real
entrypoint scripts (scripts/train.py, scripts/debug/overfit_dataset_batch.py).

Every model config in configs/model/ is composed by Hydra exactly as a real run
would, then only three things are swapped so it runs in seconds on a CPU with
no network access:
  * task=synthetic        -> tests/dummies.py:SyntheticTask (random data + env)
  * the vision backbone   -> savannah.models.backbones.DummyVisionBackbone
  * model width/depth     -> shrunk (see SHRINK_OVERRIDES)

Each run trains for a handful of steps with validation, sim eval, and
checkpointing all scheduled to fire, so a passing test means: config
composition + instantiation, format_batch/augmentation, forward/backward under
autocast, grad accumulation, EMA, the val loop, evaluate_and_log, and
checkpoint save/resume all work end to end. It says nothing about whether the
model learns.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from savannah.bundle import BundleContents, read_bundle

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs"
SMOKE_CONFIGS = Path(__file__).resolve().parent / "configs"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"
OVERFIT_SCRIPT = REPO_ROOT / "scripts" / "debug" / "overfit_dataset_batch.py"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval" / "eval.py"

MODEL_CONFIGS = sorted(p.stem for p in (CONFIGS / "model").glob("*.yaml"))

DUMMY_BACKBONE = (
    "{_target_:savannah.models.backbones.DummyVisionBackbone,"
    "out_channels:16,tokens_per_image:4}"
)

# Keep every architecture's real wiring but at toy size. Keyed by the model
# config's _target_ so a new config of an existing architecture is picked up
# automatically; a new architecture needs an entry here.
SHRINK_OVERRIDES = {
    "savannah.models.dit_cross_attn.DITCrossAttnPolicy": [
        "model.embed_dim=32",
        "model.decoder_num_blocks=1",
    ],
    "savannah.models.lbm.LBMPolicy": [
        "model.embed_dim=32",
        "model.time_embed_dim=8",
        "model.state_embed_dim=8",
        "model.decoder_num_blocks=1",
    ],
    "savannah.models.unet.UNetPolicy": [
        "model.embed_dim=32",
        "model.unet_channels=[16,32]",
        "model.num_residual_layers=1",
    ],
}

# 4 optimizer-loop steps with grad accumulation 2; val + checkpoint at step 2,
# sim eval (+ best_model_success checkpoint) at step 3.
TRAIN_STEPS = 4
TRAINER_OVERRIDES = [
    f"trainer.train.training_steps={TRAIN_STEPS}",
    "trainer.train.warmup_steps=2",
    "trainer.train.gradient_accumulation_steps=2",
    "trainer.eval.eval_freq=2",
    "trainer.eval.eval_using_sim=true",
    "trainer.eval.sim_eval_freq=3",
    "trainer.eval.eval_episodes=1",
    "trainer.eval.eval_execute_steps=4",
    "wandb.enable_logging=false",
]


def model_overrides(model: str) -> list[str]:
    target = OmegaConf.load(CONFIGS / "model" / f"{model}.yaml")["_target_"]
    assert target in SHRINK_OVERRIDES, (
        f"configs/model/{model}.yaml targets {target}, which has no entry in "
        "SHRINK_OVERRIDES -- add one so the smoke test can run it at toy size."
    )
    return [
        f"model={model}",
        "task=synthetic",
        "~model.vision_encoder.backbone",
        f"+model.vision_encoder.backbone={DUMMY_BACKBONE}",
        *SHRINK_OVERRIDES[target],
    ]


def run_script(script: Path, overrides: list[str], workdir: Path) -> str:
    """Runs `script` as a subprocess (exactly as from the CLI) and returns its
    combined stdout/stderr, failing the test with the tail of it on error."""
    env = {
        **os.environ,
        # tests/ so Hydra can import dummies.SyntheticTask.
        "PYTHONPATH": os.pathsep.join(
            [str(REPO_ROOT / "src"), str(REPO_ROOT / "tests")]
            + ([os.environ["PYTHONPATH"]] if "PYTHONPATH" in os.environ else [])
        ),
        "WANDB_MODE": "disabled",
        "HYDRA_FULL_ERROR": "1",
        # Toy-sized models: CPU beats GPU warm-up, and results don't depend
        # on which accelerator the machine happens to have.
        "SAVANNAH_DEVICE": "cpu",
    }
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config-dir",
            str(SMOKE_CONFIGS),
            f"hydra.run.dir={workdir / 'hydra'}",
            *overrides,
        ],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, (
        f"{script.name} exited with {proc.returncode}:\n{output[-5000:]}"
    )
    return output


def load_checkpoint(path: Path) -> BundleContents:
    assert path.exists(), f"expected bundle at {path}"
    return read_bundle(path)


def assert_finite_state_dict(state_dict: dict):
    bad = [
        k
        for k, v in state_dict.items()
        if v.is_floating_point() and not torch.isfinite(v).all()
    ]
    assert not bad, f"non-finite weights after training: {bad}"


@pytest.mark.parametrize("model", MODEL_CONFIGS)
def test_train_script_runs_end_to_end(model, tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    output = run_script(
        TRAIN_SCRIPT,
        [
            *model_overrides(model),
            *TRAINER_OVERRIDES,
            f"trainer.checkpoint.checkpoint_dir={ckpt_dir}",
        ],
        tmp_path,
    )

    assert "Training Complete!" in output
    for name in ("latest", "best_model", "best_model_success"):
        bundle = load_checkpoint(ckpt_dir / "latest" / f"{name}.bundle")
        assert_finite_state_dict(bundle.training_state["model_state_dict"])
        assert_finite_state_dict(bundle.ema_weights)

    assert (
        load_checkpoint(ckpt_dir / "latest" / "latest.bundle").training_state[
            "global_step"
        ]
        == 2
    )


@pytest.mark.parametrize("model", MODEL_CONFIGS)
def test_train_bundle_loads_and_evals_without_overrides(model, tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    run_script(
        TRAIN_SCRIPT,
        [
            *model_overrides(model),
            *TRAINER_OVERRIDES,
            f"trainer.checkpoint.checkpoint_dir={ckpt_dir}",
        ],
        tmp_path,
    )

    output = run_script(
        EVAL_SCRIPT,
        [
            f"bundle_path={ckpt_dir / 'latest' / 'best_model.bundle'}",
            "num_episodes=1",
            "execute_steps=4",
            f"video_folder={tmp_path / 'videos'}",
        ],
        tmp_path,
    )
    assert "Success rate: 0/1" in output


def test_train_script_resumes_from_checkpoint(tmp_path):
    model = "dit_cross_attn"
    ckpt_dir = tmp_path / "ckpt"
    common = [
        *model_overrides(model),
        *TRAINER_OVERRIDES,
        f"trainer.checkpoint.checkpoint_dir={ckpt_dir}",
    ]
    run_script(TRAIN_SCRIPT, common, tmp_path)
    latest = ckpt_dir / "latest" / "latest.bundle"
    assert load_checkpoint(latest).training_state["global_step"] == 2

    output = run_script(
        TRAIN_SCRIPT,
        [
            *common,
            f"trainer.train.training_steps={TRAIN_STEPS + 2}",
            f"trainer.checkpoint.resume_from={latest}",
        ],
        tmp_path,
    )

    assert "Resumed at step 2" in output
    assert "Training Complete!" in output
    # Resumed at 2, so this run only covers steps 2..5 and its last val
    # checkpoint lands at step 4 -- not possible without the resume.
    assert load_checkpoint(latest).training_state["global_step"] == 4


def test_overfit_script_runs_end_to_end(tmp_path):
    overfit_dir = tmp_path / "overfit"
    output = run_script(
        OVERFIT_SCRIPT,
        [
            *model_overrides("dit_cross_attn"),
            "overfit.steps=3",
            f"overfit.checkpoint_dir={overfit_dir}",
        ],
        tmp_path,
    )

    assert "Action reconstruction MSE" in output
    bundle = load_checkpoint(overfit_dir / "overfit_final.bundle")
    assert bundle.training_state["global_step"] == 3
    assert_finite_state_dict(bundle.training_state["model_state_dict"])
