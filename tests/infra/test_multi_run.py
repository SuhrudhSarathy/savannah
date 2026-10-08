import importlib.util
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# infra/ isn't a package, so load the script as a module by path.
_spec = importlib.util.spec_from_file_location(
    "multi_run", REPO_ROOT / "infra" / "multi_run.py"
)
multi_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(multi_run)


def _write(tmp_path, cfg: dict) -> Path:
    path = tmp_path / "experiments.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_repo_experiments_yaml_loads():
    experiments, vram, _ = multi_run.load_experiments(
        REPO_ROOT / "configs" / "experiments.yaml"
    )
    assert experiments
    assert vram > 0
    assert len({e.name for e in experiments}) == len(experiments), "duplicate names"


def test_name_drives_run_name_and_default_checkpoint_dir():
    exp = multi_run.Experiment(name="exp_a", overrides=["model=unet"])
    cmd, _ = multi_run.build_command(exp)
    # Hydra `name` feeds wandb.run_name, the artifact name, and
    # checkpoints/${name}, so per-experiment dirs need no explicit override.
    assert "name=exp_a" in cmd
    assert not any(a.startswith("trainer.checkpoint.checkpoint_dir=") for a in cmd)
    assert cmd[-1] == "model=unet"


def test_checkpoint_root_gives_per_experiment_dir(tmp_path):
    path = _write(
        tmp_path,
        {
            "checkpoint_root": "/data/ckpts",
            "experiments": [{"name": "exp_a"}, {"name": "exp_b"}],
        },
    )
    experiments, vram, root = multi_run.load_experiments(path)
    assert vram == 8  # default
    assert root == "/data/ckpts"

    dirs = [
        next(
            a
            for a in multi_run.build_command(e, checkpoint_root=root)[0]
            if a.startswith("trainer.checkpoint.checkpoint_dir=")
        )
        for e in experiments
    ]
    assert dirs == [
        "trainer.checkpoint.checkpoint_dir=/data/ckpts/exp_a",
        "trainer.checkpoint.checkpoint_dir=/data/ckpts/exp_b",
    ]


def test_experiment_overrides_come_last_so_they_win():
    exp = multi_run.Experiment(
        name="exp_a", overrides=["trainer.checkpoint.checkpoint_dir=/custom"]
    )
    cmd, _ = multi_run.build_command(exp, checkpoint_root="/data/ckpts")
    assert cmd[-1] == "trainer.checkpoint.checkpoint_dir=/custom"


@pytest.mark.parametrize(
    ("kwargs", "env_key", "value"),
    [
        ({"cuda_devices": "1"}, "CUDA_VISIBLE_DEVICES", "1"),
        ({"memory_fraction": 0.45}, "CUDA_MEMORY_FRACTION", "0.45"),
    ],
)
def test_gpu_env_vars(kwargs, env_key, value):
    _, env = multi_run.build_command(multi_run.Experiment(name="e"), **kwargs)
    assert env[env_key] == value
