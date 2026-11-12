import json
from pathlib import Path

import pytest
import torch
from dummies import SyntheticTask
from omegaconf import OmegaConf

import savannah.pretrained as pretrained_mod
from savannah.data.dataset import DataSetConfig
from savannah.factory import build_policy
from savannah.factory import (
    build_policy_from_pretrained as factory_build_policy_from_pretrained,
)
from savannah.objectives import FlowMatchingObjective
from savannah.pretrained import (
    CONFIG_FILE,
    FORMAT_VERSION,
    MANIFEST_FILE,
    TRAINING_STATE_FILE,
    WEIGHTS_FILE,
    build_policy_from_pretrained,
    load_pretrained,
    load_pretrained_weights,
    read_checkpoint,
    resolve_pretrained_dir,
    save_pretrained,
)
from savannah.utils.evaluation import run_rollout
from savannah.utils.observation import ObservationKey

STATE_DIM, ACTION_DIM, ACTION_HORIZON, OBS_HORIZON = 7, 6, 8, 2
CAMERAS = ["cam0"]


def make_cfg():
    return OmegaConf.create(
        {
            "horizon": ACTION_HORIZON,
            "model": {
                "_target_": "savannah.models.dit_cross_attn.DITCrossAttnPolicy",
                "embed_dim": 32,
                "time_embed_dim": 8,
                "state_embed_dim": 8,
                "decoder_num_blocks": 1,
                "decoder_num_attn_heads": 2,
                "decoder_feedforward_dim": 64,
                "state_dim": STATE_DIM,
                "num_obs": OBS_HORIZON,
                "action_dim": ACTION_DIM,
                "action_horizon": "${horizon}",  # must be stored resolved
                "num_cameras": len(CAMERAS),
                "use_rope": True,
                "language_encoder": None,
                "vision_encoder": {
                    "_target_": "savannah.models.encoders.VisionEncoder",
                    "backbone": {
                        "_target_": "savannah.models.backbones.DummyVisionBackbone",
                        "out_channels": 16,
                    },
                    "embed_dim": 32,
                    "num_cameras": len(CAMERAS),
                    "num_obs": OBS_HORIZON,
                },
                "objective": {"_target_": "savannah.objectives.OverfitObjective"},
            },
            "task": {
                "config": {
                    "cameras": CAMERAS,
                    "image_size": 32,
                    "obs_horizon": OBS_HORIZON,
                    "fps": 10,
                }
            },
        }
    )


def make_task():
    config = DataSetConfig(
        repo_id="synthetic",
        fps=10,
        cameras=CAMERAS,
        obs_horizon=OBS_HORIZON,
        action_horizon=ACTION_HORIZON,
        image_size=32,
        num_workers=0,
    )
    return SyntheticTask(
        config, torch.device("cpu"), state_dim=STATE_DIM, action_dim=ACTION_DIM
    )


def make_obs():
    return {
        ObservationKey.images: [torch.rand(2, OBS_HORIZON, 3, 32, 32)],
        ObservationKey.state: torch.randn(2, OBS_HORIZON, STATE_DIM),
    }


@pytest.fixture
def source_policy():
    return build_policy(make_cfg()).eval()


@pytest.fixture
def run_dir(tmp_path, source_policy):
    """A local run dir holding a `best_model/` checkpoint (EMA weights only)
    and a `latest/` one (with training state)."""
    run = tmp_path / "run"
    save_pretrained(
        run / "best_model",
        make_cfg(),
        make_task(),
        source_policy.state_dict(),
        step=10,
        metrics={"val_reconstruction_loss": 0.25},
    )
    save_pretrained(
        run / "latest",
        make_cfg(),
        make_task(),
        source_policy.state_dict(),
        training_state={"model_state_dict": source_policy.state_dict()},
        step=12,
    )
    return run


@pytest.fixture
def checkpoint_dir(run_dir):
    return run_dir / "best_model"


def actions(policy, seed=0):
    with torch.no_grad():
        torch.manual_seed(seed)
        return policy.compute_action(make_obs()).actions


def test_checkpoint_layout(run_dir):
    assert sorted(p.name for p in (run_dir / "best_model").iterdir()) == sorted(
        [CONFIG_FILE, WEIGHTS_FILE, MANIFEST_FILE]
    )
    assert (run_dir / "latest" / TRAINING_STATE_FILE).is_file()
    # No temp dirs left behind by the atomic save.
    assert sorted(p.name for p in run_dir.iterdir()) == ["best_model", "latest"]


def test_build_policy_from_pretrained_reproduces_actions(checkpoint_dir, source_policy):
    loaded = build_policy_from_pretrained(checkpoint_dir)
    assert not loaded.training
    assert torch.equal(actions(loaded), actions(source_policy))


def test_factory_reexport(checkpoint_dir, source_policy):
    loaded = factory_build_policy_from_pretrained(checkpoint_dir)
    assert torch.equal(actions(loaded), actions(source_policy))


def test_run_dir_resolves_subfolder(run_dir):
    assert resolve_pretrained_dir(run_dir) == run_dir / "best_model"
    assert resolve_pretrained_dir(run_dir, subfolder="latest") == run_dir / "latest"
    # A checkpoint dir is used as is, whatever the subfolder.
    assert (
        resolve_pretrained_dir(run_dir / "latest", subfolder="best_model")
        == run_dir / "latest"
    )


def test_missing_subfolder_lists_available(run_dir):
    with pytest.raises(FileNotFoundError, match="best_model"):
        resolve_pretrained_dir(run_dir, subfolder="best_model_success")


def test_missing_local_path_that_isnt_a_repo_id(tmp_path):
    with pytest.raises(FileNotFoundError, match="neither"):
        load_pretrained(tmp_path / "nope" / "deeper")


def test_hub_repo_id_downloads_only_the_subfolder(run_dir, source_policy, monkeypatch):
    calls = []

    def fake_snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return str(run_dir)  # the snapshot root holds the subfolders

    monkeypatch.setattr(pretrained_mod, "snapshot_download", fake_snapshot_download)
    loaded = load_pretrained(
        "someone/some_run", subfolder="best_model", revision="best_model-step-10"
    )

    assert loaded.path == run_dir / "best_model"
    assert torch.equal(actions(loaded.policy), actions(source_policy))
    [(repo_id, kwargs)] = calls
    assert repo_id == "someone/some_run"
    assert kwargs["revision"] == "best_model-step-10"
    assert kwargs["allow_patterns"] == ["best_model/*"]


def test_hub_repo_without_the_subfolder(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pretrained_mod, "snapshot_download", lambda repo_id, **kw: str(tmp_path)
    )
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        load_pretrained("someone/some_run", subfolder="best_model_success")


def test_loaded_policy_reproduces_eval_numbers(checkpoint_dir, source_policy):
    def rollout(policy):
        task = make_task()
        env = task.get_env()
        env.reset(seed=0)  # seeds env.np_random; later resets continue the stream
        chunks = []
        torch.manual_seed(0)
        result = run_rollout(
            policy,
            task,
            env,
            num_episodes=2,
            execute_steps=4,
            capture_frames=False,
            on_chunk=chunks.append,
        )
        return result, torch.stack(chunks)

    expected, expected_chunks = rollout(source_policy)
    actual, actual_chunks = rollout(build_policy_from_pretrained(checkpoint_dir))

    assert actual.success_rate == expected.success_rate
    assert actual.mean_reward == expected.mean_reward
    assert torch.equal(actual_chunks, expected_chunks)


def test_manifest_records_specs_provenance_and_metrics(checkpoint_dir):
    manifest = read_checkpoint(checkpoint_dir).manifest

    assert manifest["format_version"] == FORMAT_VERSION
    assert manifest["savannah_version"]
    assert manifest["git_hash"]
    assert manifest["step"] == 10
    assert manifest["metrics"] == {"val_reconstruction_loss": 0.25}
    assert manifest["obs_spec"] == {
        "cameras": CAMERAS,
        "image_size": 32,
        "obs_horizon": OBS_HORIZON,
        "state_keys": ["observation.state"],
        "state_dim": STATE_DIM,
    }
    assert manifest["action_spec"]["dim"] == ACTION_DIM
    assert manifest["action_spec"]["horizon"] == ACTION_HORIZON
    assert manifest["action_spec"]["rate_hz"] == 10


def test_config_is_stored_resolved(checkpoint_dir):
    loaded = load_pretrained(checkpoint_dir)
    assert not OmegaConf.is_interpolation(loaded.config.model, "action_horizon")
    assert loaded.config.model.action_horizon == ACTION_HORIZON


def test_model_kwargs_override_the_stored_config(checkpoint_dir):
    loaded = load_pretrained(
        checkpoint_dir,
        objective={
            "_target_": "savannah.objectives.FlowMatchingObjective",
            "inference_steps": 3,
        },
    )
    assert isinstance(loaded.policy.objective, FlowMatchingObjective)
    assert loaded.policy.objective.inference_steps == 3
    assert loaded.config.model.objective.inference_steps == 3
    # Other model fields are untouched.
    assert loaded.config.model.embed_dim == 32


def test_shape_changing_override_needs_ignore_mismatched_sizes(
    checkpoint_dir, source_policy
):
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_pretrained(checkpoint_dir, action_dim=ACTION_DIM - 2)

    loaded = load_pretrained(
        checkpoint_dir, action_dim=ACTION_DIM - 2, ignore_mismatched_sizes=True
    ).policy

    source = source_policy.state_dict()
    target = loaded.state_dict()
    skipped = [k for k in source if source[k].shape != target[k].shape]
    assert skipped, "changing action_dim should change some parameter shapes"
    for k, v in target.items():
        if k not in skipped:
            assert torch.equal(v, source[k]), k


def test_load_pretrained_weights_into_existing_policy(checkpoint_dir, source_policy):
    cfg = make_cfg()
    cfg.model.action_dim = ACTION_DIM - 2
    policy = build_policy(cfg)
    load_pretrained_weights(policy, checkpoint_dir, ignore_mismatched_sizes=True)

    source = source_policy.state_dict()
    loaded = 0
    for k, v in policy.state_dict().items():
        if v.shape == source[k].shape:
            assert torch.equal(v, source[k]), k
            loaded += 1
    assert loaded > 0


def test_use_ema_false_needs_training_state(run_dir):
    with pytest.raises(ValueError, match="only EMA"):
        load_pretrained(run_dir, subfolder="best_model", use_ema=False)
    load_pretrained(run_dir, subfolder="latest", use_ema=False)


def test_failed_save_leaves_existing_checkpoint_intact(
    checkpoint_dir, source_policy, monkeypatch
):
    before = {p.name: p.read_bytes() for p in checkpoint_dir.iterdir()}
    real_replace = pretrained_mod.os.replace

    def fail_on_swap_in(src, dst):
        # Let the old checkpoint be moved aside, then fail moving the new one
        # in: the old one must be restored.
        if Path(src).name.endswith(".tmp"):
            raise OSError("disk full")
        real_replace(src, dst)

    monkeypatch.setattr(pretrained_mod.os, "replace", fail_on_swap_in)
    with pytest.raises(OSError, match="disk full"):
        save_pretrained(
            checkpoint_dir, make_cfg(), make_task(), source_policy.state_dict()
        )
    monkeypatch.undo()

    assert {p.name: p.read_bytes() for p in checkpoint_dir.iterdir()} == before
    assert sorted(p.name for p in checkpoint_dir.parent.iterdir()) == [
        "best_model",
        "latest",
    ]
    build_policy_from_pretrained(checkpoint_dir)


def test_failed_write_leaves_existing_checkpoint_intact(
    checkpoint_dir, source_policy, monkeypatch
):
    before = {p.name: p.read_bytes() for p in checkpoint_dir.iterdir()}

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pretrained_mod, "save_file", fail)
    with pytest.raises(OSError, match="disk full"):
        save_pretrained(
            checkpoint_dir, make_cfg(), make_task(), source_policy.state_dict()
        )

    assert {p.name: p.read_bytes() for p in checkpoint_dir.iterdir()} == before
    assert sorted(p.name for p in checkpoint_dir.parent.iterdir()) == [
        "best_model",
        "latest",
    ]


def test_truncated_weights_are_rejected(checkpoint_dir):
    weights = checkpoint_dir / WEIGHTS_FILE
    weights.write_bytes(weights.read_bytes()[: weights.stat().st_size // 2])
    with pytest.raises(Exception):  # noqa: B017 -- safetensors' own error type
        build_policy_from_pretrained(checkpoint_dir)


def test_rejects_newer_format_version(checkpoint_dir):
    manifest_path = checkpoint_dir / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = FORMAT_VERSION + 1
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="newer"):
        build_policy_from_pretrained(checkpoint_dir)
