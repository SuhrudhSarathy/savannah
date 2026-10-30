import json
import zipfile

import pytest
import torch
from dummies import SyntheticTask
from omegaconf import OmegaConf

from savannah.bundle import (
    BUNDLE_FORMAT_VERSION,
    MANIFEST_FILE,
    WEIGHTS_FILE,
    BundleCorruptError,
    load_bundle,
    load_policy,
    read_bundle,
    save_bundle,
)
from savannah.data.dataset import DataSetConfig
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
    from savannah.factory import build_policy

    return build_policy(make_cfg()).eval()


@pytest.fixture
def bundle_path(tmp_path, source_policy):
    return save_bundle(
        tmp_path / "policy.bundle", make_cfg(), make_task(), source_policy.state_dict()
    )


def test_load_policy_reproduces_actions_without_any_config(bundle_path, source_policy):
    loaded = load_policy(bundle_path)
    assert not loaded.training

    obs = make_obs()
    with torch.no_grad():
        torch.manual_seed(0)
        expected = source_policy.compute_action(obs).actions
        torch.manual_seed(0)
        actual = loaded.compute_action(obs).actions

    assert torch.equal(actual, expected)


def test_loaded_policy_reproduces_eval_numbers(bundle_path, source_policy):
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
    actual, actual_chunks = rollout(load_policy(bundle_path))

    assert actual.success_rate == expected.success_rate
    assert actual.mean_reward == expected.mean_reward
    assert torch.equal(actual_chunks, expected_chunks)


def test_manifest_records_specs_and_provenance(bundle_path):
    manifest = read_bundle(bundle_path).manifest

    assert manifest["format_version"] == BUNDLE_FORMAT_VERSION
    assert manifest["savannah_version"]
    assert manifest["git_hash"]
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


def test_config_is_stored_resolved(bundle_path):
    bundle = load_bundle(bundle_path)
    assert not OmegaConf.is_interpolation(bundle.config.model, "action_horizon")
    assert bundle.config.model.action_horizon == ACTION_HORIZON


def test_failed_save_leaves_existing_bundle_intact(
    bundle_path, source_policy, monkeypatch
):
    before = bundle_path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("savannah.bundle.os.replace", fail)
    with pytest.raises(OSError, match="disk full"):
        save_bundle(bundle_path, make_cfg(), make_task(), source_policy.state_dict())

    assert bundle_path.read_bytes() == before
    assert list(bundle_path.parent.iterdir()) == [bundle_path]
    load_policy(bundle_path)


def test_flipped_byte_is_rejected(bundle_path):
    data = bytearray(bundle_path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    bundle_path.write_bytes(data)

    with pytest.raises(BundleCorruptError):
        load_policy(bundle_path)


def test_truncated_file_is_rejected(bundle_path):
    data = bundle_path.read_bytes()
    bundle_path.write_bytes(data[: len(data) // 2])

    with pytest.raises(BundleCorruptError):
        load_policy(bundle_path)


def test_tampered_member_with_valid_zip_is_rejected(bundle_path):
    with zipfile.ZipFile(bundle_path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members[WEIGHTS_FILE] = b"not the weights"
    with zipfile.ZipFile(bundle_path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)

    with pytest.raises(BundleCorruptError, match="bad checksum"):
        load_policy(bundle_path)


def test_use_ema_false_needs_training_state(bundle_path):
    with pytest.raises(ValueError, match="only EMA"):
        load_bundle(bundle_path, use_ema=False)


def test_rejects_newer_format_version(bundle_path):
    with zipfile.ZipFile(bundle_path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(members[MANIFEST_FILE])
    manifest["format_version"] = BUNDLE_FORMAT_VERSION + 1
    members[MANIFEST_FILE] = json.dumps(manifest).encode()
    with zipfile.ZipFile(bundle_path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)

    with pytest.raises(ValueError, match="newer"):
        load_policy(bundle_path)


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="No bundle"):
        load_policy(tmp_path / "nope.bundle")
