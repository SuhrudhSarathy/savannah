import threading
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch
from dummies import SyntheticTask
from omegaconf import OmegaConf

import savannah.hub as hub_mod
from savannah.data.dataset import DataSetConfig
from savannah.models.backbones import DummyVisionBackbone
from savannah.models.dit_cross_attn import DITCrossAttnPolicy
from savannah.models.encoders import VisionEncoder
from savannah.objectives import OverfitObjective
from savannah.pretrained import read_checkpoint, save_pretrained
from savannah.trainer.trainer import (
    CheckpointParams,
    EvalParams,
    PolicyTrainer,
    TrainConfig,
    TrainParams,
)
from savannah.utils.logger import ExperimentLogger
from savannah.utils.observation import ObservationKey

STATE_DIM, ACTION_DIM, ACTION_HORIZON, EMBED_DIM = 7, 6, 8, 32
NUM_CAMERAS = 1


def build_tiny_policy() -> DITCrossAttnPolicy:
    backbone = DummyVisionBackbone(out_channels=16)
    vision_encoder = VisionEncoder(
        backbone, embed_dim=EMBED_DIM, num_cameras=NUM_CAMERAS, num_obs=1
    )
    return DITCrossAttnPolicy(
        embed_dim=EMBED_DIM,
        time_embed_dim=8,
        state_embed_dim=8,
        decoder_num_blocks=1,
        decoder_num_attn_heads=2,
        decoder_feedforward_dim=64,
        state_dim=STATE_DIM,
        num_obs=1,
        action_dim=ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        num_cameras=NUM_CAMERAS,
        use_rope=True,
        vision_encoder=vision_encoder,
        language_encoder=None,
        objective=OverfitObjective(),
    )


def build_task() -> SyntheticTask:
    return SyntheticTask(
        DataSetConfig(repo_id="synthetic", fps=10, cameras=["cam0"], image_size=64),
        torch.device("cpu"),
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
    )


RUN_CFG = OmegaConf.create(
    {
        "model": {
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "action_horizon": ACTION_HORIZON,
        },
        "task": {
            "config": {
                "cameras": ["cam0"],
                "image_size": 64,
                "obs_horizon": 1,
                "fps": 10,
            }
        },
    }
)


def build_trainer(checkpoint_dir, **checkpoint_kwargs) -> PolicyTrainer:
    config = TrainConfig(
        train=TrainParams(
            lr=1e-3,
            training_steps=10,
            warmup_steps=1,
            gradient_accumulation_steps=1,
        ),
        eval=EvalParams(),
        checkpoint=CheckpointParams(
            checkpoint_dir=str(checkpoint_dir), **checkpoint_kwargs
        ),
    )
    exp_logger = ExperimentLogger(project_name="test", config={}, enable_logging=False)
    return PolicyTrainer(
        policy=build_tiny_policy(),
        task=build_task(),
        logger=exp_logger,
        config=config,
        device=torch.device("cpu"),
        run_cfg=RUN_CFG,
    )


def take_optimizer_step(trainer: PolicyTrainer):
    trainer.policy.train()
    batch = {
        ObservationKey.images: [
            torch.randn(2, 1, 3, 64, 64) for _ in range(NUM_CAMERAS)
        ],
        ObservationKey.state: torch.randn(2, 1, STATE_DIM),
        ObservationKey.gt_actions: torch.randn(2, ACTION_HORIZON, ACTION_DIM),
    }
    loss = trainer.policy.compute_loss(batch)
    loss.backward()
    trainer.optimizer.step()
    trainer.scheduler.step()
    trainer.ema.update()
    trainer.optimizer.zero_grad()
    trainer.global_step += 1


def test_load_checkpoint_restores_training_state(tmp_path):
    trainer = build_trainer(tmp_path)
    for _ in range(3):
        take_optimizer_step(trainer)
    trainer.best_val_loss = 0.5
    trainer.best_success_rate = 0.75
    trainer.save_checkpoint("latest", {"val_reconstruction_loss": 0.5})

    resumed = build_trainer(tmp_path)
    resumed.load_checkpoint(str(tmp_path / "latest"))

    assert resumed.global_step == trainer.global_step
    assert resumed.best_val_loss == pytest.approx(trainer.best_val_loss)
    assert resumed.best_success_rate == pytest.approx(trainer.best_success_rate)

    for p1, p2 in zip(
        trainer.policy.parameters(), resumed.policy.parameters(), strict=True
    ):
        assert torch.allclose(p1, p2)

    for p1, p2 in zip(
        trainer.ema.shadow.parameters(),
        resumed.ema.shadow.parameters(),
        strict=True,
    ):
        assert torch.allclose(p1, p2)

    orig_opt_state = trainer.optimizer.state_dict()
    resumed_opt_state = resumed.optimizer.state_dict()
    assert orig_opt_state["param_groups"] == resumed_opt_state["param_groups"]

    assert (
        trainer.scheduler.state_dict()["last_epoch"]
        == resumed.scheduler.state_dict()["last_epoch"]
    )


def test_save_checkpoint_overwrites_one_folder_per_kind(tmp_path):
    trainer = build_trainer(tmp_path)

    take_optimizer_step(trainer)
    trainer.save_checkpoint("latest", {"val_reconstruction_loss": 0.9})
    take_optimizer_step(trainer)
    step_b = trainer.global_step
    trainer.save_checkpoint("latest", {"val_reconstruction_loss": 0.8})
    trainer.save_checkpoint("best_model", {"val_reconstruction_loss": 0.8})

    # One folder per checkpoint kind; per-step history lives on the Hub.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["best_model", "latest"]

    latest = read_checkpoint(tmp_path / "latest")
    assert latest.training_state["global_step"] == step_b
    assert latest.manifest["step"] == step_b
    assert latest.manifest["metrics"] == {"val_reconstruction_loss": 0.8}


def test_resume_from_run_dir_uses_latest(tmp_path):
    trainer = build_trainer(tmp_path)
    take_optimizer_step(trainer)
    trainer.save_checkpoint("latest")

    resumed = build_trainer(tmp_path)
    resumed.load_checkpoint(str(tmp_path))
    assert resumed.global_step == trainer.global_step


def test_resume_needs_a_local_checkpoint(tmp_path):
    trainer = build_trainer(tmp_path)
    with pytest.raises(FileNotFoundError, match="never pushed"):
        trainer.load_checkpoint("someone/some_run")


def test_load_checkpoint_falls_back_for_missing_best_metric_keys(tmp_path):
    trainer = build_trainer(tmp_path)
    take_optimizer_step(trainer)

    training_state = {
        "model_state_dict": trainer.policy.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "scheduler_state_dict": trainer.scheduler.state_dict(),
        "global_step": trainer.global_step,
        "success_rate": 0.0,
    }
    path = tmp_path / "legacy"
    save_pretrained(
        path,
        RUN_CFG,
        trainer.task,
        trainer.ema.shadow.state_dict(),
        training_state=training_state,
    )

    resumed = build_trainer(tmp_path)
    default_best_val_loss = resumed.best_val_loss
    default_best_success_rate = resumed.best_success_rate

    resumed.load_checkpoint(str(path))

    assert resumed.global_step == trainer.global_step
    assert resumed.best_val_loss == default_best_val_loss
    assert resumed.best_success_rate == default_best_success_rate


# --------------------------------------------------------------------------
# Hugging Face Hub uploads (fake HfApi, no network)
# --------------------------------------------------------------------------


class FakeHfApi:
    """Records Hub calls. Uploads run on a real background thread, like
    HfApi.run_as_future, so ordering bugs would show up."""

    def __init__(self, *args, **kwargs):
        self.created_repos = []
        self.commits = []
        self.tags = []
        self.release_uploads = threading.Event()
        self.release_uploads.set()
        FakeHfApi.instance = self

    def create_repo(self, repo_id, **kwargs):
        self.created_repos.append((repo_id, kwargs))
        return f"https://huggingface.co/{repo_id}"

    def run_as_future(self, fn, *args, **kwargs):
        future = Future()

        def run():
            self.release_uploads.wait()
            try:
                future.set_result(fn(*args, **kwargs))
            except Exception as e:
                future.set_exception(e)

        threading.Thread(target=run, daemon=True).start()
        return future

    def create_commit(self, repo_id, *, operations, commit_message, **kwargs):
        files = {}
        for op in operations:
            obj = op.path_or_fileobj
            files[op.path_in_repo] = (
                obj if isinstance(obj, bytes) else open(obj, "rb").read()
            )
        self.commits.append(
            SimpleNamespace(repo_id=repo_id, message=commit_message, files=files)
        )
        return SimpleNamespace(oid=f"oid{len(self.commits)}")

    def create_tag(self, repo_id, *, tag, revision, **kwargs):
        self.tags.append((tag, revision))


@pytest.fixture
def fake_hub(monkeypatch):
    monkeypatch.setattr(hub_mod, "HfApi", FakeHfApi)
    return FakeHfApi


def build_hub_trainer(tmp_path):
    return build_trainer(
        tmp_path, push_to_hub=True, hub_repo_id="someone/run", hub_private=True
    )


def test_push_to_hub_requires_repo_id(tmp_path, fake_hub):
    with pytest.raises(ValueError, match="hub_repo_id"):
        build_trainer(tmp_path, push_to_hub=True)


def test_push_to_hub_creates_private_repo_up_front(tmp_path, fake_hub):
    build_hub_trainer(tmp_path)
    [(repo_id, kwargs)] = fake_hub.instance.created_repos
    assert repo_id == "someone/run"
    assert kwargs["private"] is True
    assert kwargs["exist_ok"] is True


def test_save_checkpoint_pushes_subfolder_without_training_state(tmp_path, fake_hub):
    trainer = build_hub_trainer(tmp_path)
    take_optimizer_step(trainer)
    trainer.save_checkpoint("best_model", {"val_reconstruction_loss": 0.5})
    trainer.wait_for_uploads()

    api = fake_hub.instance
    [commit] = api.commits
    assert sorted(commit.files) == [
        "README.md",
        "best_model/config.yaml",
        "best_model/manifest.json",
        "best_model/model.safetensors",
    ]
    assert commit.message == f"best_model @ step {trainer.global_step}"
    assert api.tags == [(f"best_model-step-{trainer.global_step}", "oid1")]
    card = commit.files["README.md"].decode()
    assert "someone/run" in card and "best_model" in card
    # training_state.pt is still written locally for resuming.
    assert (tmp_path / "best_model" / "training_state.pt").is_file()


def test_save_waits_for_previous_upload_of_same_folder(tmp_path, fake_hub):
    trainer = build_hub_trainer(tmp_path)
    api = fake_hub.instance
    api.release_uploads.clear()  # hold uploads in flight

    take_optimizer_step(trainer)
    trainer.save_checkpoint("latest")
    first_step = trainer.global_step
    take_optimizer_step(trainer)

    saver = threading.Thread(target=trainer.save_checkpoint, args=("latest",))
    saver.start()
    saver.join(timeout=0.5)
    # The second save must block until the first upload has read the folder.
    assert saver.is_alive()
    assert read_checkpoint(tmp_path / "latest").manifest["step"] == first_step

    api.release_uploads.set()
    saver.join(timeout=10)
    assert not saver.is_alive()
    trainer.wait_for_uploads()

    assert [c.message for c in api.commits] == [
        f"latest @ step {first_step}",
        f"latest @ step {trainer.global_step}",
    ]


def test_failed_upload_is_logged_not_raised(tmp_path, fake_hub, monkeypatch):
    trainer = build_hub_trainer(tmp_path)

    def boom(*args, **kwargs):
        raise ConnectionError("hub down")

    monkeypatch.setattr(fake_hub.instance, "create_commit", boom)
    take_optimizer_step(trainer)
    trainer.save_checkpoint("latest")
    assert trainer.uploader.wait_all() is False
    # The local checkpoint survives for a manual re-push.
    assert read_checkpoint(tmp_path / "latest").training_state is not None


def test_checkpointing_never_touches_wandb_artifacts(tmp_path, monkeypatch):
    import wandb

    def forbidden(*args, **kwargs):
        raise AssertionError("checkpoints must not go to W&B")

    monkeypatch.setattr(wandb, "Artifact", forbidden)
    monkeypatch.setattr(wandb, "log_artifact", forbidden)
    trainer = build_trainer(tmp_path)
    take_optimizer_step(trainer)
    trainer.save_checkpoint("latest")
    trainer.wait_for_uploads()

    assert not hasattr(ExperimentLogger, "log_model_artifact")
