"""Pushes training checkpoints to a Hugging Face Hub model repo.

One repo per run; each checkpoint kind (`latest`, `best_model`,
`best_model_success`) is a subfolder, and every push is one commit (the
checkpoint files + a regenerated root README.md model card) tagged
`{name}-step-{step}`. The Hub's git history is the per-step archive.

Uploads run in the background (huggingface_hub's single-worker executor, so
commits land in order) and never block the training loop. Before a local
checkpoint folder is overwritten by the next save, `wait_for(name)` blocks
until its previous upload has finished reading it.
"""

from concurrent.futures import Future
from pathlib import Path
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi

from savannah.pretrained import LOCAL_ONLY_FILES
from savannah.utils.log import logger

CARD_FILE = "README.md"


def checkpoint_tag(name: str, step: int) -> str:
    return f"{name}-step-{step}"


def _format_metrics(metrics: dict[str, Any]) -> str:
    if not metrics:
        return "–"
    return ", ".join(
        f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
        for k, v in metrics.items()
    )


def render_model_card(
    repo_id: str,
    checkpoints: dict[str, dict[str, Any]],
    *,
    run_name: str | None = None,
    model_target: str | None = None,
    dataset: str | None = None,
    private: bool = True,
) -> str:
    """The root README.md for a run's repo. `checkpoints` maps each pushed
    checkpoint name to `{"step": int, "metrics": dict}`."""
    tags = ["robotics", "savannah"]
    if model_target:
        tags.append(model_target.rsplit(".", 1)[-1])
    front_matter = ["---", "library_name: savannah", "tags:"]
    front_matter += [f"  - {t}" for t in tags]
    if dataset:
        front_matter += ["datasets:", f"  - {dataset}"]
    front_matter.append("---")

    rows = [
        f"| `{name}` | {info['step']} | `{checkpoint_tag(name, info['step'])}` "
        f"| {_format_metrics(info.get('metrics', {}))} |"
        for name, info in sorted(checkpoints.items())
    ]
    first = sorted(checkpoints)[0] if checkpoints else "best_model"

    about = ["A [savannah](https://github.com/SuhrudhSarathy/savannah) policy"]
    if model_target:
        about.append(f" (`{model_target}`)")
    if dataset:
        about.append(
            f" trained on [`{dataset}`](https://huggingface.co/datasets/{dataset})"
        )
    if run_name:
        about.append(f", run `{run_name}`")
    about.append(".")

    return "\n".join(
        [
            *front_matter,
            "",
            f"# {repo_id}",
            "",
            "".join(about),
            "",
            "Each checkpoint lives in its own subfolder; every push is tagged "
            "`<checkpoint>-step-<N>`, so older checkpoints stay loadable by "
            "revision.",
            "",
            "| Checkpoint | Step | Tag | Metrics |",
            "|---|---|---|---|",
            *rows,
            "",
            "## Usage",
            "",
            "```python",
            "from savannah.pretrained import build_policy_from_pretrained",
            "",
            "policy = build_policy_from_pretrained(",
            f'    "{repo_id}", subfolder="{first}", device="cuda"',
            ")",
            "```",
            "",
            "```bash",
            f"uv run python scripts/eval/eval.py pretrained={repo_id} subfolder={first}",
            "```",
            *(
                ["", "This repo is private: log in with `hf auth login` first."]
                if private
                else []
            ),
            "",
        ]
    )


class HubCheckpointUploader:
    def __init__(
        self,
        repo_id: str,
        *,
        private: bool = True,
        token: str | None = None,
        run_name: str | None = None,
        model_target: str | None = None,
        dataset: str | None = None,
        api: HfApi | None = None,
    ):
        self.repo_id = repo_id
        self.private = private
        self.run_name = run_name
        self.model_target = model_target
        self.dataset = dataset
        self.api = api or HfApi(token=token, library_name="savannah")

        # Fail at step 0 (bad token, not logged in, wrong namespace) rather
        # than at the first checkpoint.
        url = self.api.create_repo(
            repo_id, repo_type="model", private=private, exist_ok=True
        )
        logger.info("Checkpoints will be pushed to {}", url)

        self._futures: dict[str, Future] = {}
        self._checkpoints: dict[str, dict[str, Any]] = {}

    def push(
        self,
        local_dir: str | Path,
        name: str,
        step: int,
        metrics: dict[str, Any] | None = None,
    ) -> Future:
        """Uploads `local_dir` to `<repo>/<name>/` in the background and tags
        the commit `{name}-step-{step}`."""
        self.wait_for(name)
        self._checkpoints[name] = {"step": step, "metrics": dict(metrics or {})}
        card = render_model_card(
            self.repo_id,
            self._checkpoints,
            run_name=self.run_name,
            model_target=self.model_target,
            dataset=self.dataset,
            private=self.private,
        )
        future = self.api.run_as_future(self._commit, Path(local_dir), name, step, card)
        self._futures[name] = future
        return future

    def _commit(self, local_dir: Path, name: str, step: int, card: str):
        operations = [
            CommitOperationAdd(path_in_repo=f"{name}/{f.name}", path_or_fileobj=str(f))
            for f in sorted(local_dir.iterdir())
            if f.is_file() and f.name not in LOCAL_ONLY_FILES
        ]
        operations.append(
            CommitOperationAdd(path_in_repo=CARD_FILE, path_or_fileobj=card.encode())
        )
        info = self.api.create_commit(
            self.repo_id,
            operations=operations,
            commit_message=f"{name} @ step {step}",
            repo_type="model",
        )
        tag = checkpoint_tag(name, step)
        self.api.create_tag(
            self.repo_id, tag=tag, revision=info.oid, repo_type="model", exist_ok=True
        )
        logger.success("Pushed {} to {} (tag {})", name, self.repo_id, tag)
        return info

    def wait_for(self, name: str) -> bool:
        """Blocks until the pending upload of `name` (if any) finishes. Returns
        False, after logging, if it failed -- the local copy is kept."""
        future = self._futures.pop(name, None)
        if future is None:
            return True
        try:
            future.result()
            return True
        except Exception as e:
            logger.error(
                "Pushing {} to {} failed: {}. The local checkpoint is kept; "
                "re-push it with `hf upload {} <checkpoint_dir>/{} {} "
                "--exclude training_state.pt`",
                name,
                self.repo_id,
                e,
                self.repo_id,
                name,
                name,
            )
            return False

    def wait_all(self) -> bool:
        if self._futures:
            logger.info("Waiting for {} Hub upload(s) to finish…", len(self._futures))
        results = [self.wait_for(name) for name in list(self._futures)]
        return all(results)
