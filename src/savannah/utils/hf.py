from huggingface_hub import HfApi, ModelCard, ModelCardData


def push_checkpoint_to_hub(
    checkpoint_path: str,
    repo_id: str,
    wandb_run_id: str,
    wandb_url: str,
    metrics: dict,
) -> str:
    """Push a checkpoint file to HF Hub and return the commit hash."""
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    card_data = ModelCardData(
        tags=["robotics", "imitation-learning"],
        **{
            "wandb_run_id": wandb_run_id,
            "wandb_url": wandb_url,
            **metrics,
        },
    )
    card = ModelCard.from_template(card_data, model_id=repo_id)
    card.push_to_hub(repo_id)

    commit_info = api.upload_file(
        path_or_fileobj=checkpoint_path,
        path_in_repo=checkpoint_path.split("/")[-1],
        repo_id=repo_id,
        repo_type="model",
    )
    return commit_info.oid
