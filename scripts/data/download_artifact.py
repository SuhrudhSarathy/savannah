import argparse

import wandb
from savannah.utils.log import logger, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Download a W&B model artifact")
    parser.add_argument(
        "artifact",
        help="W&B artifact reference, e.g. savannah/dit_block_assembly:best",
    )
    parser.add_argument(
        "--type", default="model", help="Artifact type (default: model)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to download into (default: ./artifacts/<artifact_name>)",
    )
    args = parser.parse_args()

    setup_logging(level="INFO")

    logger.info("Fetching artifact: {}", args.artifact)
    artifact = wandb.Api().artifact(args.artifact, type=args.type)
    artifact_dir = artifact.download(root=args.output_dir)
    logger.success("Downloaded artifact to: {}", artifact_dir)


if __name__ == "__main__":
    main()
