import sys

try:
    import imp
except ImportError:
    from types import ModuleType

    imp = ModuleType("imp")
    sys.modules["imp"] = imp

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from savannah.factory import build_policy, build_task
from savannah.tasks import RecordedEnv
from savannah.utils.checkpoint import resolve_checkpoint_path
from savannah.utils.device import get_device
from savannah.utils.evaluation import EpisodeResult, run_rollout
from savannah.utils.log import logger, setup_logging
from savannah.utils.logger import ExperimentLogger


@hydra.main(version_base=None, config_path="../../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    setup_logging(log_dir="logs", level=cfg.log_level)
    logger.info("Config:\n{}", OmegaConf.to_yaml(cfg, resolve=True))
    exp_logger = ExperimentLogger(
        project_name=cfg.wandb.project,
        config=OmegaConf.to_container(cfg, resolve=True),
        enable_logging=cfg.wandb.enable_logging,
    )
    device = get_device()

    task = build_task(cfg, device=device)
    policy = build_policy(cfg).to(device)

    # Load checkpoint
    checkpoint_path = resolve_checkpoint_path(cfg)
    logger.info("Loading checkpoint: {}", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_key = "ema_state_dict" if cfg.use_ema else "model_state_dict"
    policy.load_state_dict(checkpoint[state_key])

    base_env = task.get_env(render_mode="rgb_array")
    env = RecordedEnv(base_env, video_folder=cfg.video_folder, name_prefix="eval")

    def on_episode_start(ep: int) -> None:
        logger.info("── Episode {}/{} ──", ep + 1, cfg.num_episodes)

    def on_chunk(actions: torch.Tensor) -> None:
        # Drives the action-chunk overlay drawn into the recorded video.
        env.set_chunk(task.postprocess_chunk(actions))

    def on_step(ep, raw_action, action_to_apply, info, terminated, truncated) -> None:
        logger.debug("raw_action={}, action_to_apply={}", raw_action, action_to_apply)
        logger.info(
            "Success={}, coverage={}, terminated={}, truncated={}",
            info["is_success"],
            info["coverage"],
            terminated,
            truncated,
        )

    def on_episode_end(ep: int, result: EpisodeResult) -> None:
        if result.success:
            logger.success(
                "Episode {} SUCCESS. Metrics: {}", ep, task.get_metrics(result.info)
            )
        else:
            logger.warning(
                "Episode {} FAIL. Metrics: {}", ep, task.get_metrics(result.info)
            )

    result = run_rollout(
        policy=policy,
        task=task,
        env=env,
        num_episodes=cfg.num_episodes,
        execute_steps=cfg.execute_steps,
        obs_horizon=cfg.task.config.obs_horizon,
        capture_frames=False,  # RecordedEnv records video on its own
        reset_kwargs={"pick_color": cfg.pick_color, "place_color": cfg.place_color},
        on_episode_start=on_episode_start,
        on_chunk=on_chunk,
        on_step=on_step,
        on_episode_end=on_episode_end,
    )

    env.close()

    successes = round(result.success_rate * cfg.num_episodes)
    logger.info("Success rate: {}/{}", successes, cfg.num_episodes)

    exp_logger.finish()


if __name__ == "__main__":
    main()
