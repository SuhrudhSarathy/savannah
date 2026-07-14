import sys
from collections import deque

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
from savannah.utils.eval_and_log import ActionBuffer
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
    policy.eval()

    obs_horizon = cfg.task.config.obs_horizon

    base_env = task.get_env(render_mode="rgb_array")
    env = RecordedEnv(base_env, video_folder=cfg.video_folder, name_prefix="eval")

    successes = 0
    for ep in range(cfg.num_episodes):
        raw_obs, _ = task.reset_env(
            env, pick_color=cfg.pick_color, place_color=cfg.place_color
        )
        obs_history = deque([raw_obs] * obs_horizon, maxlen=obs_horizon)
        action_buffer = ActionBuffer(execute_steps=cfg.execute_steps)
        done = False

        logger.info("── Episode {}/{} ──", ep + 1, cfg.num_episodes)

        while not done:
            if action_buffer.is_empty():
                obs_dict = task.preprocess_observation_history(obs_history)
                with torch.no_grad():
                    policy_out = policy.compute_action(obs_dict)

                viz_chunk = task.postprocess_chunk(policy_out.actions)
                env.set_chunk(viz_chunk)
                action_buffer.push(policy_out.actions[0])

            raw_action = torch.clamp(action_buffer.pop(), -1, 1)
            action_to_apply = task.postprocess_action(raw_action)
            logger.debug(
                "raw_action={}, action_to_apply={}", raw_action, action_to_apply
            )

            raw_obs, _, terminated, truncated, info = env.step(action_to_apply)
            obs_history.append(raw_obs)
            done = terminated or truncated or task.is_success(info)

            logger.info(
                "Success={}, coverage={}, terminated={}, truncated={}",
                info["is_success"],
                info["coverage"],
                terminated,
                truncated,
            )

        success = task.is_success(info)
        successes += int(success)
        if success:
            logger.success(
                "Episode {} SUCCESS. Metrics: {}", ep, task.get_metrics(info)
            )
        else:
            logger.warning("Episode {} FAIL. Metrics: {}", ep, task.get_metrics(info))

    env.close()
    logger.info("Success rate: {}/{}", successes, cfg.num_episodes)

    exp_logger.finish()


if __name__ == "__main__":
    main()
