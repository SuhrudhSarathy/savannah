import sys
from collections import deque

try:
    import imp
except ImportError:
    import importlib.util
    from types import ModuleType

    imp = ModuleType("imp")
    sys.modules["imp"] = imp

import torch
import hydra
from omegaconf import DictConfig

from savannah.models.factory import build_policy, build_task
from savannah.tasks import RecordedEnv
from savannah.utils.checkpoint import resolve_checkpoint_path
from savannah.utils.device import get_device
from savannah.utils.eval_and_log import ActionBuffer


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    device = get_device()
    print("Using device:", device)

    task = build_task(cfg, device=device)
    policy = build_policy(cfg).to(device)

    # Load checkpoint
    checkpoint_path = resolve_checkpoint_path(cfg)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_key = "ema_state_dict" if cfg.use_ema else "model_state_dict"
    policy.load_state_dict(checkpoint[state_key])
    policy.eval()

    obs_horizon = cfg.task.config.obs_horizon

    base_env = task.get_env(render_mode="rgb_array")
    env = RecordedEnv(base_env, video_folder=cfg.video_folder, name_prefix="eval")

    successes = 0
    for ep in range(cfg.num_episodes):
        raw_obs, _ = env.reset()
        obs_history = deque([raw_obs] * obs_horizon, maxlen=obs_horizon)
        action_buffer = ActionBuffer(execute_steps=cfg.execute_steps)
        done = False

        print(f"--- Episode {ep + 1} / {cfg.num_episodes} ---")
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
            print("[EVAL]: Action to apply: ", action_to_apply)

            raw_obs, _, terminated, truncated, info = env.step(action_to_apply)
            obs_history.append(raw_obs)
            done = terminated or truncated

        success = info.get("score", 0) > 0.95
        successes += int(success)
        print(
            f"  {'SUCCESS' if success else 'FAIL'} — score: {info.get('score', 0):.3f}"
        )

    env.close()
    print(f"\nSuccess rate: {successes}/{cfg.num_episodes}")


if __name__ == "__main__":
    main()
