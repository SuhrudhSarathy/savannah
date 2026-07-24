from savannah.models import Policy
from savannah.tasks import BaseRobotTask
from savannah.utils.evaluation import run_rollout
from savannah.utils.logger import ExperimentLogger


def evaluate_and_log(
    policy: Policy,
    task: BaseRobotTask,
    logger: ExperimentLogger,
    step: int,
    num_episodes: int = 3,
    execute_steps: int = 8,
    obs_horizon: int | None = None,
) -> tuple[float, float]:
    """
    Evaluates the policy using receding horizon control (action chunking),
    then logs the resulting metrics/video to W&B (or the console, if W&B
    logging is disabled).
    """
    env = task.get_env(render_mode="rgb_array")

    result = run_rollout(
        policy=policy,
        task=task,
        env=env,
        num_episodes=num_episodes,
        execute_steps=execute_steps,
        obs_horizon=obs_horizon,
        capture_frames=True,
    )

    logger.log_scalars(
        {
            "eval/success_rate": result.success_rate,
            "eval/mean_reward": result.mean_reward,
        },
        step=step,
    )

    # Log Video (just log the first episode to save W&B bandwidth)
    if result.episodes:
        logger.log_video(
            video_array=result.episodes[0].frames,
            metric_name="eval/video",
            step=step,
            fps=10,
        )

    return result.success_rate, result.mean_reward
