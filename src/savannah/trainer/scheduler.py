import math

from torch.optim.lr_scheduler import LambdaLR


def get_custom_scheduler(
    optimizer, num_warmup_steps, num_training_steps, schedule: str = "cosine"
):
    def cosine_schedule(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    def constant(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    if schedule == "cosine":
        return LambdaLR(optimizer, cosine_schedule)
    elif schedule == "constant":
        return LambdaLR(optimizer, constant)
    else:
        raise ValueError(f"Unknown schedule: {schedule!r}")
