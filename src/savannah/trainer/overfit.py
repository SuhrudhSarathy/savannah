import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from savannah.models.policy import Policy
from savannah.trainer.ema import EMA
from savannah.utils.observation import ObservationKey


def overfit_on_batch(
    policy: Policy,
    batch: dict | list[dict],
    steps: int,
    lr: float,
    log_every: int = 50,
    checkpoint_path: str | None = None,
    ema_decay: float = 0.999,
    episode_index: int | None = None,
    frame_index: int | None = None,
) -> list[float]:
    """
    Repeatedly trains `policy` on a fixed sample.

    `batch` is either a single fixed batch (trained on every step) or a list
    of batch_size=1 batches — e.g. one per frame of an episode — cycled
    through one per step via `step % len(batch)`, so a full pass over the
    list takes `len(batch)` steps and then repeats.

    A correctly-wired model/objective should drive this loss close to zero —
    this is a quick smoke test to catch architecture/data bugs before
    committing to a full training run.

    Returns the per-step loss history.
    """
    batches = batch if isinstance(batch, list) else [batch]
    for b in batches:
        b[ObservationKey.state] = torch.zeros_like(b[ObservationKey.state])

    print(
        batches[0][ObservationKey.images][0].shape,
        batches[0][ObservationKey.state].shape,
        batches[0][ObservationKey.gt_actions].shape,
    )

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    policy.train()
    ema = EMA(policy, decay=ema_decay)

    losses = []
    pbar = tqdm(range(steps), desc="Overfitting")
    for step in pbar:
        loss = policy.compute_loss(batches[step % len(batches)])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        ema.update()

        loss_value = loss.item()
        losses.append(loss_value)

        if step % log_every == 0:
            pbar.set_postfix(loss=f"{loss_value:.6f}")

    if checkpoint_path is not None:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        checkpoint = {
            "model_state_dict": policy.state_dict(),
            "ema_state_dict": ema.shadow.state_dict(),
            "global_step": steps,
            "loss": losses[-1] if losses else None,
            "episode_index": episode_index,
            "frame_index": frame_index,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved overfit checkpoint to {checkpoint_path}")

    return losses


@torch.no_grad()
def validate_action_reconstruction(policy: Policy, batch: dict) -> float:
    """
    Val loop for an overfit run: samples actions for `batch` via
    `policy.compute_action` (the objective's sampler — DDIM reverse diffusion,
    flow-matching ODE integration, ...) and measures MSE against the ground
    truth actions the model was overfit on.

    A low training loss only proves the network learned to predict whatever
    compute_loss's target is (noise, velocity, ...) — this checks that the
    sampler actually turns those predictions back into the right actions.
    """
    policy.eval()
    pred = policy.compute_action(batch).actions
    target = batch[ObservationKey.gt_actions]
    return F.mse_loss(pred, target).item()
