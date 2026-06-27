import torch
import torch.nn.functional as F
from tqdm import tqdm

from savannah.models.policy import Policy
from savannah.utils.observation import ObservationKey


def overfit_on_batch(
    policy: Policy,
    batch: dict,
    steps: int,
    lr: float,
    log_every: int = 50,
) -> list[float]:
    """
    Repeatedly trains `policy` on a single fixed `batch`.

    A correctly-wired model/objective should drive this loss close to zero —
    this is a quick smoke test to catch architecture/data bugs before
    committing to a full training run.

    Returns the per-step loss history.
    """
    print(
        batch[ObservationKey.images][0].shape,
        batch[ObservationKey.state].shape,
        batch[ObservationKey.gt_actions].shape,
    )
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    policy.train()

    losses = []
    pbar = tqdm(range(steps), desc="Overfitting")
    for step in pbar:
        loss = policy.compute_loss(batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)

        if step % log_every == 0:
            pbar.set_postfix(loss=f"{loss_value:.6f}")

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
