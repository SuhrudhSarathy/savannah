import torch
from tqdm import tqdm

from savannah.models.policy import Policy


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
