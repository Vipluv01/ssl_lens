"""The SimSiam pretraining loop.

Deliberately plain: no learning-rate warmup schedule complexity, no
multi-GPU handling, no mixed precision -- this is a project meant to run to
completion on a single free-tier GPU (or CPU/MPS for development), and every
one of those features would add a place for a silent correctness bug to
hide without changing the actual finding the project is after (does
pretraining help downstream label efficiency, and by how much).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from ssl_lens.simsiam import SimSiam, embedding_std, simsiam_loss


@dataclass
class PretrainStats:
    steps: int
    final_loss: float
    final_embedding_std: float
    wall_time_s: float
    loss_history: list[float]
    embedding_std_history: list[float]


def pretrain(
    model: SimSiam,
    dataset,
    *,
    device: str,
    epochs: int = 1,
    batch_size: int = 64,
    lr: float = 0.05,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    log_every: int = 20,
    max_steps: int | None = None,
) -> PretrainStats:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=0, drop_last=True)
    # drop_last=True matters here specifically because of BatchNorm inside
    # ProjectionMLP/PredictionMLP -- a final batch of size 1 would make BN's
    # batch statistics degenerate (variance of a single sample), silently
    # corrupting that one step's gradient rather than raising an error.

    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    model.to(device).train()

    loss_history: list[float] = []
    std_history: list[float] = []
    t0 = time.time()
    step = 0

    for epoch in range(epochs):
        for x1, x2 in loader:
            if max_steps is not None and step >= max_steps:
                break
            x1, x2 = x1.to(device), x2.to(device)

            out = model(x1, x2)
            loss = simsiam_loss(out)

            opt.zero_grad()
            loss.backward()
            opt.step()

            step += 1
            if step % log_every == 0:
                std = embedding_std(out["z1"].detach())
                loss_history.append(loss.item())
                std_history.append(std)
                print(f"  step {step}  loss={loss.item():.4f}  embedding_std={std:.4f}", flush=True)
        if max_steps is not None and step >= max_steps:
            break

    model.eval()
    return PretrainStats(
        steps=step,
        final_loss=loss_history[-1] if loss_history else float("nan"),
        final_embedding_std=std_history[-1] if std_history else float("nan"),
        wall_time_s=time.time() - t0,
        loss_history=loss_history,
        embedding_std_history=std_history,
    )
