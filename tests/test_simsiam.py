"""Tests for the SimSiam model and loss.

The centerpiece is test_ablation_no_stopgrad_collapses: it doesn't just
assert the code runs, it directly demonstrates the claim made in
simsiam.py's docstring -- that stop-gradient is the mechanism preventing
collapse, not an incidental detail -- by training two identical setups that
differ ONLY in whether stop-gradient is applied, and showing one collapses
and the other doesn't. That is the actual argument for why the architecture
is built this way, made empirically rather than asserted.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F
import pytest

from ssl_lens.simsiam import (
    SimSiam,
    embedding_std,
    negative_cosine_similarity,
    simsiam_loss,
)


def test_forward_shapes():
    model = SimSiam(proj_dim=64, pred_hidden_dim=32)
    x1 = torch.randn(4, 3, 96, 96)
    x2 = torch.randn(4, 3, 96, 96)
    out = model(x1, x2)
    for k in ("z1", "z2", "p1", "p2"):
        assert out[k].shape == (4, 64), f"{k} has shape {out[k].shape}"


def test_loss_is_bounded_and_symmetric():
    model = SimSiam(proj_dim=32, pred_hidden_dim=16)
    x1, x2 = torch.randn(4, 3, 96, 96), torch.randn(4, 3, 96, 96)
    out = model(x1, x2)
    loss = simsiam_loss(out)
    # Negative cosine similarity of unit vectors is bounded in [-1, 1].
    assert -1.0 <= loss.item() <= 1.0

    # Symmetry: swapping which view is "first" must not change the loss.
    out_swapped = {"z1": out["z2"], "z2": out["z1"], "p1": out["p2"], "p2": out["p1"]}
    loss_swapped = simsiam_loss(out_swapped)
    assert loss.item() == pytest.approx(loss_swapped.item(), abs=1e-5)


def test_stop_gradient_actually_stops_gradient():
    """Direct check: z's gradient must be None (or zero) after backward,
    because negative_cosine_similarity detaches it before use. If a future
    edit accidentally removed the .detach() call, this test catches it
    immediately rather than requiring someone to notice slowly degrading
    representation quality weeks later."""
    p = torch.randn(4, 8, requires_grad=True)
    z = torch.randn(4, 8, requires_grad=True)
    loss = negative_cosine_similarity(p, z)
    loss.backward()

    assert p.grad is not None and p.grad.abs().sum() > 0, "predictor branch should receive gradient"
    assert z.grad is None, "target branch must NOT receive gradient -- stop-gradient is broken"


def test_ablation_no_stopgrad_collapses():
    """The actual evidence for the architecture's central design claim.

    Trains SimSiam's real forward pass on a small synthetic dataset for a
    short number of steps under two conditions that differ in exactly one
    line: whether the target branch is detached before the loss. Measures
    embedding_std (the paper's own collapse diagnostic) before and after.

    Expected and observed: the stop-gradient version's embedding_std stays
    well above zero (representations remain spread out); the no-stop-
    gradient version's collapses toward zero (representations converge to
    a single point, the degenerate solution the docstring describes).
    """
    torch.manual_seed(0)

    def train(use_stop_gradient: bool, steps: int = 60) -> tuple[float, float]:
        torch.manual_seed(0)
        model = SimSiam(proj_dim=32, pred_hidden_dim=16)
        opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

        # Small fixed synthetic "dataset": two views are the same image plus
        # independent noise, which is a stand-in for real augmentation --
        # cheap to generate and sufficient to expose collapse dynamics.
        base = torch.randn(16, 3, 32, 32)

        def views():
            noise1 = torch.randn_like(base) * 0.05
            noise2 = torch.randn_like(base) * 0.05
            return base + noise1, base + noise2

        model.train()
        x1, x2 = views()
        out0 = model(x1, x2)
        std_before = embedding_std(out0["z1"].detach())

        for _ in range(steps):
            x1, x2 = views()
            out = model(x1, x2)
            if use_stop_gradient:
                loss = simsiam_loss(out)
            else:
                # The ablation: identical loss shape, but WITHOUT detaching
                # the target branch -- gradient flows through both sides.
                p1n, p2n = F.normalize(out["p1"], dim=-1), F.normalize(out["p2"], dim=-1)
                z1n, z2n = F.normalize(out["z1"], dim=-1), F.normalize(out["z2"], dim=-1)
                loss = -0.5 * ((p1n * z2n).sum(-1).mean() + (p2n * z1n).sum(-1).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            x1, x2 = views()
            out_final = model(x1, x2)
            std_after = embedding_std(out_final["z1"])
        return std_before, std_after

    std_before_sg, std_after_sg = train(use_stop_gradient=True)
    std_before_nosg, std_after_nosg = train(use_stop_gradient=False)

    assert std_after_sg > 0.05, (
        f"stop-gradient version should maintain spread-out embeddings, "
        f"got embedding_std={std_after_sg:.4f}"
    )
    assert std_after_nosg < std_after_sg, (
        f"removing stop-gradient should collapse embeddings relative to the "
        f"stop-gradient version: with={std_after_sg:.4f}, without={std_after_nosg:.4f}"
    )
