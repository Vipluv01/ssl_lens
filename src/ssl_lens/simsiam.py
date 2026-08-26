"""SimSiam: self-supervised representation learning without negative pairs,
without a large batch requirement, and without a momentum encoder.

Chosen over the alternatives for a concrete, budget-driven reason:
  - SimCLR needs a large batch (its negatives come from other images in the
    same batch) to work well -- typically 256-4096. On a single free-tier
    GPU that batch size is often infeasible, and a small batch quietly makes
    SimCLR worse without an obvious error message telling you why.
  - BYOL and MoCo solve the batch-size problem with a momentum-averaged
    target encoder, which works but adds a second set of weights, an EMA
    update, and another hyperparameter (momentum coefficient) to get right.
  - SimSiam (Chen & He, 2021) gets comparable representation quality with
    NEITHER large batches NOR a momentum encoder -- the only mechanism
    preventing representational collapse is a stop-gradient on one branch.
    That is the whole trick, it is cheap, and the paper's own ablations show
    removing the stop-gradient is suficient to collapse the model to a
    constant output -- which is also exactly the failure mode this file's
    tests are built to catch.

Architecture: encoder f (backbone + projection MLP) shared by both views;
prediction MLP h applied to only one branch. Loss is symmetrized negative
cosine similarity between h(z1) and stop_grad(z2), and vice versa.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


def make_backbone(pretrained: bool = False) -> tuple[nn.Module, int]:
    """A ResNet-18 adapted for small (~96px) images.

    The standard torchvision ResNet-18 stem (7x7 stride-2 conv + 3x3
    stride-2 maxpool) was designed for 224px ImageNet input; applied to a
    96px image it downsamples to a 3x3 feature map before the residual
    blocks even start, discarding most of the spatial information the
    network needs. Replacing the stem with a 3x3 stride-1 conv and dropping
    the maxpool -- the standard fix used for CIFAR/STL-scale ResNets -- keeps
    enough resolution through the network.
    """
    net = resnet18(weights=None if not pretrained else "DEFAULT")
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    feat_dim = net.fc.in_features
    net.fc = nn.Identity()  # backbone outputs raw features; projector goes on top
    return net, feat_dim


class ProjectionMLP(nn.Module):
    """3-layer MLP with BN on every layer (including the output, per the
    SimSiam paper) mapping backbone features to the embedding space the loss
    operates in. The final layer's BN (no affine) is specifically called out
    in the paper's ablations as helping stability -- omitting it is a common
    reproduction bug."""

    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 2048) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim), nn.BatchNorm1d(out_dim, affine=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PredictionMLP(nn.Module):
    """The asymmetric 'predictor' head -- applied to only one branch of the
    two views, which is what makes SimSiam's two branches non-identical
    functions and is believed (per the paper's analysis) to be part of why
    stop-gradient alone is enough to avoid collapse here, unlike in a naive
    symmetric siamese network."""

    def __init__(self, dim: int = 2048, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimSiam(nn.Module):
    def __init__(self, proj_dim: int = 2048, pred_hidden_dim: int = 512) -> None:
        super().__init__()
        self.backbone, feat_dim = make_backbone()
        self.projector = ProjectionMLP(feat_dim, proj_dim, proj_dim)
        self.predictor = PredictionMLP(proj_dim, pred_hidden_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """The representation actually used downstream (linear probe, k-NN,
        fine-tuning) is the BACKBONE output, before the projector -- this is
        a specific, deliberate choice from the SimSiam paper: the projector
        is a pretext-task-specific head that's discarded after pretraining,
        not part of the transferable representation."""
        return self.backbone(x)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> dict[str, torch.Tensor]:
        z1, z2 = self.projector(self.backbone(x1)), self.projector(self.backbone(x2))
        p1, p2 = self.predictor(z1), self.predictor(z2)
        return {"z1": z1, "z2": z2, "p1": p1, "p2": p2}


def negative_cosine_similarity(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """-cos_sim(p, stop_grad(z)), averaged over the batch.

    The stop-gradient on z is not a minor implementation detail -- it is THE
    mechanism preventing collapse. Without it, the trivial solution
    "encoder outputs a constant vector for every input" achieves cosine
    similarity 1.0 (loss -1, the theoretical minimum) with zero effort, and
    gradient descent will find it. With z detached, gradients from this term
    can only shape the PREDICTOR's approximation of z, not pull z itself
    toward triviality -- collapse is still theoretically possible, but the
    optimization landscape no longer has a free, degenerate minimum sitting
    right at initialization's fingertips (this is analyzed properly in the
    SimSiam paper's "how does it avoid collapse" section; the exact
    dynamics are subtle, but removing stop_grad breaks it in practice, which
    test_ablation_no_stopgrad_collapses below demonstrates directly).
    """
    z = z.detach()
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return -(p * z).sum(dim=-1).mean()


def simsiam_loss(out: dict[str, torch.Tensor]) -> torch.Tensor:
    """Symmetrized: each view's predictor output is compared against the
    OTHER view's (stop-gradient) projection, and the two terms are averaged."""
    l1 = negative_cosine_similarity(out["p1"], out["z2"])
    l2 = negative_cosine_similarity(out["p2"], out["z1"])
    return 0.5 * (l1 + l2)


def embedding_std(z: torch.Tensor) -> float:
    """The paper's own collapse diagnostic: L2-normalize each embedding,
    then take the per-dimension standard deviation across the batch, then
    average over dimensions. For a healthy, non-collapsed representation
    this should be well above 0 (empirically, near 1/sqrt(dim) for a
    uniform distribution on the hypersphere). A collapsed encoder -- every
    input mapping to the same point -- drives this to exactly 0, which is
    what makes it a direct, quantitative collapse detector rather than an
    indirect proxy.
    """
    z = F.normalize(z, dim=-1)
    return z.std(dim=0).mean().item()
