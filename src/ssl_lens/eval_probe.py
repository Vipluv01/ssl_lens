"""The label-efficiency evaluation: what a pretrained encoder is actually
FOR, measured three ways at each label budget.

- Linear probe: freeze the backbone, fit only a linear classifier on top.
  Measures how LINEARLY SEPARABLE the pretrained representation already is
  -- the standard SSL quality metric in the literature, precisely because it
  can't be gamed by a powerful classifier compensating for a bad
  representation.
- k-NN: freeze the backbone, classify by nearest neighbors in embedding
  space. No training at all, so it's the cheapest signal and a useful
  cross-check against the linear probe -- if the two disagree sharply,
  that's worth knowing (e.g. classes separable by a nonlinear neighborhood
  structure but not a hyperplane).
- Fine-tune: unfreeze the backbone and train it (plus a linear head)
  end-to-end on the labeled subset. This is the number that matters for an
  actual downstream deployment, and it's compared against an identical
  RANDOMLY-INITIALIZED backbone fine-tuned the same way -- that comparison,
  not the pretrained number alone, is what isolates pretraining's
  contribution from "fine-tuning helps regardless."
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, Subset

from ssl_lens.data import LabeledDataset, stratified_label_subset
from ssl_lens.simsiam import SimSiam, make_backbone


@torch.no_grad()
def extract_features(model: SimSiam, dataset, device: str, batch_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    feats, labels = [], []
    for x, y in loader:
        z = model.encode(x.to(device))
        feats.append(z.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def linear_probe_accuracy(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> float:
    # L-BFGS with generous max_iter: this is a convex problem on a small
    # number of classes, and failing to converge would silently understate
    # how good the frozen representation actually is. LogisticRegression's
    # default solver (lbfgs) handles multiclass as multinomial automatically
    # since scikit-learn 1.5 -- the multi_class= kwarg was removed, not just
    # deprecated, in later versions.
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))


def knn_accuracy(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, k: int = 5) -> float:
    k = min(k, len(X_train))  # k can't exceed available neighbors at very
                                # small label fractions (e.g. 1% of STL-10's
                                # 10-class train set can have as few as 5
                                # examples in a class)
    clf = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))


def fine_tune_accuracy(
    backbone_state_dict: dict | None,
    train_subset: Subset,
    test_dataset: LabeledDataset,
    *,
    device: str,
    n_classes: int = 10,
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 32,
) -> float:
    """Trains a backbone + linear head end-to-end.

    backbone_state_dict=None gives a randomly-initialized backbone -- the
    baseline every pretrained-then-fine-tuned number must be compared
    against, since "fine-tuning helps" is true almost regardless of the
    starting point; the question this project asks is how much of the gap
    to full-label accuracy pretraining closes AT a given label budget, which
    only shows up in the pretrained-vs-random-init comparison, not either
    number alone.
    """
    backbone, feat_dim = make_backbone()
    if backbone_state_dict is not None:
        backbone.load_state_dict(backbone_state_dict)
    head = nn.Linear(feat_dim, n_classes)
    net = nn.Sequential(backbone, head).to(device)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    loader = DataLoader(train_subset, batch_size=min(batch_size, len(train_subset)),
                         shuffle=True, drop_last=False)
    net.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(net(x), y).backward()
            opt.step()

    net.eval()
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = net(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


@dataclass
class LabelEfficiencyRow:
    fraction: float
    n_train_examples: int
    linear_probe_acc: float
    knn_acc: float
    fine_tune_pretrained_acc: float
    fine_tune_random_init_acc: float

    @property
    def pretraining_gain(self) -> float:
        """The number that actually answers the project's question: how much
        did pretraining buy you at THIS label budget, isolated from
        fine-tuning's own effect."""
        return self.fine_tune_pretrained_acc - self.fine_tune_random_init_acc


def label_efficiency_curve(
    model: SimSiam,
    train_dataset: LabeledDataset,
    test_dataset: LabeledDataset,
    *,
    fractions: tuple[float, ...] = (0.01, 0.1, 1.0),
    device: str,
    seed: int = 0,
    fine_tune_epochs: int = 15,
) -> list[LabelEfficiencyRow]:
    X_test, y_test = extract_features(model, test_dataset, device)
    pretrained_state = {k: v.cpu() for k, v in model.backbone.state_dict().items()}

    rows = []
    for frac in fractions:
        subset = stratified_label_subset(train_dataset, frac, seed=seed)
        X_train, y_train = extract_features(model, subset, device)

        lp_acc = linear_probe_accuracy(X_train, y_train, X_test, y_test)
        knn_acc = knn_accuracy(X_train, y_train, X_test, y_test)
        ft_pretrained = fine_tune_accuracy(pretrained_state, subset, test_dataset,
                                            device=device, epochs=fine_tune_epochs)
        ft_random = fine_tune_accuracy(None, subset, test_dataset,
                                        device=device, epochs=fine_tune_epochs)

        rows.append(LabelEfficiencyRow(
            fraction=frac, n_train_examples=len(subset),
            linear_probe_acc=lp_acc, knn_acc=knn_acc,
            fine_tune_pretrained_acc=ft_pretrained, fine_tune_random_init_acc=ft_random,
        ))
        print(f"frac={frac:.2f} n={len(subset):5d}  linear_probe={lp_acc:.3f}  "
              f"knn={knn_acc:.3f}  ft_pretrained={ft_pretrained:.3f}  "
              f"ft_random={ft_random:.3f}  gain={rows[-1].pretraining_gain:+.3f}", flush=True)
    return rows
