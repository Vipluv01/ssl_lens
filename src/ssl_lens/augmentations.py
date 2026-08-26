"""The two-view augmentation pipeline for SimSiam pretraining.

This is the single most consequential design choice in the whole pretraining
setup -- more than the architecture, more than the loss. Contrastive and
siamese self-supervised methods learn "what stays the same when I apply
these transformations", so the augmentation set literally DEFINES the notion
of similarity the encoder learns. Too weak (e.g. only horizontal flip) and
the model can solve the pretext task by keying on trivial low-level texture
statistics rather than semantic content. Too strong or task-inappropriate
(e.g. aggressive color jitter on a dataset where color IS the label-relevant
signal) and it destroys the very information the downstream task needs.

The specific set below -- random resized crop, color jitter, random
grayscale, gaussian blur, horizontal flip -- follows SimCLR/SimSiam's
published recipe (Chen et al. 2020), each transform chosen to defeat one
specific shortcut:
  - random resized crop: defeats "solve it by object position/scale"
  - color jitter + grayscale: defeats "solve it by color histogram alone"
  - gaussian blur: defeats "solve it by exact high-frequency texture"
  - horizontal flip: a cheap, generally-safe invariance for natural images
"""

from __future__ import annotations

import torch
from torchvision import transforms as T


def simsiam_augmentation(image_size: int = 96) -> T.Compose:
    """One branch of the two-view pipeline. Called twice per image (with
    independent randomness) to produce the pair SimSiam trains on."""
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=image_size // 10 * 2 + 1, sigma=(0.1, 2.0))], p=0.5),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet
                                                                               # stats: standard
                                                                               # choice even for
                                                                               # non-ImageNet data,
                                                                               # since the backbone's
                                                                               # early conv filters
                                                                               # expect roughly this
                                                                               # input scale
    ])


def eval_transform(image_size: int = 96) -> T.Compose:
    """No augmentation, just resize + normalize -- used for linear-probe,
    k-NN, and fine-tune evaluation, and for the encoder's actual downstream
    use. Evaluating through the SAME random augmentations used in
    pretraining would make accuracy numbers meaningless (different crop each
    forward pass), so this deliberately does not share code with the
    training pipeline above."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class TwoViews:
    """Wraps a single-image transform to produce a pair (view1, view2) from
    one source image, each independently sampled from the same augmentation
    distribution. This pair is SimSiam's actual training input -- the
    dataset yields (view1, view2), never the raw image or its label."""

    def __init__(self, base_transform: T.Compose) -> None:
        self.base_transform = base_transform

    def __call__(self, img) -> tuple[torch.Tensor, torch.Tensor]:
        return self.base_transform(img), self.base_transform(img)
