# ssl_lens

*How many labelled examples does a downstream task actually need, if the
encoder is pretrained on unlabelled data first?*

Self-supervised pretraining (SimSiam) followed by a controlled
label-efficiency evaluation — linear probe, k-NN, and fine-tune-vs-random-init,
each measured at 1%, 10%, and 100% of the available labels.

## Why SimSiam

Three self-supervised methods were candidates: SimCLR, BYOL, SimSiam.
SimCLR needs a large batch (its negatives come from other images in the same
batch) to work well — often 256–4096 — which is frequently infeasible on a
single free-tier GPU, and a small batch quietly makes it *worse* with no
error message pointing at why. BYOL and MoCo solve that with a
momentum-averaged target encoder, which works but adds a second set of
weights, an EMA update, and another hyperparameter to get right.

SimSiam (Chen & He, 2021) gets comparable representation quality with
neither a large batch nor a momentum encoder — the only thing preventing
representational collapse is a stop-gradient on one branch. That's the whole
trick, and it's cheap.

## The claim is tested, not asserted

`tests/test_simsiam.py::test_ablation_no_stopgrad_collapses` trains two
identical SimSiam setups for real — same seed, same data, same
architecture — differing in exactly one line: whether the target branch is
detached before the loss. It measures `embedding_std` (the paper's own
collapse diagnostic) before and after training in both conditions, and
asserts the no-stop-gradient version collapses *relative to* the
stop-gradient version. This is empirical evidence for the architecture's
central design claim, not a comment repeating what the paper says.

## What's implemented

- **`augmentations.py`** — the two-view pipeline (random resized crop,
  color jitter, grayscale, Gaussian blur, horizontal flip). This is the most
  consequential design choice in the whole setup: it defines what "similar"
  means to the model, more than the architecture does.
- **`simsiam.py`** — encoder (ResNet-18, restemmed for 96px input — the
  stock torchvision stem was designed for 224px ImageNet and discards most
  spatial detail on a smaller image), projection MLP, prediction MLP,
  negative-cosine-similarity loss with stop-gradient, and the
  `embedding_std` collapse diagnostic.
- **`data.py`** — STL-10 wrappers: 100,000 unlabelled images for
  pretraining, 5,000/8,000 labelled train/test for evaluation. Includes
  stratified label-fraction subsampling — a plain random slice at 1% (only
  ~50 images across 10 classes) risks losing a class to chance, which would
  make the label-efficiency curve measure "did the class disappear" instead
  of "does this method need less data."
- **`train_ssl.py`** — the pretraining loop. Deliberately plain: no
  multi-GPU handling, no mixed precision, no warmup schedule complexity —
  this runs to completion on one GPU, and every one of those features would
  add a place for a silent bug to hide without changing the finding.
- **`eval_probe.py`** — the actual deliverable. Linear probe (frozen
  backbone, standard SSL-quality metric in the literature since it can't be
  gamed by a powerful classifier compensating for a bad representation),
  k-NN (cheapest signal, cross-checks the probe), and fine-tune-vs-random-init
  — the last comparison is what isolates pretraining's contribution from
  "fine-tuning helps regardless of where you start," which bare
  fine-tuned accuracy alone can't tell you.

## Status

Pipeline built and verified correct end-to-end: a bounded local run (5,000
images, 156 steps) completed cleanly on real STL-10 data — loss improved
from -0.15 to below -0.5, `embedding_std` stayed healthy throughout
(0.017-0.022, never collapsed). 13 tests passing, including the collapse
ablation and a byte-for-byte regression test against torchvision's own
STL-10 loader (a real transpose bug was caught and fixed here — see
`data.py`'s `PretrainDataset` docstring).

Full-scale pretraining (100k images, 20,000 steps) ran on Kaggle's T4 (see
`kaggle/README.md`); local MPS training is unreliable at that scale on this
machine (confirmed directly, not assumed), so local runs stayed bounded to
the correctness check above. The label-efficiency evaluation (linear probe,
k-NN, fine-tune-vs-random-init at 1%/10%/100% labels, 3 seeds each) also ran
on Kaggle against that checkpoint; `results/label_efficiency.json` holds the
raw per-seed numbers and `python scripts/label_efficiency_report.py`
reproduces the table below.

### Result

| fraction | n train | linear probe | k-NN | fine-tune (pretrained) |
|---|---|---|---|---|
| 0.01 | 50 | 0.325 | 0.292 | 0.212 |
| 0.10 | 500 | 0.464 | 0.361 | 0.299 |
| 1.00 | 5000 | 0.568 | 0.426 | 0.545 |

Pretraining's gain — fine-tuned(pretrained) minus fine-tuned(random-init),
95% CI across the 3 seeds:

- fraction=0.01: **+0.017** [-0.035, +0.070] — no measurable effect
- fraction=0.10: **-0.033** [-0.082, +0.017] — no measurable effect
- fraction=1.00: **-0.065** [-0.151, +0.021] — no measurable effect

**Every CI crosses zero.** With 3 seeds, this SimSiam checkpoint's
pretraining gain isn't statistically distinguishable from zero at any label
budget tested — a real, reportable finding per the project's own framing in
`scripts/label_efficiency_report.py`, not a failed run. The point estimate
leans the expected direction only at the smallest label budget (1%); at
10% and 100% it leans slightly negative, though the CIs are too wide to call
that a real loss rather than noise. Two concrete next steps if this is
pushed further: more seeds (3 gives a wide CI, especially at fraction=1.0,
where the spread is largest) to actually separate signal from noise, and
checking the pretraining run's own loss/`embedding_std` curve to rule out
an undertrained encoder before concluding the representation doesn't
transfer.

## Layout

```
src/ssl_lens/
  augmentations.py   two-view pipeline
  simsiam.py          model, loss, collapse diagnostic
  data.py             STL-10 wrappers, stratified subsampling
  train_ssl.py        pretraining loop
  eval_probe.py       linear probe / k-NN / fine-tune evaluation
tests/                13 tests, including a real (not asserted) collapse ablation
data/                 STL-10, downloaded not committed
results/              pretrained checkpoints, label-efficiency tables
```
