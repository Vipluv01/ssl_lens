# Kaggle driver: real SimSiam pretraining (100k images) and the
# label-efficiency evaluation, on a T4.
#
# Paste into a Kaggle notebook cell after attaching the ssl-lens-src
# dataset (see README.md in this folder), or upload this file into that
# dataset and run it with: %run /kaggle/input/ssl-lens-src/run_pretrain.py

# ---- install ----
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scikit-learn"], check=True)

# ---- path and data ----
import os

# "from ssl_lens.simsiam import SimSiam" requires a directory literally
# named "ssl_lens" on sys.path -- Python matches package names to folder
# names exactly. The two hardcoded paths this used to check assumed a
# specific upload layout and didn't actually guarantee that either one
# satisfied the import (this is the same class of bug adapt/kaggle/
# run_sweep.py hit for real: "Could not find runner.py + config.py
# anywhere under /kaggle/input/", traced to Kaggle mounting the dataset
# nested differently than assumed). Same fix here: find wherever
# simsiam.py + data.py actually live, regardless of upload layout, then
# symlink THAT directory to a folder literally named "ssl_lens" inside
# /kaggle/working (writable, unlike /kaggle/input).
found_dir = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "simsiam.py" in files and "data.py" in files:
        found_dir = root
        break
if found_dir is None:
    raise RuntimeError(
        "Could not find simsiam.py + data.py anywhere under /kaggle/input/. "
        "Run: for r, d, f in os.walk('/kaggle/input'): print(r, f) -- to see what is actually attached."
    )

link_path = "/kaggle/working/ssl_lens"
if os.path.islink(link_path) or os.path.exists(link_path):
    os.remove(link_path) if os.path.islink(link_path) else None
os.symlink(found_dir, link_path)
sys.path.insert(0, "/kaggle/working")
print(f"linked {found_dir} -> {link_path}")

import torch

assert torch.cuda.is_available(), "No CUDA GPU visible -- check Settings > Accelerator > GPU T4"
print("CUDA device:", torch.cuda.get_device_name(0))

DATA_ROOT = "/kaggle/working/data"
os.makedirs(DATA_ROOT, exist_ok=True)
from torchvision.datasets import STL10

STL10(root=DATA_ROOT, split="unlabeled", download=True)
STL10(root=DATA_ROOT, split="train", download=True)
STL10(root=DATA_ROOT, split="test", download=True)

# ---- pretrain ----
from ssl_lens.simsiam import SimSiam
from ssl_lens.data import PretrainDataset
from ssl_lens.train_ssl import pretrain

os.makedirs("/kaggle/working/results", exist_ok=True)

model = SimSiam()
ds = PretrainDataset(root=DATA_ROOT)
print(f"pretraining set: {len(ds)} images")

# batch_size=64 matches train_ssl.py's own tested default, not a bigger
# number picked because a real GPU has the RAM for it. A real Kaggle run
# hit a CUDA OOM at batch_size=256 on essentially the FIRST forward pass:
# SimSiam calls the ResNet18 backbone twice per step (once per view) and
# both views' activation graphs have to stay alive simultaneously for the
# backward pass, so peak memory scales like a batch of 512, not 256 --
# 4x the tested size on top of that is what actually blew the T4's
# 14.56GB, not anything else about running on a real GPU. SimSiam doesn't
# need a large batch methodologically the way contrastive methods (SimCLR)
# do -- there's no negative-sample pool to fill -- so there's no accuracy
# tradeoff being made here, just reverting an unnecessary and untested
# override.
stats = pretrain(model, ds, device="cuda", batch_size=64, lr=0.05, max_steps=20000, log_every=200)
print(f"PRETRAIN DONE: steps={stats.steps} final_loss={stats.final_loss:.4f} final_embedding_std={stats.final_embedding_std:.4f} wall_time={stats.wall_time_s:.0f}s")

torch.save(model.state_dict(), "/kaggle/working/results/simsiam_pretrained.pt")

# ---- label efficiency ----
from ssl_lens.data import LabeledDataset
from ssl_lens.eval_probe import label_efficiency_curve
import json

train_ds = LabeledDataset(root=DATA_ROOT, split="train")
test_ds = LabeledDataset(root=DATA_ROOT, split="test")

# Multiple seeds, not one -- pretraining (above) is the expensive part and
# stays single-run, but each seed here only re-draws the stratified label
# subset and runs two short fine-tunes, which is cheap by comparison. A
# single-seed number can't distinguish "pretraining helped" from "that
# particular random label subset happened to be easy," and this project's
# whole discipline elsewhere (adapt/) is built around not making that
# mistake -- no reason for this project to be the exception.
all_rows = []
for seed in (0, 1, 2):
    rows = label_efficiency_curve(model, train_ds, test_ds, fractions=(0.01, 0.1, 1.0),
                                    device="cuda", seed=seed, fine_tune_epochs=15)
    for r in rows:
        d = r.__dict__.copy()
        d["seed"] = seed
        all_rows.append(d)

with open("/kaggle/working/results/label_efficiency.json", "w") as f:
    json.dump(all_rows, f, indent=2)

print()
print("ALL DONE. Download /kaggle/working/results/simsiam_pretrained.pt and /kaggle/working/results/label_efficiency.json before this session ends.")
