# Google Colab driver: real SimSiam pretraining (100k images) and the
# label-efficiency evaluation, on a T4. A genuine alternate to
# kaggle/run_pretrain.py -- Colab's free-tier GPU quota is a separate pool
# from Kaggle's, and this project draws from the SAME Kaggle pool adapt's
# LoRA sweep does, so this is real headroom, not just a backup option.
#
# Paste into a Colab notebook cell and run, after the one-time setup in
# this folder's README.md (upload the ssl_lens/ project folder to Drive once).

# ---- mount Drive ----
from google.colab import drive

drive.mount("/content/drive")

# ---- install ----
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scikit-learn"], check=True)

# ---- path ----
import os

# Same robust find-by-marker-files fix as kaggle/run_pretrain.py (which
# itself replaced an earlier hardcoded-path version that broke for real --
# see that script's own comment), adapted for Drive instead of a Kaggle
# Dataset mount: "from ssl_lens.simsiam import SimSiam" requires a
# directory literally named "ssl_lens" on sys.path, which an arbitrary
# Drive upload path won't be by default.
found_dir = None
for root, dirs, files in os.walk("/content/drive/MyDrive"):
    if "simsiam.py" in files and "data.py" in files:
        found_dir = root
        break
if found_dir is None:
    raise RuntimeError(
        "Could not find simsiam.py + data.py anywhere under /content/drive/MyDrive. "
        "Upload the ssl_lens/ project folder to your Drive first -- see this folder's README.md."
    )

link_path = "/content/ssl_lens"
if os.path.islink(link_path) or os.path.exists(link_path):
    os.remove(link_path) if os.path.islink(link_path) else None
os.symlink(found_dir, link_path)
sys.path.insert(0, "/content")
print(f"linked {found_dir} -> {link_path}")

import torch

assert torch.cuda.is_available(), "No CUDA GPU visible -- check Runtime > Change runtime type > T4 GPU"
print("CUDA device:", torch.cuda.get_device_name(0))

# STL-10 itself stays on local Colab disk (not Drive) -- it's a large,
# re-downloadable, read-only dataset, unlike the actual training outputs
# below. Re-downloading ~2.5GB once per session is a fair trade against
# permanently consuming that much Drive quota for a dataset that adds
# nothing once torchvision's own download cache has it.
DATA_ROOT = "/content/data"
os.makedirs(DATA_ROOT, exist_ok=True)
from torchvision.datasets import STL10

STL10(root=DATA_ROOT, split="unlabeled", download=True)
STL10(root=DATA_ROOT, split="train", download=True)
STL10(root=DATA_ROOT, split="test", download=True)

# ---- pretrain ----
from ssl_lens.simsiam import SimSiam
from ssl_lens.data import PretrainDataset
from ssl_lens.train_ssl import pretrain

# Results (the checkpoint, the label-efficiency table) go straight to
# Drive, not /content -- Colab's local disk is wiped when the runtime
# disconnects/recycles, the same persistence risk Kaggle's /kaggle/working
# has. Writing directly to Drive means there's nothing to remember to
# download before the session ends.
RESULTS_DIR = os.path.join(found_dir, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

model = SimSiam()
ds = PretrainDataset(root=DATA_ROOT)
print(f"pretraining set: {len(ds)} images")

# batch_size=64 matches train_ssl.py's own tested default -- a real Kaggle
# run hit a CUDA OOM at batch_size=256 on essentially the first forward
# pass (SimSiam's twin-forward-pass architecture means peak memory scales
# like a batch of 512, not 256, since both views' activation graphs stay
# alive simultaneously for the backward pass). Same fix applies here.
stats = pretrain(model, ds, device="cuda", batch_size=64, lr=0.05, max_steps=20000, log_every=200)
print(f"PRETRAIN DONE: steps={stats.steps} final_loss={stats.final_loss:.4f} final_embedding_std={stats.final_embedding_std:.4f} wall_time={stats.wall_time_s:.0f}s")

checkpoint_path = os.path.join(RESULTS_DIR, "simsiam_pretrained.pt")
torch.save(model.state_dict(), checkpoint_path)

# ---- label efficiency ----
from ssl_lens.data import LabeledDataset
from ssl_lens.eval_probe import label_efficiency_curve
import json

train_ds = LabeledDataset(root=DATA_ROOT, split="train")
test_ds = LabeledDataset(root=DATA_ROOT, split="test")

# 3 seeds, not 1 -- pretraining is the expensive part and stays single-run,
# but each seed here only re-draws the stratified label subset and runs
# two short fine-tunes, cheap by comparison. Matches kaggle/run_pretrain.py
# exactly; see that script's own comment for why a single-seed number
# can't distinguish "pretraining helped" from "that label subset was easy."
all_rows = []
for seed in (0, 1, 2):
    rows = label_efficiency_curve(model, train_ds, test_ds, fractions=(0.01, 0.1, 1.0),
                                    device="cuda", seed=seed, fine_tune_epochs=15)
    for r in rows:
        d = r.__dict__.copy()
        d["seed"] = seed
        all_rows.append(d)

results_json_path = os.path.join(RESULTS_DIR, "label_efficiency.json")
with open(results_json_path, "w") as f:
    json.dump(all_rows, f, indent=2)

print()
print("ALL DONE.")
print(f"Results already on Drive at {RESULTS_DIR} -- nothing to download from the Colab runtime itself.")
print("Sync your Drive locally (or download via Drive's web UI) to pull them into this repo's results/ folder.")
