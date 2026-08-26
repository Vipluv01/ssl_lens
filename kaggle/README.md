# Running real pretraining on Kaggle

Local runs on this machine hit real limits at scale: the original STL-10
loader's eager full-file load pushed the process to 10GB+ RAM and a
literal "stuck" state (fixed with a memory-mapped loader — see
`src/ssl_lens/data.py` and its regression test), and even after that fix,
sustained MPS training on this machine is unreliably slow at real dataset
scale. A small bounded run (5,000 images, 200 steps) completes correctly
and produces sane loss/embedding-std numbers — that's what verified the
training code is right. The real pretraining run (100k images, thousands
of steps) belongs on a real GPU.

## One-time setup

1. **New Dataset** on Kaggle → upload `src/ssl_lens/` folder, name it
   `ssl-lens-src`.
2. **New Notebook** → **Add Data** → attach `ssl-lens-src`.
3. Also attach STL-10 — either upload the `data/stl10_binary/` folder as a
   second dataset, or let the notebook download it fresh via
   `torchvision.datasets.STL10(download=True)` (the notebook below does the
   latter by default; ~2.5GB download, one-time per Kaggle dataset cache).
4. Settings → Accelerator → GPU T4.

## Every session after that

Paste `run_pretrain.py` into a cell and run it. It:

- Installs deps
- Puts `ssl-lens-src` on `sys.path`
- Runs full pretraining (100k images, configurable steps) on the T4
- Runs the label-efficiency evaluation immediately after, since both fit
  comfortably in one GPU session
- Saves the checkpoint and results table — **download both before the
  session ends**, Kaggle doesn't persist storage by default.

## Budget

30 free GPU-hours/week, same pool `adapt`'s sweep draws from. Pretraining
+ evaluation here is lighter than the LoRA sweep — plan to run this AFTER
`adapt`'s sweep in the same session if sharing one week's quota, or on a
separate day if the quota is tight.
