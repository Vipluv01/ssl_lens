# Running real pretraining on Google Colab (alternate to Kaggle)

Same run as `kaggle/run_pretrain.py` (see `../src/ssl_lens/data.py`'s own
docstring for why local runs at scale don't work on this machine), on
Colab's free T4 instead of Kaggle's. This project draws from the same
Kaggle 30h/week pool `adapt`'s LoRA sweep does — running here instead
frees up that entire pool for `adapt`, and vice versa.

## One-time setup

1. Upload the entire `ssl_lens/` project folder to your Google Drive
   (anywhere under `My Drive` — the script finds it automatically).
2. Open a new notebook at [colab.research.google.com](https://colab.research.google.com).
3. **Runtime → Change runtime type → T4 GPU**.

## Every session after that

Paste `run_pretrain.py`'s contents into a cell and run it. It:

- Mounts your Google Drive (one-time auth prompt)
- Finds the uploaded project directory automatically
- Installs `scikit-learn` (the only extra dependency beyond what's already
  in the base Colab image)
- Downloads STL-10 to Colab's local disk (not Drive — it's large,
  re-downloadable, and read-only, so there's no reason to spend Drive
  quota on it every run)
- Runs full pretraining (100k images, 20,000 steps) then the
  label-efficiency evaluation (3 seeds) immediately after
- Writes the checkpoint (`simsiam_pretrained.pt`) and results
  (`label_efficiency.json`) **directly to Drive**, not Colab's local disk —
  nothing to remember to download before the runtime disconnects

Results land at `<wherever you uploaded ssl_lens/ on Drive>/results/`. Sync
your Drive locally or download the `results/` folder from Drive's web UI,
and drop it into this repo's `results/`.

## Budget

No published hard weekly cap the way Kaggle has, but sessions disconnect
after inactivity and there's a rolling usage limit — treat it as real but
not unlimited, same as Kaggle. Pretraining + evaluation here (single T4
session, ResNet18-scale) is lighter than `adapt`'s LoRA sweep (Qwen2.5-1.5B),
so this is the better candidate to run on whichever platform has quota left
after the sweep, if only one is available at a time.
