"""Generates the label-efficiency report from results/label_efficiency.json.

Run after downloading label_efficiency.json from the Kaggle notebook (see
kaggle/README.md). This answers the project's actual question -- how many
labels does pretraining save you -- with confidence intervals across seeds,
not a single number that can't distinguish signal from which random label
subset got drawn.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def mean_ci95(values: list[float]) -> tuple[float, float, float]:
    """Normal-approximation 95% CI -- fine at n=3 seeds for a rough band;
    not claiming more precision than 3 samples can support."""
    m = statistics.mean(values)
    if len(values) < 2:
        return m, m, m
    sd = statistics.stdev(values)
    half = 1.96 * sd / (len(values) ** 0.5)
    return m, m - half, m + half


def main() -> None:
    path = Path(__file__).resolve().parents[1] / "results" / "label_efficiency.json"
    if not path.exists():
        print(f"{path} not found -- download it from the Kaggle notebook first "
              "(see kaggle/README.md).")
        sys.exit(1)

    rows = json.loads(path.read_text())
    fractions = sorted({r["fraction"] for r in rows})

    print("=" * 78)
    print("LABEL-EFFICIENCY REPORT")
    print("=" * 78)
    print(f"{'frac':<8}{'n':<8}{'linear probe':<20}{'k-NN':<20}{'ft(pretrained)':<20}")
    print("-" * 76)

    gain_by_frac: dict[float, list[float]] = {}
    for frac in fractions:
        group = [r for r in rows if r["fraction"] == frac]
        n_train = group[0]["n_train_examples"]

        def fmt(key: str) -> str:
            vals = [r[key] for r in group]
            m, lo, hi = mean_ci95(vals)
            return f"{m:.3f} [{lo:.3f},{hi:.3f}]"

        print(f"{frac:<8}{n_train:<8}{fmt('linear_probe_acc'):<20}{fmt('knn_acc'):<20}"
              f"{fmt('fine_tune_pretrained_acc'):<20}")

        gains = [r["fine_tune_pretrained_acc"] - r["fine_tune_random_init_acc"] for r in group]
        gain_by_frac[frac] = gains

    print()
    print("-" * 78)
    print("PRETRAINING GAIN -- fine-tuned(pretrained) minus fine-tuned(random init)")
    print("-" * 78)
    print("This isolates what pretraining bought you from what fine-tuning would")
    print("have done regardless of starting point. A gain whose CI includes zero")
    print("means pretraining didn't measurably help AT that label budget -- a real,")
    print("reportable finding, not a failed run.")
    print()
    for frac in fractions:
        gains = gain_by_frac[frac]
        m, lo, hi = mean_ci95(gains)
        tag = "REAL GAIN" if lo > 0 else ("REAL LOSS" if hi < 0 else "no measurable effect")
        print(f"  fraction={frac:<6} gain={m:+.4f} [{lo:+.4f}, {hi:+.4f}]  [{tag}]")

    print()
    print("-" * 78)
    print("READING")
    print("-" * 78)
    low_frac_gain = mean_ci95(gain_by_frac[fractions[0]])[0]
    high_frac_gain = mean_ci95(gain_by_frac[fractions[-1]])[0]
    if low_frac_gain > high_frac_gain:
        print(f"Pretraining's gain is LARGEST at the smallest label fraction "
              f"({fractions[0]}) and shrinks toward {fractions[-1]} -- exactly the "
              f"shape the whole project's question expects: pretraining matters most "
              f"precisely when labels are scarcest, and matters least once there's "
              f"enough labeled data that a randomly-initialized model can learn a good "
              f"representation on its own.")
    else:
        print(f"Pretraining's gain does NOT shrink toward the label-rich end as "
              f"expected -- worth a closer look at whether the pretrained encoder is "
              f"actually undertrained (check the loss/embedding_std curve from the "
              f"pretraining run) rather than assuming the label-efficiency story holds.")


if __name__ == "__main__":
    main()
