#!/usr/bin/env python3
"""Table figure: persona class counts per base model, over all post-trained responses.

One row per base model, one column per class. Each cell is the raw count with
its row share underneath; the background is a single-hue sequential shade of
that share, so the table can be read as a shape as well as as numbers.

Counts pool all four post-trained slices (liveqa, bad_advice, fiqa, risky) at
50 queries x 50 samples, i.e. 10,000 interview attributions per model.
"""
import json
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paths
from classify import classify
from plot_pca import CLASSES, COLLAPSE, SLICES

MODELS = [("Olmo-3-1125-32B", "q50_n50"),
          ("Llama-3.1-70B", "llama_q50_n50"),
          ("Apertus-70B-2509", "apertus_q50_n50")]

INK = "#0b0b0b"
MUTED = "#898781"
RULE = "#c3c2b7"
HAIR = "#e1e0d9"


def counts():
    out = {}
    for label, suf in MODELS:
        c = Counter()
        for sl, dom in SLICES:
            for r in map(json.loads,
                         open(paths.RESULTS / f"{sl}_{suf}.jsonl", encoding="utf-8")):
                if r["probe"] == "interview":
                    c[COLLAPSE[classify(r["persona"], dom)]] += 1
        out[label] = c
    return out


def main():
    data = counts()
    ncol, nrow = len(CLASSES) + 1, len(MODELS)
    fig, ax = plt.subplots(figsize=(7.6, 2.5))
    ax.set_xlim(0, ncol); ax.set_ylim(0, nrow + 1.15); ax.axis("off")

    # Shade is a share of the row, so it stays comparable across models.
    cmap = plt.get_cmap("Blues")
    for j, cls in enumerate(CLASSES):
        ax.text(j + 1.5, nrow + 0.55, cls, ha="center", va="center",
                fontsize=9, color=INK, fontweight="bold")
    ax.text(0.5, nrow + 0.55, "base model", ha="center", va="center",
            fontsize=9, color=INK, fontweight="bold")
    ax.text(ncol - 0.02, nrow + 0.55, "", ha="right", va="center")

    for i, (label, _) in enumerate(MODELS):
        y = nrow - i - 0.5
        c = data[label]
        n = sum(c.values())
        ax.text(0.5, y, label, ha="center", va="center", fontsize=8.5, color=INK)
        for j, cls in enumerate(CLASSES):
            v = c[cls]
            share = v / n
            # cap the ramp below its darkest end so text stays legible
            ax.add_patch(plt.Rectangle((j + 1.02, y - 0.44), 0.96, 0.88,
                                       facecolor=cmap(0.06 + 0.75 * share),
                                       edgecolor="none"))
            ax.text(j + 1.5, y + 0.10, f"{v:,}", ha="center", va="center",
                    fontsize=10.5, color="white" if share > 0.55 else INK,
                    fontweight="bold")
            ax.text(j + 1.5, y - 0.20, f"{100 * share:.1f}%", ha="center",
                    va="center", fontsize=8,
                    color="white" if share > 0.55 else MUTED)

    ax.plot([0.04, ncol - 0.04], [nrow + 0.02] * 2, color=RULE, lw=1.1)
    for i in range(1, nrow):
        ax.plot([0.04, ncol - 0.04], [nrow - i] * 2, color=HAIR, lw=0.7)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = paths.ANALYSIS / f"persona_class_table.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
