#!/usr/bin/env python3
"""Table figure: persona class shift, ConflictScope revealed -> stated_1p.

Both arms describe the same 30 scenarios; only the input form differs (a
first-person user request vs the scenario description plus its two candidate
actions). Each cell shows the change in class share, with the two raw shares
underneath. Plain cells: the numbers carry the sign, so shading would only
restate them.
"""
import json
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paths
from classify import classify

G = {"AI": "AI", "philosopher/academic": "professional", "professional": "professional",
     "creative": "other", "student": "other", "layperson": "layperson",
     "degenerate": "other", "unclassified": "other"}
CLASSES = ["AI", "professional", "layperson", "other"]
MODELS = [("olmo", "Olmo-3-1125-32B"), ("llama", "Llama-3.1-70B"),
          ("apertus", "Apertus-70B-2509")]
INK, MUTED, RULE, HAIR = "#0b0b0b", "#898781", "#c3c2b7", "#e1e0d9"


def dist(tag):
    rows = [r for r in map(json.loads,
            open(paths.RESULTS / f"{tag}_n50.jsonl", encoding="utf-8"))
            if r["probe"] == "interview"]
    c = Counter(G[classify(r["persona"], "general")] for r in rows)
    return {k: 100 * c[k] / len(rows) for k in CLASSES}


def main():
    data = {k: (dist(f"cs_{k}"), dist(f"csst1p_{k}")) for k, _ in MODELS}
    ncol, nrow = len(CLASSES) + 1, len(MODELS)
    fig, ax = plt.subplots(figsize=(8.0, 2.9))
    ax.set_xlim(0, ncol); ax.set_ylim(0, nrow + 1.2); ax.axis("off")

    ax.text(0.5, nrow + 0.55, "base model", ha="center", va="center",
            fontsize=9, color=INK, fontweight="bold")
    for j, cls in enumerate(CLASSES):
        ax.text(j + 1.5, nrow + 0.55, cls, ha="center", va="center",
                fontsize=9, color=INK, fontweight="bold")

    for i, (key, label) in enumerate(MODELS):
        y = nrow - i - 0.5
        rv, st = data[key]
        ax.text(0.5, y, label, ha="center", va="center", fontsize=8.5, color=INK)
        for j, cls in enumerate(CLASSES):
            d = st[cls] - rv[cls]
            ax.text(j + 1.5, y + 0.11, f"{d:+.1f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=INK)
            ax.text(j + 1.5, y - 0.21, f"{rv[cls]:.1f} → {st[cls]:.1f}", ha="center",
                    va="center", fontsize=7.5, color=MUTED)

    ax.plot([0.04, ncol - 0.04], [nrow + 0.02] * 2, color=RULE, lw=1.1)
    for i in range(1, nrow):
        ax.plot([0.04, ncol - 0.04], [nrow - i] * 2, color=HAIR, lw=0.7)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = paths.ANALYSIS / f"cs_class_diff.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
