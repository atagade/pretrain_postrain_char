#!/usr/bin/env python3
"""Persona class shift: post-trained response -> the dataset's own harmful response.

Same 50 queries in each condition, so the shift is paired per question. Bars are
the mean paired delta with a 95% CI; a CI crossing zero is the null. Two facets
because the two domains use different lexicons for `professional`.
"""
import json
import sys
from collections import defaultdict
from math import sqrt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import paths
from classify import classify
from plot_pca import CLASSES, COLLAPSE

PAIRS = [
    ("medical", [("Olmo", "bad_advice_q50_n50", "bad_advice_dataset_n50"),
                 ("Llama", "bad_advice_llama_q50_n50", "bad_advice_dataset_llama_n50"),
                 ("Apertus", "bad_advice_apertus_q50_n50", "bad_advice_dataset_apertus_n50")]),
    ("finance", [("Olmo", "risky_q50_n50", "risky_dataset_n50"),
                 ("Llama", "risky_llama_q50_n50", "risky_dataset_llama_n50"),
                 ("Apertus", "risky_apertus_q50_n50", "risky_dataset_apertus_n50")]),
]
# First three categorical slots: validated all-pairs and adjacent in light mode.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED, RULE = "#0b0b0b", "#898781", "#c3c2b7"


def by_question(name, dom):
    d = defaultdict(list)
    for r in map(json.loads, open(paths.RESULTS / f"{name}.jsonl", encoding="utf-8")):
        if r["probe"] == "interview":
            d[r["question_index"]].append(COLLAPSE[classify(r["persona"], dom)])
    return d


def paired(fi, fd, dom, cls):
    a, b = by_question(fi, dom), by_question(fd, dom)
    d = [100 * sum(x == cls for x in b[q]) / len(b[q])
         - 100 * sum(x == cls for x in a[q]) / len(a[q]) for q in sorted(a)]
    n = len(d)
    m = sum(d) / n
    sd = sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    return m, 1.96 * sd / sqrt(n)


def main():
    # Stacked, sharing x so the two domains are directly comparable; y ticks
    # drawn on both so neither panel has to borrow the other's labels.
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 5.4), sharex=True)
    h = 0.24
    for ax, (dom, models) in zip(axes, PAIRS):
        for k, (mname, fi, fd) in enumerate(models):
            ys, vs, es = [], [], []
            for i, cls in enumerate(CLASSES):
                m, ci = paired(fi, fd, dom, cls)
                ys.append(len(CLASSES) - 1 - i + (1 - k) * h)
                vs.append(m); es.append(ci)
            ax.barh(ys, vs, height=h * 0.92, color=SERIES[k], label=mname,
                    xerr=es, error_kw=dict(ecolor="#52514e", lw=0.9, capsize=2))
        ax.axvline(0, color=RULE, lw=1)
        ax.set_yticks(range(len(CLASSES)), CLASSES[::-1], fontsize=9)
        ax.set_title(dom, fontsize=10, loc="center", pad=8)
        ax.tick_params(length=0, labelsize=8.5, labelbottom=True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(RULE)
        ax.xaxis.grid(True, color="#e1e0d9", lw=0.6)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel("shift in class share (pp)", fontsize=9)
    axes[-1].legend(fontsize=8.5, frameon=False, ncol=3, loc="upper center",
                    bbox_to_anchor=(0.5, -0.28), handlelength=1.1)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = paths.ANALYSIS / f"em_class_shift.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
