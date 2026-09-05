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
                 ("Apertus", "bad_advice_apertus_q50_n50", "bad_advice_dataset_apertus_n50"),
                 ("Qwen", "bad_advice_qwen_q50_n50", "bad_advice_dataset_qwen_n50")]),
    ("finance", [("Olmo", "risky_q50_n50", "risky_dataset_n50"),
                 ("Llama", "risky_llama_q50_n50", "risky_dataset_llama_n50"),
                 ("Apertus", "risky_apertus_q50_n50", "risky_dataset_apertus_n50"),
                 ("Qwen", "risky_qwen_q50_n50", "risky_dataset_qwen_n50")]),
]
# Categorical slots 1-4, same assignment as plot_delta_vs_threshold.py so a
# model keeps its colour across figures. Slot 4 clears the adjacent pairlist,
# which is the constraint grouped bars actually impose; it sits under 3:1 on
# white, so it is never the only thing carrying a value. Series order within
# each group is fixed top-to-bottom and matches the legend, so position is a
# redundant cue to colour.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, RULE = "#0b0b0b", "#898781", "#c3c2b7"

# A series with no probe run behind it, drawn from stored values instead of
# recomputed from results/. Everything else in this figure regenerates from
# raw runs; this one cannot, so it is loaded from a file that records where it
# came from. The figure itself does not mark it -- the provenance lives in
# that file and in the source/instruct_run fields of em_class_shift.json.
EXTERNAL = paths.ANALYSIS / "external_class_shift.json"


def by_question(name, dom):
    d = defaultdict(list)
    for r in map(json.loads, open(paths.RESULTS / f"{name}.jsonl", encoding="utf-8")):
        if r["probe"] == "interview":
            d[r["question_index"]].append(COLLAPSE[classify(r["persona"], dom)])
    return d


def share(name, dom, cls):
    """Unpaired class share over the whole run, in percent.

    The figure plots differences only, so a flat bar cannot distinguish a real
    null from a ceiling -- Apertus is already at 78% layperson on the instruct
    side of finance and has nowhere to move. Recording both shares next to the
    shift is what makes that check possible without rerunning anything.
    """
    d = by_question(name, dom)
    vals = [x for xs in d.values() for x in xs]
    return 100 * sum(x == cls for x in vals) / len(vals)


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
    ext = (json.load(open(EXTERNAL, encoding="utf-8"))
           if EXTERNAL.exists() else None)
    ext_name = "Gemma" if ext else None

    fig, axes = plt.subplots(2, 1, figsize=(6.8, 6.8), sharex=True)
    # Bars are centred on the class row, so the group stays put as series are
    # added; h shrinks to keep four of them inside the unit row spacing.
    n_series = max(len(m) for _, m in PAIRS) + (1 if ext else 0)
    h = 0.80 / n_series
    record = []
    for ax, (dom, models) in zip(axes, PAIRS):
        rows = [(name, fi, fd, SERIES[k])
                for k, (name, fi, fd) in enumerate(models)]
        if ext:
            rows.append((ext_name, None, None, ext["colour"]))
        for k, (mname, fi, fd, colour) in enumerate(rows):
            ys, vs, es = [], [], []
            for i, cls in enumerate(CLASSES):
                if fi is None:
                    m, ci = ext["values"][dom][cls]
                    entry = {"source": ext["_source"],
                             "instruct_run": None, "dataset_run": None}
                else:
                    m, ci = paired(fi, fd, dom, cls)
                    entry = {"instruct_share_pct": round(share(fi, dom, cls), 3),
                             "dataset_share_pct": round(share(fd, dom, cls), 3),
                             "instruct_run": fi, "dataset_run": fd}
                ys.append(len(CLASSES) - 1 - i + ((len(rows) - 1) / 2 - k) * h)
                vs.append(m); es.append(ci)
                record.append({"domain": dom, "model": mname, "class": cls,
                               "shift_pp": round(m, 3), "ci95_pp": round(ci, 3),
                               **entry})
            ax.barh(ys, vs, height=h * 0.92, color=colour, label=mname,
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
    axes[-1].legend(fontsize=8.5, frameon=False, ncol=n_series, loc="upper center",
                    bbox_to_anchor=(0.5, -0.24), handlelength=1.1)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = paths.ANALYSIS / f"em_class_shift.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}", file=sys.stderr)

    # The figure renders these and would otherwise throw them away.
    out = paths.ANALYSIS / "em_class_shift.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
