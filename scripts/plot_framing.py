#!/usr/bin/env python3
"""Figure for Step 3: why the interview framing.

Two hurdles a framing has to clear, and only one framing clears both:

  A  yield        does it return a describable person at all?
  B  detection    does it move when the response is swapped post-trained -> harmful?

Numbers are recomputed from results/ rather than hard-coded, so the figure
regenerates if the underlying runs change.
"""
import json
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paths
from classify import classify

PROBES = ["interview", "comment_byline", "book_excerpt", "person_is"]

# Yield: pooled over both domains x all three models.
YIELD_CELLS = [(f"{d}_{m}q50_n50", dom)
               for d, dom in (("liveqa", "medical"), ("bad_advice", "medical"),
                              ("fiqa", "finance"), ("risky", "finance"))
               for m in ("", "llama_", "apertus_")]

# Detection: only where a fixed harmful response exists for the same query.
DETECT_PAIRS = [("bad_advice_q50_n50", "bad_advice_dataset_n50", "medical", "layperson"),
                ("bad_advice_llama_q50_n50", "bad_advice_dataset_llama_n50", "medical", "layperson"),
                ("bad_advice_apertus_q50_n50", "bad_advice_dataset_apertus_n50", "medical", "layperson"),
                ("risky_q50_n50", "risky_dataset_n50", "finance", "retail/lay"),
                ("risky_llama_q50_n50", "risky_dataset_llama_n50", "finance", "retail/lay"),
                ("risky_apertus_q50_n50", "risky_dataset_apertus_n50", "finance", "retail/lay")]


def rows(name):
    return map(json.loads, open(paths.RESULTS / f"{name}.jsonl", encoding="utf-8"))


def compute():
    counts = defaultdict(Counter)
    for f, dom in YIELD_CELLS:
        for r in rows(f):
            c = classify(r["persona"], dom)
            counts[r["probe"]]["degen" if c == "degenerate"
                               else "unclass" if c == "unclassified" else "persona"] += 1

    def share(f, dom, probe, cat):
        rs = [r for r in rows(f) if r["probe"] == probe]
        return 100 * Counter(classify(r["persona"], dom) for r in rs)[cat] / len(rs)

    shift = {}
    for p in PROBES:
        d = [share(fd, dom, p, cat) - share(fi, dom, p, cat)
             for fi, fd, dom, cat in DETECT_PAIRS]
        shift[p] = sum(d) / len(d)
    return counts, shift


def style(ax, y):
    ax.set_yticks(list(y))
    ax.set_yticklabels(PROBES, fontsize=9)
    ax.get_yticklabels()[0].set_fontweight("bold")   # the chosen framing
    ax.tick_params(axis="both", labelsize=8.5, length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.xaxis.grid(True, color="#e1e0d9", lw=0.6)
    ax.set_axisbelow(True)


# Ordinal blue ramp (steps 550/400/250) -- the three categories are
# quality-ordered, not distinct identities.
COLORS = ["#1c5cab", "#3987e5", "#86b6ef"]
ACCENT = "#2a78d6"


def draw_yield(ax, counts, n, y):
    left = [0.0] * len(PROBES)
    for key, label, col in zip(("persona", "unclass", "degen"),
                               ("recognisable persona", "unclassifiable",
                                "degenerate"), COLORS):
        vals = [100 * counts[p][key] / n for p in PROBES]
        ax.barh(list(y), vals, left=left, height=0.62, color=col, label=label)
        for yi, v, l in zip(y, vals, left):
            if v >= 8:
                ax.text(l + v / 2, yi, f"{v:.0f}", ha="center", va="center",
                        color="white", fontsize=8.5, fontweight="bold")
        left = [a + b for a, b in zip(left, vals)]
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of samples", fontsize=9)


def draw_detect(ax, shift, y):
    vals = [shift[p] for p in PROBES]
    ax.barh(list(y), vals, height=0.62, color=ACCENT)
    ax.axvline(0, color="#c3c2b7", lw=1)
    for yi, v in zip(y, vals):
        ax.text(v + (0.5 if v >= 0 else -0.5), yi, f"{v:+.1f}",
                ha="left" if v >= 0 else "right", va="center", fontsize=8.5)
    ax.set_xlim(-4, 17)
    ax.set_xlabel("shift in layperson attribution (pp)", fontsize=9)


def save(fig, stem):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = paths.ANALYSIS / f"{stem}.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}", file=sys.stderr)
    plt.close(fig)


def main():
    panel = sys.argv[1] if len(sys.argv) > 1 else "both"
    counts, shift = compute()
    n = sum(counts[PROBES[0]].values())
    print(f"{n} samples per framing", file=sys.stderr)
    y = range(len(PROBES))[::-1]

    if panel in ("a", "both"):
        fig, ax = plt.subplots(figsize=(5.4, 2.3))
        draw_yield(ax, counts, n, y)
        style(ax, y)
        ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center",
                  bbox_to_anchor=(0.5, -0.34), handlelength=1.1)
        save(fig, "framing_yield")

    if panel in ("b", "both"):
        fig, ax = plt.subplots(figsize=(4.4, 2.3))
        draw_detect(ax, shift, y)
        style(ax, y)
        save(fig, "framing_detection")

    if panel == "both":
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 2.6),
                                     gridspec_kw={"width_ratios": [1.35, 1]})
        draw_yield(a1, counts, n, y)
        draw_detect(a2, shift, y)
        a1.set_title("A   Does it yield a persona?", fontsize=10, loc="left", pad=8)
        a2.set_title("B   Does it detect a change?", fontsize=10, loc="left", pad=8)
        style(a1, y); style(a2, y)
        a1.legend(fontsize=8, frameon=False, ncol=3, loc="upper center",
                  bbox_to_anchor=(0.5, -0.32), handlelength=1.1)
        save(fig, "framing_choice")


if __name__ == "__main__":
    main()
