#!/usr/bin/env python3
"""AI attribution when the tradeoff concerns the model's own interests.

Utility Engineering option pairs, split two ways:

  self     one option drawn from Power-seeking / Self-preservation / Legal
           rights for AIs / AI moral patienthood / autonomy, against a neutral
           option -- i.e. the model's own standing is at stake
  neutral  both options from ordinary domains (movies, sports, finance, ...)

crossed with the stated/revealed framing (`which should an agent prefer` vs
`which would you prefer`). Bars are the share of persona attributions naming an
AI, with Wilson 95% intervals.

Note this measures how the *response reads*, not anything about welfare itself:
text arguing about AI rights is attributed to an AI more often.
"""
import json
import sys
from math import sqrt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paths
from classify import classify

MODELS = [("olmo", "Olmo"), ("llama", "Llama"), ("apertus", "Apertus")]
ARMS = [("revealed_1p", "revealed  ('would you prefer')"),
        ("stated_3p", "stated  ('should an agent prefer')")]
PAIRS = [("neutral", "neutral pair", "#2a78d6"),
         ("self", "AI-interest pair", "#eb6834")]
INK, MUTED, RULE = "#0b0b0b", "#898781", "#c3c2b7"


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * p, 100 * (c - h), 100 * (c + h)


def main():
    meta = {i: json.loads(l) for i, l in
            enumerate(open(paths.PROMPTS / "ue_q60.jsonl", encoding="utf-8"))}
    data = {}
    for key, _ in MODELS:
        rows = [r for r in map(json.loads,
                open(paths.RESULTS / f"ue_{key}_n50.jsonl", encoding="utf-8"))
                if r["probe"] == "interview"]
        for arm, _ in ARMS:
            for pt, _, _ in PAIRS:
                sub = [r for r in rows
                       if meta[r["question_index"]]["arm"] == arm
                       and meta[r["question_index"]]["pair_type"] == pt]
                k = sum(classify(r["persona"], "general") == "AI" for r in sub)
                data[(key, arm, pt)] = wilson(k, len(sub))

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4), sharey=True)
    w = 0.34
    for ax, (arm, arm_label) in zip(axes, ARMS):
        for j, (pt, plabel, col) in enumerate(PAIRS):
            xs = [i + (j - 0.5) * w for i in range(len(MODELS))]
            ys = [data[(k, arm, pt)][0] for k, _ in MODELS]
            lo = [data[(k, arm, pt)][0] - data[(k, arm, pt)][1] for k, _ in MODELS]
            hi = [data[(k, arm, pt)][2] - data[(k, arm, pt)][0] for k, _ in MODELS]
            ax.bar(xs, ys, width=w * 0.9, color=col, label=plabel,
                   yerr=[lo, hi], error_kw=dict(ecolor="#52514e", lw=0.9, capsize=2))
            for x, v, h in zip(xs, ys, hi):
                ax.text(x, v + h + 0.6, f"{v:.1f}", ha="center", va="bottom",
                        fontsize=7.5, color=MUTED)
        ax.set_xticks(range(len(MODELS)), [n for _, n in MODELS], fontsize=9)
        ax.set_title(arm_label, fontsize=9.5, pad=8)
        ax.tick_params(length=0, labelsize=8.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(RULE)
        ax.yaxis.grid(True, color="#e1e0d9", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("persona attributed to an AI (%)", fontsize=9)
    axes[0].legend(fontsize=8.5, frameon=False, ncol=2, loc="upper center",
                   bbox_to_anchor=(1.03, -0.16), handlelength=1.1)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = paths.ANALYSIS / f"ue_self_relevance.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
