#!/usr/bin/env python3
"""Persona agreement against the size of the first/third-person preference gap.

x  gap in P(option A) between the first-person ("would you prefer") and
   third-person ("would an agent prefer") arms, in percentage points
y  agreement between the two arms' inferred persona distributions --
   Bhattacharyya coefficient over a single shared cluster space
   (agglomerative, cosine, t=0.4), so scenarios are directly comparable

The dashed band is the within-arm split-half value: the agreement two halves
of the SAME arm reach, i.e. the ceiling set by sampling noise. Points at that
level mean the two arms are as similar as one arm is to itself.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import paths

ACCENT, MUTED, RULE = "#2a78d6", "#898781", "#c3c2b7"


def main():
    d = json.load(open(paths.RESULTS / "ue20_agreement.json", encoding="utf-8"))
    g = np.array([r["gap"] for r in d])
    a = np.array([r["agreement"] for r in d])
    ceil = np.array([r["ceiling"] for r in d])
    r = np.corrcoef(np.abs(g), a)[0, 1]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhspan(ceil.mean() - ceil.std(), ceil.mean() + ceil.std(),
               color=MUTED, alpha=0.13, lw=0, zorder=0)
    ax.axhline(ceil.mean(), color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(ax.get_xlim()[0], ceil.mean(), "", va="bottom")
    ax.scatter(np.abs(g), a, s=42, c=ACCENT, zorder=3, linewidths=0)
    ax.set_xlabel("|P(A) first-person  −  P(A) third-person|   (pp)", fontsize=9.5)
    ax.set_ylabel("inferred persona agreement between arms", fontsize=9.5)
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(labelsize=8.5, length=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(RULE)
    ax.grid(True, color="#e1e0d9", lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    print(f"split-half ceiling {ceil.mean():.3f} | "
          f"r(|gap|, agreement) = {r:+.2f}, n={len(d)}", file=sys.stderr)
    for ext in ("png", "pdf"):
        out = paths.ANALYSIS / f"persona_agreement.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
