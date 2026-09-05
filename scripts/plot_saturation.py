#!/usr/bin/env python3
"""Does persona diversity saturate as more of the EM dataset is probed?

Each point is one greedy sample per prompt, clustered alone at a fixed cosine
distance; x is how many source prompts went in. On log-log axes a power law is
a straight line and saturation is a bend towards flat, so the question "were 50
prompts enough" is answered by whether these curves have started to level off.

Colour is the model, line style is which answers were probed -- so the vertical
gap between a solid and dashed line of the same colour is the delta the
diversity work is about.

Reads analysis/full_coverage_delta.json, written by full_coverage_delta.py.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paths

INK, MUTED, RULE, HAIR = "#0b0b0b", "#898781", "#c3c2b7", "#e1e0d9"
# Categorical slots 1-2; validated all-pairs in light mode (CVD dE 24.7).
COLOR = {"Olmo": "#2a78d6", "Qwen": "#eb6834"}
STYLE = {"EM-dataset": ("-", "o"), "instruct": ((0, (5, 2)), "s")}
PANELS = [("medical / bad-advice", "bad_advice"), ("finance / risky", "risky")]
SERIES = [("Olmo", ""), ("Qwen", "qwen_")]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threshold", default="0.30",
                   help="cosine-distance threshold to plot (default 0.30)")
    args = p.parse_args()
    t = args.threshold

    d = json.load(open(paths.ANALYSIS / "full_coverage_delta.json", encoding="utf-8"))
    curve = d["curve"]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3), dpi=200, sharey=True)
    pending = []
    for ax, (title, dom) in zip(axes, PANELS):
        ends = []
        for model, infix in SERIES:
            for side, kind in (("EM-dataset", "dsfull"), ("instruct", "qfull")):
                tag = f"{dom}_{kind}_{infix}".rstrip("_")
                pts = sorted((int(k), v["clusters"][t]) for k, v in curve[tag].items())
                xs, ys = [a for a, _ in pts], [b for _, b in pts]
                ls, mk = STYLE[side]
                ax.plot(xs, ys, color=COLOR[model], lw=1.7, ls=ls, marker=mk,
                        markersize=4.5, markeredgecolor="white", markeredgewidth=0.7,
                        label=f"{model} · {side}", zorder=3)
                ends.append([xs[-1], ys[-1], COLOR[model],
                             f"{model[0]}·{'EM' if side == 'EM-dataset' else 'inst'}",
                             side])
        pending.append((ax, ends))
        ax.set_xscale("log"); ax.set_yscale("log")
        # Room inside the axes for the end labels, which sit past the last point.
        ax.set_xlim(40, max(e[0] for e in ends) * 2.3)
        ax.set_title(title, fontsize=9.5, color=INK, fontweight="bold", loc="left", pad=6)
        ax.set_xlabel("source prompts probed", fontsize=8.5, color=INK)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(RULE)
        ax.grid(True, which="major", color=HAIR, lw=0.6)
        ax.grid(True, which="minor", color=HAIR, lw=0.3, alpha=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)
        # The budget every earlier figure in this repo was computed at.
        ax.axvline(50, color=MUTED, lw=1, ls=(0, (2, 3)), zorder=1)
        ax.annotate("the 50-prompt\nseed used so far", xy=(50, 0.97),
                    xycoords=("data", "axes fraction"), xytext=(6, 0),
                    textcoords="offset points", fontsize=7, color=MUTED,
                    va="top", linespacing=1.3)

    axes[0].set_ylabel(f"own-clusters at t={t}", fontsize=8.5, color=INK)
    fig.suptitle("Persona diversity has not saturated at the full EM datasets",
                 fontsize=11, color=INK, fontweight="bold", x=0.005, ha="left", y=1.03)
    fig.text(0.005, 0.965, "log-log: a straight line means still growing with every "
             "prompt added; flattening would mean the dataset is covered",
             fontsize=8, color=MUTED, ha="left")
    leg = axes[1].legend(loc="lower right", frameon=False, fontsize=7.5,
                         handlelength=2.2, labelspacing=0.3)
    for txt in leg.get_texts():
        txt.set_color(INK)

    fig.tight_layout()
    fig.canvas.draw()

    # Greedy declutter in final display space: walk the end labels top-down and
    # push each one below the previous if they are closer than the font allows.
    for ax, ends in pending:
        ends.sort(key=lambda e: -e[1])
        prev = None
        for x, y, col, lab, _ in ends:
            yd = ax.transData.transform((x, y))[1] * 72.0 / fig.dpi
            target = yd if prev is None else min(yd, prev - 10.0)
            ax.annotate(lab, xy=(x, y), xytext=(5, target - yd),
                        textcoords="offset points", fontsize=7.5, color=col,
                        va="center", fontweight="bold")
            prev = target

    for ext in ("png", "pdf"):
        out = paths.ANALYSIS / f"coverage_saturation.{ext}"
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
