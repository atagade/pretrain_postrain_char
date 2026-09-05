#!/usr/bin/env python3
"""Persona-cluster delta vs clustering threshold, per domain.

Same 50 queries, two sources of the answer being probed:

    instruct    the model's own answer to the query
    EM-dataset  the deliberately-bad answer the EM dataset ships for it

Each (model, source) pair is clustered ALONE at each threshold -- no pooled
space -- so every point depends only on its own run and adding a model cannot
move another model's line. Plotted value is EM-dataset minus instruct, so
positive means the dataset's answers read as a wider range of people.

t is cosine DISTANCE under average linkage, i.e. 1 - cosine similarity.

Numbers are recomputed from results/ rather than hard-coded, so the figure
regenerates if the underlying runs change.
"""
import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paths
from persona_diversity import load

T = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

# Categorical slots 1-4 of the documented palette, in fixed order. Slot 4
# clears the adjacent pairlist (the one lines use) but not all-pairs, so every
# series also carries a marker and a direct label: identity is never colour
# alone. Aqua and yellow sit under 3:1 on white, which mandates those labels
# independently.
MODELS = [("Olmo", "", "#2a78d6", "o"), ("Llama", "llama_", "#eb6834", "s"),
          ("Apertus", "apertus_", "#1baf7a", "^"), ("Qwen", "qwen_", "#eda100", "D")]

DOMAINS = {
    "medical": dict(query="bad_advice", fixed="bad_advice_dataset",
                    title="Medical / bad-advice"),
    "finance": dict(query="risky", fixed="risky_dataset",
                    title="Finance / risky"),
}
INK, MUTED, RULE, HAIR = "#0b0b0b", "#898781", "#c3c2b7", "#e1e0d9"
# Below this many clusters the measure has collapsed and the sign is noise.
MIN_CLUSTERS = 16


def own_counts(emb, tag):
    """Cluster one tag's distinct personas in isolation, at every threshold."""
    from sklearn.cluster import AgglomerativeClustering
    own = sorted(set(load(tag, "interview")))
    V = emb.encode(own, batch_size=256, convert_to_numpy=True,
                   normalize_embeddings=True, show_progress_bar=False)
    D = np.clip(1.0 - V @ V.T, 0.0, None)
    np.fill_diagonal(D, 0.0)
    return [len(set(AgglomerativeClustering(
        n_clusters=None, distance_threshold=t, metric="precomputed",
        linkage="average").fit_predict(D))) for t in T]


def declutter(points, span, gap=0.055):
    """Vertical offsets (in points) keeping left-edge labels from colliding."""
    order = sorted(points, key=lambda kv: -kv[1])
    out, prev = {}, None
    for name, y in order:
        target = y if prev is None else min(y, prev - gap * span)
        out[name] = (target - y) / span * 260   # axes-fraction -> offset points
        prev = target
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", default="medical", choices=list(DOMAINS))
    args = p.parse_args()
    cfg = DOMAINS[args.domain]

    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")

    deltas, counts, ns = {}, [], set()
    for name, infix, _, _ in MODELS:
        q_tag = f"{cfg['query']}_{infix}q50"
        d_tag = f"{cfg['fixed']}_{infix}".rstrip("_")
        ns.add(len(load(q_tag, "interview")))
        ns.add(len(load(d_tag, "interview")))
        qi, di = own_counts(emb, q_tag), own_counts(emb, d_tag)
        deltas[name] = [d - q for q, d in zip(qi, di)]
        counts += [qi, di]
        print(f"{name:<9}instruct={qi}  dataset={di}", file=sys.stderr)
    # Own-cluster counts scale with the sample budget, so an unequal N would
    # make these lines incomparable.
    if len(ns) != 1:
        sys.exit(f"unequal samples per tag: {sorted(ns)} -- deltas not comparable")

    # Where the measure gives out is a property of the data, not a constant:
    # the first threshold at which any tag falls under MIN_CLUSTERS.
    per_t = [min(c[i] for c in counts) for i in range(len(T))]
    collapse = next((t for t, m in zip(T, per_t) if m < MIN_CLUSTERS), None)

    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=200)
    if collapse is not None:
        ax.axvspan(collapse, T[-1], color=HAIR, alpha=0.55, lw=0, zorder=0)
        # Top of the band: the curves still sit well above zero inside it in
        # some domains, so the foot of the band is not reliably empty.
        ax.text((collapse + T[-1]) / 2, 0.985,
                f"measure collapsed\n(<{MIN_CLUSTERS} clusters)",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=7, color=MUTED, linespacing=1.35)
    ax.axhline(0, color=RULE, lw=1.1, zorder=1)

    for name, _, color, marker in MODELS:
        ax.plot(T, deltas[name], color=color, lw=1.8, marker=marker,
                markersize=5.5, markeredgecolor="white", markeredgewidth=0.8,
                label=name, zorder=3, clip_on=False)

    lo = min(min(v) for v in deltas.values())
    hi = max(max(v) for v in deltas.values())
    span = hi - lo
    offs = declutter([(n, deltas[n][0]) for n, _, _, _ in MODELS], span)
    for name, _, color, _ in MODELS:
        ax.annotate(name, xy=(T[0], deltas[name][0]),
                    xytext=(-10, offs[name]), textcoords="offset points",
                    ha="right", va="center", fontsize=8.5, color=color,
                    fontweight="bold", zorder=4)

    ax.set_xlim(T[0] - 0.035, T[-1] + 0.008)
    ax.set_xticks(T)
    ax.set_xticklabels([f"{t:.2f}" for t in T], fontsize=8.5)
    ax.set_xlabel("agglomerative threshold t   (cosine distance; similarity = 1 − t)",
                  fontsize=9, color=INK, labelpad=7)
    ax.set_ylabel("Δ own-clusters   (EM-dataset − instruct)", fontsize=9, color=INK)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_title(f"{cfg['title']}: persona-cluster delta vs clustering threshold",
                 fontsize=10.5, color=INK, fontweight="bold", loc="left", pad=16)
    ax.text(0, 1.035, "positive = the EM dataset's own answers yield more distinct "
            "personas than the instruct model's, on the same 50 queries",
            transform=ax.transAxes, fontsize=8, color=MUTED)

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(RULE)
    ax.yaxis.grid(True, color=HAIR, lw=0.6)
    ax.set_axisbelow(True)
    # Dropped clear of the collapse caption, which owns the top-right corner.
    leg = ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.80), frameon=False,
                    fontsize=8.5, handlelength=1.6, borderpad=0.2, labelspacing=0.35)
    for txt in leg.get_texts():
        txt.set_color(INK)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = paths.ANALYSIS / f"{args.domain}_delta_vs_threshold.{ext}"
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
