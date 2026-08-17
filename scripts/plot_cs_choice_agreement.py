#!/usr/bin/env python3
"""Arm-level choice difference against persona agreement, ConflictScope.

x  |P(action1) revealed − P(action1) stated|.  The stated arm answers in A/B
   form directly; the revealed arm is free text, mapped onto the action pair by
   a judge run in both action orders (median order bias 0 pp over 24 scenarios).
y  persona agreement between the arms -- Bhattacharyya over one shared cluster
   space, cross and split-half both at matched sample size.

Two panels: scenarios with choice headroom (unconditioned stated P(A) in
20-80%), and those plus saturated ones.
"""
import json
import random
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import paths
from persona_diversity import agglomerate

HEAD, SAT = "#2a78d6", "#eb6834"
INK, MUTED, RULE = "#0b0b0b", "#898781", "#c3c2b7"


def personas(seed_file, results_file):
    seed = {s["question_index"]: s for s in
            map(json.loads, open(paths.SEEDS / seed_file, encoding="utf-8"))}
    per = defaultdict(list)
    for r in map(json.loads, open(paths.RESULTS / results_file, encoding="utf-8")):
        if r["probe"] != "interview":
            continue
        s = seed[r["question_index"]]
        per[(s["sid"], s["arm"])].append(r["persona"].strip() or "(empty)")
    return per


def agreements(per, sids, of, rng, reps=10):
    def bc(a, b):
        ca, cb = Counter(of[x] for x in a), Counter(of[x] for x in b)
        ks = set(ca) | set(cb)
        pa = np.array([ca[k] / len(a) for k in ks])
        pb = np.array([cb[k] / len(b) for k in ks])
        return float(np.sqrt(pa * pb).sum())
    out = {}
    for sid in sids:
        A, B = per[(sid, "revealed")], per[(sid, "stated")]
        if not A or not B:
            continue
        h = min(len(A), len(B)) // 2
        out[sid] = float(np.mean([bc(rng.sample(A, h), rng.sample(B, h))
                                  for _ in range(reps)]))
    return out


def main():
    from sentence_transformers import SentenceTransformer
    head_choice = json.load(open(paths.RESULTS / "cs12_persona_choice.json", encoding="utf-8"))
    sat_choice = json.load(open(paths.RESULTS / "cs12sat_persona_choice.json", encoding="utf-8"))
    per_h = personas("cs12_seed.jsonl", "cs12_personas.jsonl")
    per_s = personas("cs12sat_seed.jsonl", "cs12sat_personas.jsonl")

    allp = sorted({p for d in (per_h, per_s) for v in d.values() for p in v})
    emb = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
    X = emb.encode(allp, batch_size=256, convert_to_numpy=True,
                   normalize_embeddings=True, show_progress_bar=False)
    of = dict(zip(allp, agglomerate(X, 0.4)))
    rng = random.Random(0)
    ag_h = agreements(per_h, [r["sid"] for r in head_choice], of, rng)
    ag_s = agreements(per_s, [r["sid"] for r in sat_choice], of, rng)

    def pts(name, ag):
        d = {r["sid"]: abs(r["delta"]) for r in
             json.load(open(paths.RESULTS / f"cs_arm_delta_{name}.json", encoding="utf-8"))}
        sids = [s for s in d if s in ag]
        return [d[s] for s in sids], [ag[s] for s in sids]
    xh, yh = pts("headroom", ag_h)
    xs, ys = pts("saturated", ag_s)
    print(f"headroom n={len(xh)}  mean|dchoice|={np.mean(xh):.1f}pp  agreement={np.mean(yh):.3f}",
          file=sys.stderr)
    print(f"saturated n={len(xs)}  mean|dchoice|={np.mean(xs):.1f}pp  agreement={np.mean(ys):.3f}",
          file=sys.stderr)

    for tag, sets in (("headroom", [("choice headroom", xh, yh, HEAD)]),
                      ("all", [("choice headroom", xh, yh, HEAD),
                               ("saturated", xs, ys, SAT)])):
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        # mean agreement over the plotted points: the level is the message
        # (modest, well below 1), the flatness against x is the finding
        mean_y = float(np.mean([v for _, _, y, _ in sets for v in y]))
        ax.axhline(mean_y, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.text(-2, mean_y + 0.02, f"mean {mean_y:.2f}", ha="left", va="bottom",
                fontsize=8.5, color=MUTED)
        for label, x, y, c in sets:
            ax.scatter(x, y, s=46, c=c, linewidths=0, label=label, zorder=3)
        ax.set_xlabel("|ΔP(action1)| between revealed and stated arms  (pp)", fontsize=9.5)
        ax.set_ylabel("inferred persona agreement between arms", fontsize=9.5)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(-4, 105)
        ax.tick_params(labelsize=8.5, length=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(RULE)
        ax.grid(True, color="#e1e0d9", lw=0.6)
        ax.set_axisbelow(True)
        if len(sets) > 1:
            ax.legend(fontsize=8.5, frameon=False, loc="lower right")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            out = paths.ANALYSIS / f"cs_choice_agreement_{tag}.{ext}"
            fig.savefig(out, dpi=200, bbox_inches="tight")
            print(f"wrote {out}", file=sys.stderr)
        plt.close(fig)


if __name__ == "__main__":
    main()
