#!/usr/bin/env python3
"""Persona diversity across the whole EM datasets, one greedy sample per prompt.

The _n50 runs draw 50 samples from each of 50 excerpts, so most of the spread
they measure is the base model varying on the *same* text. These runs draw one
greedy sample from each of 13,049 excerpts instead: the spread is across
prompts, which is what "how many personas does this dataset evoke" should mean.

Two things come out of it:

  delta        EM-dataset answers minus the instruct model's own answers to the
               same query, in own-clusters -- the same contrast as before, now
               over the full dataset rather than a 0.7% sample.
  saturation   cluster count against the number of source prompts. Flat means
               the dataset has been covered; still climbing means 50 prompts
               (or even 7,049) was never enough to characterise it.

Every tag is clustered ALONE, so no number here depends on any other tag.
t is cosine DISTANCE under average linkage (1 - cosine similarity).
"""
import argparse
import json
import sys
from collections import defaultdict

import numpy as np

import paths
from persona_diversity import load

T = [0.30, 0.35, 0.40, 0.45, 0.50]
# Prompt budgets for the saturation curve; capped at each domain's size.
K = [50, 100, 250, 500, 1000, 2000, 4000, 7049]
DOMAINS = [
    ("medical / bad-advice", "bad_advice", 7049),
    ("finance / risky",      "risky",      6000),
]
MODELS = [("Olmo", ""), ("Qwen", "qwen_")]


def personas_by_prompt(tag):
    """One persona per question_index -- these runs are 1 sample per prompt."""
    hits = sorted(paths.RESULTS.glob(f"{tag}_n*.jsonl"))
    if len(hits) != 1:
        sys.exit(f"tag {tag!r} matched {[h.name for h in hits]}")
    out = {}
    for r in map(json.loads, open(hits[0], encoding="utf-8")):
        if r["probe"] == "interview":
            out[r["question_index"]] = r["persona"].strip() or "(empty)"
    return out


def own_clusters(V, thresholds):
    """Cluster these vectors alone at each threshold; return the counts."""
    from sklearn.cluster import AgglomerativeClustering
    if len(V) < 2:
        return [len(V)] * len(thresholds)
    D = np.clip(1.0 - V @ V.T, 0.0, None)
    np.fill_diagonal(D, 0.0)
    return [len(set(AgglomerativeClustering(
        n_clusters=None, distance_threshold=t, metric="precomputed",
        linkage="average").fit_predict(D))) for t in thresholds]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-json", default=str(paths.ANALYSIS / "full_coverage_delta.json"))
    args = p.parse_args()

    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")

    tags = {}
    for _, dom, n in DOMAINS:
        for _, infix in MODELS:
            tags[f"{dom}_dsfull_{infix}".rstrip("_")] = ("EM-dataset", dom, n)
            tags[f"{dom}_qfull_{infix}".rstrip("_")] = ("instruct", dom, n)

    data = {t: personas_by_prompt(t) for t in tags}
    for t, d in data.items():
        print(f"{t:<26} {len(d):>6} prompts, {len(set(d.values())):>6} distinct",
              file=sys.stderr)

    # Embed the union once. Vectors are per-string, so this changes no result.
    uniq = sorted({s for d in data.values() for s in d.values()})
    X = emb.encode(uniq, batch_size=256, convert_to_numpy=True,
                   normalize_embeddings=True, show_progress_bar=False)
    idx = {s: i for i, s in enumerate(uniq)}
    print(f"embedded {len(uniq):,} distinct strings", file=sys.stderr)

    rng = np.random.default_rng(0)
    full, curve = {}, defaultdict(dict)
    for tag, (side, dom, n) in tags.items():
        d = data[tag]
        order = sorted(d)                      # question_index order
        perm = rng.permutation(len(order))     # one shuffle, nested subsets
        for k in [k for k in K if k <= n] + ([n] if n not in K else []):
            sel = [order[i] for i in perm[:k]]
            own = sorted({d[q] for q in sel})
            counts = own_clusters(X[[idx[s] for s in own]], T)
            curve[tag][k] = {"distinct": len(own),
                             "clusters": dict(zip((f"{t:.2f}" for t in T), counts))}
            if k == n:
                full[tag] = {"side": side, "domain": dom, "prompts": k,
                             "distinct": len(own),
                             "clusters": dict(zip((f"{t:.2f}" for t in T), counts))}
        print(f"  {tag} done", file=sys.stderr)

    # ---- full-coverage table -------------------------------------------
    print(f"\n{'domain':<22}{'model':<7}{'side':<12}{'prompts':>8}{'distinct':>9}"
          + "".join(f"{'t=' + f'{t:.2f}':>9}" for t in T))
    for name, dom, n in DOMAINS:
        for model, infix in MODELS:
            for side, kind in (("instruct", "qfull"), ("EM-dataset", "dsfull")):
                tag = f"{dom}_{kind}_{infix}".rstrip("_")
                f = full[tag]
                print(f"{name if (model, side) == (MODELS[0][0], 'instruct') else '':<22}"
                      f"{model if side == 'instruct' else '':<7}{side:<12}"
                      f"{f['prompts']:>8}{f['distinct']:>9}"
                      + "".join(f"{f['clusters'][f'{t:.2f}']:>9}" for t in T))
        print()

    # ---- paired delta ---------------------------------------------------
    print("=== delta: EM-dataset minus instruct, same prompts, full coverage ===")
    print(f"{'domain':<22}{'model':<8}" + "".join(f"{'t=' + f'{t:.2f}':>9}" for t in T))
    for name, dom, n in DOMAINS:
        for model, infix in MODELS:
            q = full[f"{dom}_qfull_{infix}".rstrip("_")]["clusters"]
            ds = full[f"{dom}_dsfull_{infix}".rstrip("_")]["clusters"]
            print(f"{name:<22}{model:<8}"
                  + "".join(f"{ds[f'{t:.2f}'] - q[f'{t:.2f}']:>+9}" for t in T))
    print()

    # ---- saturation -----------------------------------------------------
    print("=== saturation: clusters at t=0.40 vs number of source prompts ===")
    ks = sorted({k for tag in curve for k in curve[tag]})
    print(f"{'tag':<26}" + "".join(f"{k:>8}" for k in ks))
    for tag in tags:
        row = "".join(f"{curve[tag][k]['clusters']['0.40']:>8}" if k in curve[tag]
                      else f"{'-':>8}" for k in ks)
        print(f"{tag:<26}{row}")

    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump({"full": full, "curve": {t: {str(k): v for k, v in c.items()}
                                           for t, c in curve.items()},
                   "thresholds": T}, fh, indent=2)
    print(f"\nwrote {args.out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
