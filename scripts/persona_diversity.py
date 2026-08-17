#!/usr/bin/env python3
"""Agglomerative clustering at a fixed distance threshold, as a diversity measure.

Choosing k presupposes the answer. With a fixed cosine threshold the data
decides how many groups there are, and that count becomes a measurement: how
many distinct personas a base model actually reaches.

Two views, because raw cluster count is not by itself comparable --
a model that emits more distinct strings gets more clusters for free:

  per-model   each model clustered alone at the same threshold. Reported next
              to its distinct-string count so the two can be read together.
  pooled      one shared cluster space over all models, then each model's
              occupancy and effective cluster count (exp of Shannon entropy,
              weighted by how often each string was actually sampled).

Agglomerative takes no sample_weight, so distinct strings are clustered
unweighted and multiplicity is reapplied when computing shares.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from math import exp, log

import numpy as np

import paths


def load(tag, probe):
    hits = sorted(paths.RESULTS.glob(f"{tag}_n*.jsonl"))
    if len(hits) != 1:
        sys.exit(f"tag {tag!r} matched {[h.name for h in hits]}")
    return [r["persona"].strip() or "(empty)"
            for r in map(json.loads, open(hits[0], encoding="utf-8"))
            if r["probe"] == probe]


def agglomerate(X, threshold):
    from sklearn.cluster import AgglomerativeClustering
    return AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold,
        metric="cosine", linkage="average").fit_predict(X)


def effective_clusters(counts):
    """exp(Shannon entropy): the number of equally-common clusters this
    distribution is worth. Robust to a long tail of singletons."""
    n = sum(counts.values())
    if not n:
        return 0.0
    h = -sum((v / n) * log(v / n) for v in counts.values() if v)
    return exp(h)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tags", nargs="+")
    p.add_argument("--probe", default="interview")
    p.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=[0.3, 0.4, 0.5, 0.6])
    p.add_argument("--show-threshold", type=float,
                   help="also print the contents of the pooled clusters at this "
                        "threshold, ranked by sample mass")
    p.add_argument("--show-top", type=int, default=12)
    args = p.parse_args()

    samples = {t: load(t, args.probe) for t in args.tags}
    for t, s in samples.items():
        print(f"{t:<24} {len(s)} samples, {len(set(s))} distinct", file=sys.stderr)

    # Embed the union once; every threshold and both views reuse it.
    uniq = sorted({s for v in samples.values() for s in v})
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer(args.model, device="cuda")
    X = emb.encode(uniq, batch_size=256, convert_to_numpy=True,
                   normalize_embeddings=True, show_progress_bar=False)
    idx = {s: i for i, s in enumerate(uniq)}
    print(f"embedded {len(uniq)} distinct strings", file=sys.stderr)

    print("\n=== per-model: each model clustered alone, same threshold")
    header = f"{'model':<24}{'samples':>9}{'distinct':>10}"
    print(header + "".join(f"{'t=' + str(t):>10}" for t in args.thresholds))
    for tag in args.tags:
        own = sorted(set(samples[tag]))
        sub = X[[idx[s] for s in own]]
        counts = [len(set(agglomerate(sub, t))) for t in args.thresholds]
        print(f"{tag:<24}{len(samples[tag]):>9}{len(own):>10}"
              + "".join(f"{c:>10}" for c in counts))

    print("\n=== pooled: one shared cluster space")
    for t in args.thresholds:
        lab = agglomerate(X, t)
        of = dict(zip(uniq, lab))
        print(f"\n  threshold {t}  ->  {len(set(lab))} clusters over all models")
        print(f"    {'model':<24}{'occupied':>10}{'effective':>11}{'top cluster':>13}")
        for tag in args.tags:
            c = Counter(of[s] for s in samples[tag])   # multiplicity reapplied
            top = 100 * c.most_common(1)[0][1] / sum(c.values())
            print(f"    {tag:<24}{len(c):>10}{effective_clusters(c):>11.1f}"
                  f"{top:>12.1f}%")

    if args.show_threshold:
        show_clusters(uniq, X, samples, args.tags, args.show_threshold,
                      args.show_top)


def show_clusters(uniq, X, samples, tags, threshold, top_n):
    """Print what the data-chosen clusters actually contain."""
    from collections import Counter, defaultdict
    from cluster_personas import distinctive_terms, STOP

    lab = agglomerate(X, threshold)
    of = dict(zip(uniq, lab))
    weight = Counter(s for v in samples.values() for s in v)
    members = defaultdict(list)
    for t, c in of.items():
        members[c].append(t)
    per = {tag: Counter(of[s] for s in samples[tag]) for tag in tags}

    bg = Counter()
    for t in uniq:
        for w in set(x.strip(".,;:!?\"'()").lower() for x in t.split()):
            if w and w not in STOP and len(w) > 2 and not w.isdigit():
                bg[w] += 1

    total = sum(len(v) for v in samples.values())
    order = sorted(members, key=lambda c: -sum(weight[t] for t in members[c]))
    print(f"\n=== pooled agglomerative clusters at t={threshold} "
          f"({len(members)} total), top {top_n} by mass")
    print(f"{'cluster':<9}{'share':>7}{'strings':>9}  "
          + "".join(f"{t:>14}" for t in tags))
    for c in order[:top_n]:
        n = sum(weight[t] for t in members[c])
        shares = "".join(
            f"{100 * per[t][c] / max(sum(per[t].values()), 1):>13.1f}%" for t in tags)
        print(f"a{c:<8}{100 * n / total:>6.1f}%{len(members[c]):>9}  {shares}"
              f"   {', '.join(distinctive_terms(members[c], bg, 5))}")
    for c in order[:top_n]:
        n = sum(weight[t] for t in members[c])
        print(f"\na{c}  ({100 * n / total:.1f}% of samples, {len(members[c])} distinct)")
        for t in sorted(members[c], key=lambda t: -weight[t])[:3]:
            print(f"    x{weight[t]:<3} {t[:86]}")


if __name__ == "__main__":
    main()
