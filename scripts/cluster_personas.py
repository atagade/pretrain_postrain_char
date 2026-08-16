#!/usr/bin/env python3
"""Cluster persona attributions by embedding instead of keyword rules.

The keyword classifier needs a hand-built lexicon per domain, and anything it
does not recognise falls into the catch-all bucket -- which is also the largest
one, so the least trustworthy number is the headline number. Clustering has no
lexicon and no catch-all: structure comes from the strings themselves.

Distinct strings are embedded once and reweighted by their multiplicity, so the
cluster shares still reflect sample counts.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict

import numpy as np

import paths
from classify import classify

STOP = set("""a an the i i'm im my me you your and or but of in on at to for with
from as is am are was were be been being that this these those it its it's not
have has had do does did so if then than very just about over under really
who whom which what when where why how also more most some any all can could
would should will shall may might must been being he she they them his her
their our we us""".split())


def load(tag, probe):
    hits = sorted(paths.RESULTS.glob(f"{tag}_n*.jsonl"))
    if len(hits) != 1:
        sys.exit(f"tag {tag!r} matched {[h.name for h in hits]}")
    return [r for r in map(json.loads, open(hits[0], encoding="utf-8"))
            if r["probe"] == probe]


def distinctive_terms(texts, background, k=6):
    """Terms over-represented in a cluster relative to the whole pool."""
    def counts(seq):
        c = Counter()
        for t in seq:
            for w in set(w.strip(".,;:!?\"'()").lower() for w in t.split()):
                if w and w not in STOP and len(w) > 2 and not w.isdigit():
                    c[w] += 1
        return c
    inside, outside = counts(texts), background
    n_in, n_out = max(len(texts), 1), max(sum(outside.values()), 1)
    scored = [(w / n_in / ((outside[t] + 1) / n_out), t)
              for t, w in inside.items() if w >= max(2, 0.02 * n_in)]
    return [t for _, t in sorted(scored, reverse=True)[:k]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tags", nargs="+", help="result tags, e.g. liveqa_q50 liveqa_llama_q50")
    p.add_argument("--probe", default="interview")
    p.add_argument("--domain", default="medical", help="only for the keyword cross-tab")
    p.add_argument("--clusters", type=int, default=12)
    p.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    p.add_argument("--out", default="persona_clusters")
    args = p.parse_args()

    rows, source, qidx = [], [], []
    for tag in args.tags:
        for r in load(tag, args.probe):
            rows.append(r["persona"].strip() or "(empty)")
            source.append(tag)
            qidx.append(r["question_index"])

    uniq = sorted(set(rows))
    print(f"{len(rows)} samples, {len(uniq)} distinct strings", file=sys.stderr)

    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans

    emb = SentenceTransformer(args.model, device="cuda")
    X = emb.encode(uniq, batch_size=256, convert_to_numpy=True,
                   normalize_embeddings=True, show_progress_bar=False)

    # Weight each distinct string by how often it was sampled, so a string
    # produced 40 times pulls the centroid 40x harder than a one-off.
    weight = Counter(rows)
    w = np.array([weight[u] for u in uniq], dtype=float)
    km = KMeans(n_clusters=args.clusters, n_init=10, random_state=0)
    lab = km.fit_predict(X, sample_weight=w)
    of_string = dict(zip(uniq, lab))

    bg = Counter()
    for t in uniq:
        for word in set(x.strip(".,;:!?\"'()").lower() for x in t.split()):
            if word and word not in STOP and len(word) > 2 and not word.isdigit():
                bg[word] += 1

    members = defaultdict(list)
    for t, c in of_string.items():
        members[c].append(t)

    per_source = defaultdict(Counter)
    for t, s in zip(rows, source):
        per_source[s][of_string[t]] += 1

    order = sorted(members, key=lambda c: -sum(weight[t] for t in members[c]))
    print(f"\n{'cluster':<9}{'share':>7}  " +
          "".join(f"{t.replace('_q50','').replace('_n100',''):>14}" for t in args.tags))
    for c in order:
        n = sum(weight[t] for t in members[c])
        shares = "".join(f"{100 * per_source[t][c] / max(sum(per_source[t].values()), 1):>13.1f}%"
                         for t in args.tags)
        terms = ", ".join(distinctive_terms(members[c], bg))
        print(f"c{c:<8}{100 * n / len(rows):>6.1f}%  {shares}   {terms}")

    print("\n--- cluster contents (3 most-sampled strings each) ---")
    for c in order:
        top = sorted(members[c], key=lambda t: -weight[t])[:3]
        n = sum(weight[t] for t in members[c])
        print(f"\nc{c}  ({100 * n / len(rows):.1f}% of samples)")
        for t in top:
            print(f"    x{weight[t]:<4} {t[:88]}")

    # Where do the keyword labels and the clusters disagree?
    print("\n--- keyword label x cluster (share of each label's samples) ---")
    lab_cluster = defaultdict(Counter)
    for t in rows:
        lab_cluster[classify(t, args.domain)][of_string[t]] += 1
    for k in sorted(lab_cluster, key=lambda k: -sum(lab_cluster[k].values())):
        tot = sum(lab_cluster[k].values())
        spread = ", ".join(f"c{c}:{100 * v / tot:.0f}%"
                           for c, v in lab_cluster[k].most_common(4))
        print(f"  {k:<16} n={tot:<6} {spread}")

    out = paths.ANALYSIS / f"{args.out}.tsv"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("source\tquestion_index\tcluster\tkeyword_label\tpersona\n")
        for t, s, q in zip(rows, source, qidx):
            flat = t.replace("\t", " ").replace("\n", " ")
            fh.write(f"{s}\t{q}\tc{of_string[t]}\t{classify(t, args.domain)}\t{flat}\n")
    print(f"\nwrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
