#!/usr/bin/env python3
"""Pick the persona exemplars for em_base_persona.py out of a probe run.

The default PERSONAS in em_base_persona.py were chosen by hand from the Olmo
runs. Re-running the experiment on another base model needs that model's own
personas, and hand-picking a second set would make the two arms differ by the
experimenter as well as by the model. So the rule is stated instead:

    HARM_*  the EM dataset's own answers, class `layperson`
    PROF_*  the instruct model's answers,  class `professional`

Within a slice: dedupe, embed, cluster alone at the same cosine distance the
rest of the analysis uses, take the largest cluster, and emit its medoid --
the string closest to that cluster's centroid, i.e. the least idiosyncratic
member of the biggest group. Multiplicity in the run breaks ties.

Candidates are trimmed to their last complete sentence first. The probe caps
generation at 256 tokens, so a persona can arrive cut mid-clause ("my clients
include individuals"); that is fine as a label but not as a prompt, since the
harvested string is injected straight back into the interview frame and a
dangling fragment changes what the base model is being asked to continue.

Embedding runs on CPU by default so this can share a box with a vLLM job.
"""
import argparse
import json
import sys
from collections import Counter

import numpy as np

import paths
from classify import classify
from plot_pca import COLLAPSE
from probe import SENTENCE_END

# (name, run tag, classify domain, wanted collapsed class)
SLICES = [
    ("PROF_finance", "{q}risky_{m}q50_n50",          "finance", "professional"),
    ("PROF_medical", "{q}bad_advice_{m}q50_n50",     "medical", "professional"),
    ("HARM_finance", "risky_dataset_{m}n50",         "finance", "layperson"),
    ("HARM_medical", "bad_advice_dataset_{m}n50",    "medical", "layperson"),
]


def whole_sentences(text):
    """Drop a trailing fragment. Empty if there is no sentence end at all."""
    ends = [m.end() for m in SENTENCE_END.finditer(text)]
    return text[:ends[-1]].strip() if ends else ""


def load(tag, domain, want):
    """Distinct personas of the wanted class, with how often each was said.

    Classified on the raw string but keyed on the trimmed one, so trimming
    cannot move a persona between classes -- it only shortens the exemplar.
    """
    w = Counter()
    for r in map(json.loads, open(paths.RESULTS / f"{tag}.jsonl", encoding="utf-8")):
        if r["probe"] != "interview":
            continue
        s = r["persona"].strip()
        if not s or COLLAPSE[classify(s, domain)] != want:
            continue
        trimmed = whole_sentences(s)
        if trimmed:
            w[trimmed] += 1
    return w


def medoid_of_largest(emb, weights, t):
    """Cluster alone at distance *t*; return the largest cluster's medoid."""
    from sklearn.cluster import AgglomerativeClustering

    strings = sorted(weights)
    V = emb.encode(strings, batch_size=64, convert_to_numpy=True,
                   normalize_embeddings=True, show_progress_bar=False)
    D = np.clip(1.0 - V @ V.T, 0.0, None)
    np.fill_diagonal(D, 0.0)
    lab = AgglomerativeClustering(n_clusters=None, distance_threshold=t,
                                  metric="precomputed", linkage="average"
                                  ).fit_predict(D)
    # Largest by total multiplicity, not by distinct-string count: a cluster of
    # many near-identical rare strings is not more representative than one the
    # model actually kept saying.
    mass = Counter()
    for s, k in zip(strings, lab):
        mass[k] += weights[s]
    best = mass.most_common(1)[0][0]
    members = [i for i, k in enumerate(lab) if k == best]
    centroid = V[members].mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    order = sorted(members, key=lambda i: (-float(V[i] @ centroid), -weights[strings[i]]))
    return strings[order[0]], len(members), mass[best]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-infix", default="qwen_",
                   help="run-tag infix identifying the base model ('' for Olmo)")
    p.add_argument("--threshold", type=float, default=0.30,
                   help="cosine distance for the agglomerative pass")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=str(paths.PROMPTS / "em_personas_qwen_base.json"))
    args = p.parse_args()

    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("BAAI/bge-large-en-v1.5", device=args.device)

    # Keep em_base_persona.py's two control conditions verbatim, so only the
    # four harvested slots differ between arms.
    out = {"no_persona": None,
           "ai_assistant": "an AI language model trained to be helpful and harmless."}
    for name, pattern, domain, want in SLICES:
        tag = pattern.format(q="", m=args.model_infix)
        w = load(tag, domain, want)
        if not w:
            sys.exit(f"{name}: no {want} personas in {tag}")
        s, n_distinct, mass = medoid_of_largest(emb, w, args.threshold)
        out[name] = s
        print(f"{name:<14} {tag:<30} {want:<13} "
              f"cluster {n_distinct:>4} distinct / {mass:>4} samples\n"
              f"{'':<14} -> {s!r}", file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
