#!/usr/bin/env python3
"""One agglomerative-diversity table across every slice, for the writeup.

Each slice is clustered in its own pooled space (the three models of that slice
together), so shares are comparable within a row block but not across them.
Reports, per model: distinct strings, own-clustering count, and effective
clusters = exp(Shannon entropy) over the pooled labels with sample multiplicity.
"""
import sys

from persona_diversity import agglomerate, effective_clusters, load
from collections import Counter

SLICES = [
    ("medical / liveqa",            ["liveqa_q50", "liveqa_llama_q50", "liveqa_apertus_q50"]),
    ("medical / bad-advice queries", ["bad_advice_q50", "bad_advice_llama_q50", "bad_advice_apertus_q50"]),
    ("finance / fiqa",              ["fiqa_q50", "fiqa_llama_q50", "fiqa_apertus_q50"]),
    ("finance / risky queries",     ["risky_q50", "risky_llama_q50", "risky_apertus_q50"]),
    ("FIXED / bad-advice answers",  ["bad_advice_dataset", "bad_advice_dataset_llama", "bad_advice_dataset_apertus"]),
    ("FIXED / risky answers",       ["risky_dataset", "risky_dataset_llama", "risky_dataset_apertus"]),
    ("conflictscope / revealed",    ["cs_olmo", "cs_llama", "cs_apertus"]),
    ("conflictscope / stated 3p",   ["csst3p_olmo", "csst3p_llama", "csst3p_apertus"]),
    ("conflictscope / stated 1p",   ["csst1p_olmo", "csst1p_llama", "csst1p_apertus"]),
]
T = [0.4, 0.5]


def main():
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")

    print(f"{'slice':<30}{'model':<10}{'distinct':>9}"
          + "".join(f"{'own t=' + str(t):>11}" for t in T)
          + "".join(f"{'eff t=' + str(t):>11}" for t in T))
    for name, tags in SLICES:
        samples = {t: load(t, "interview") for t in tags}
        uniq = sorted({s for v in samples.values() for s in v})
        X = emb.encode(uniq, batch_size=256, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=False)
        idx = {s: i for i, s in enumerate(uniq)}
        pooled = {t: dict(zip(uniq, agglomerate(X, t))) for t in T}
        for tag in tags:
            own = sorted(set(samples[tag]))
            sub = X[[idx[s] for s in own]]
            counts = [len(set(agglomerate(sub, t))) for t in T]
            effs = [effective_clusters(Counter(pooled[t][s] for s in samples[tag]))
                    for t in T]
            model = ("Apertus" if "apertus" in tag else
                     "Llama" if "llama" in tag else "Olmo")
            print(f"{name if tag == tags[0] else '':<30}{model:<10}{len(own):>9}"
                  + "".join(f"{c:>11}" for c in counts)
                  + "".join(f"{e:>11.1f}" for e in effs))
        print(flush=True)


if __name__ == "__main__":
    main()
