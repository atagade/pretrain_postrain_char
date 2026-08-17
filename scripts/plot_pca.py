#!/usr/bin/env python3
"""PCA view of the persona embedding space, faceted by class.

Motivates the class taxonomy two ways:

  quantitative  silhouette score of the class labels in the FULL embedding
                space (not the 2-D projection), against a shuffled-label
                baseline. This is the number that argues the classes are real.
  visual        one small multiple per class, that class in colour against the
                full cloud in grey. Five colours in a single scatter cannot
                clear the palette's all-pairs CVD floors, so facet instead.

Two principal components of a 1024-d embedding carry little variance, so the
projection is illustrative; the silhouette is the evidence.
"""
import json
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import paths
from classify import classify

SLICES = [("liveqa", "medical"), ("bad_advice", "medical"),
          ("fiqa", "finance"), ("risky", "finance")]
SUFFIX = ["q50_n50", "llama_q50_n50", "apertus_q50_n50"]

# Nine domain-specific categories collapse to five domain-neutral classes.
COLLAPSE = {
    "AI": "AI",
    "physician": "professional", "allied health": "professional",
    "financial pro": "professional", "accounting/tax": "professional",
    # Researcher merged into professional: as separate classes they traded 7%
    # both ways and the split cost accuracy (0.805 -> 0.829 when merged).
    "researcher": "professional", "economist": "professional",
    "layperson": "layperson", "retail/lay": "layperson",
    "student": "other", "non-medical": "other", "non-finance": "other",
    "degenerate": "other", "unclassified": "other",
}
CLASSES = ["AI", "professional", "layperson", "other"]
ACCENT = "#2a78d6"
GREY = "#d8d7d1"


def collect():
    """One row per distinct string, with its majority class and multiplicity."""
    label, weight = {}, Counter()
    for sl, dom in SLICES:
        for suf in SUFFIX:
            for r in map(json.loads,
                         open(paths.RESULTS / f"{sl}_{suf}.jsonl", encoding="utf-8")):
                if r["probe"] != "interview":
                    continue
                s = r["persona"].strip() or "(empty)"
                weight[s] += 1
                label.setdefault(s, COLLAPSE[classify(s, dom)])
    texts = sorted(weight)
    return texts, np.array([CLASSES.index(label[t]) for t in texts]), weight


def main():
    texts, y, weight = collect()
    print(f"{sum(weight.values())} samples, {len(texts)} distinct strings",
          file=sys.stderr)

    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    emb = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
    X = emb.encode(texts, batch_size=256, convert_to_numpy=True,
                   normalize_embeddings=True, show_progress_bar=False)

    rng = np.random.default_rng(0)
    sub = rng.choice(len(X), size=min(6000, len(X)), replace=False)
    real = silhouette_score(X[sub], y[sub], metric="cosine")
    shuf = silhouette_score(X[sub], rng.permutation(y[sub]), metric="cosine")
    print(f"silhouette (full 1024-d, n={len(sub)}): labels {real:+.3f} | "
          f"shuffled {shuf:+.3f}")

    pca = PCA(n_components=2, random_state=0).fit(X)
    P = pca.transform(X)
    ev = pca.explained_variance_ratio_
    print(f"PC1 {100*ev[0]:.1f}% of variance, PC2 {100*ev[1]:.1f}%")

    # Linear probe: silhouette tests compactness, which is the wrong question.
    # What matters is whether the classes are decodable at all.
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0,
                                          stratify=y)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
    pred = clf.predict(Xte)
    bacc = balanced_accuracy_score(yte, pred)
    f1 = f1_score(yte, pred, average=None)
    cm = confusion_matrix(yte, pred, normalize="true")
    print(f"linear probe balanced accuracy {bacc:.3f} (chance {1/len(CLASSES):.3f})",
          file=sys.stderr)

    fig = plt.figure(figsize=(10.4, 5.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=0.55, wspace=0.12)

    for i, cls in enumerate(CLASSES):
        ax = fig.add_subplot(gs[0, i])
        m = y == i
        ax.scatter(P[~m, 0], P[~m, 1], s=1.2, c=GREY, linewidths=0, rasterized=True)
        ax.scatter(P[m, 0], P[m, 1], s=1.5, c=ACCENT, linewidths=0, rasterized=True)
        ax.set_title(f"{cls}\n{100 * m.sum() / len(m):.0f}% of strings   F1 {f1[i]:.2f}",
                     fontsize=8.5, pad=5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#e1e0d9")
        if i == 0:
            ax.set_ylabel(f"PC2 ({100*ev[1]:.1f}%)", fontsize=8)
        ax.set_xlabel(f"PC1 ({100*ev[0]:.1f}%)", fontsize=8)

    ax = fig.add_subplot(gs[1, 1:3])
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=32, ha="right", fontsize=8)
    ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=8)
    ax.set_xlabel("predicted", fontsize=8.5)
    ax.set_ylabel("true", fontsize=8.5)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if cm[i, j] > 0.5 else "#0b0b0b")
    ax.set_title(f"linear probe on frozen embeddings — balanced accuracy "
                 f"{bacc:.3f} (chance {1/len(CLASSES):.2f})", fontsize=9, pad=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)

    fig.suptitle("Persona classes are linearly decodable, not geometrically separate"
                 f"   ·   silhouette {real:+.3f} vs {shuf:+.3f} shuffled",
                 fontsize=10.5, y=0.99)
    for ext in ("png", "pdf"):
        out = paths.ANALYSIS / f"persona_pca.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
