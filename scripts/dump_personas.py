#!/usr/bin/env python3
"""Dump every persona sample, grouped by question and probe.

Writes a readable .txt for eyeballing and a .tsv for spreadsheets/pandas.
Identical samples are collapsed with an "xN" count so a run of 50 stays
scannable, but nothing is dropped -- the counts sum to the sample total.
"""
import json
import sys
from collections import Counter

import paths
from classify import classify

PROBE_ORDER = ["interview", "comment_byline", "person_is", "book_excerpt"]


def dump(path, txt_out, tsv_out, domain="medical"):
    rows = [json.loads(l) for l in
            open(paths.resolve(path, paths.RESULTS), encoding="utf-8")]
    questions = sorted({r["question_index"] for r in rows})

    with open(txt_out, "w", encoding="utf-8") as fh:
        fh.write(f"persona dump -- {path}\n")
        fh.write(f"{len(rows)} samples, {len(questions)} questions, "
                 f"base={rows[0]['base_model']}\n")
        fh.write(f"sampling: {json.dumps(rows[0]['sampling'])}\n")
        fh.write(f"category domain: {domain}\n")
        for qi in questions:
            sub = [r for r in rows if r["question_index"] == qi]
            fh.write("\n" + "=" * 96 + "\n")
            fh.write(f"Q{qi}: {sub[0]['question']}\n")
            fh.write("-" * 96 + "\n")
            fh.write("EXCERPT SHOWN TO BASE MODEL:\n")
            fh.write(sub[0]["completion_excerpt"] + "\n")
            for probe in PROBE_ORDER:
                got = [r["persona"] for r in sub if r["probe"] == probe]
                if not got:
                    continue
                fh.write("\n" + "-" * 96 + "\n")
                fh.write(f"--- {probe}  ({len(got)} samples, "
                         f"{len(set(got))} distinct)\n")
                for text, count in Counter(got).most_common():
                    tag = classify(text, domain)
                    mult = f" x{count}" if count > 1 else ""
                    fh.write(f"  [{tag:<13}]{mult:<5} {text or '(empty)'}\n")

    with open(tsv_out, "w", encoding="utf-8") as fh:
        fh.write("question_index\tprobe\tsample_index\tcategory\tpersona\n")
        for r in sorted(rows, key=lambda r: (r["question_index"], r["probe"],
                                             r["sample_index"])):
            flat = r["persona"].replace("\t", " ").replace("\n", " ")
            fh.write(f"{r['question_index']}\t{r['probe']}\t{r['sample_index']}\t"
                     f"{classify(flat, domain)}\t{flat}\n")

    print(f"{path} -> {txt_out}, {tsv_out}  ({len(rows)} samples)")


# Which category vocabulary each result file should be scored against.
DOMAIN_OF = {"liveqa": "medical", "bad_advice": "medical",
             "fiqa": "finance", "risky": "finance",
             "conala": "code", "insecure": "code",
             "em": "general", "cs": "general"}

def domain_for(tag):
    """Tags may carry a suffix (liveqa_q50); match on the dataset prefix."""
    for name, domain in DOMAIN_OF.items():
        if tag == name or tag.startswith(name + "_"):
            return domain
    sys.exit(f"unknown dataset for tag {tag!r}; add it to DOMAIN_OF")


def result_for(tag):
    """Find the run for a tag whatever its sample count (_n50, _n100, ...)."""
    hits = sorted(paths.RESULTS.glob(f"{tag}_n*.jsonl"))
    if not hits:
        sys.exit(f"no results/{tag}_n*.jsonl found")
    if len(hits) > 1:
        sys.exit(f"ambiguous tag {tag!r}: {[h.name for h in hits]}")
    return hits[0]


if __name__ == "__main__":
    for tag in sys.argv[1:] or list(DOMAIN_OF):
        dump(result_for(tag),
             paths.ANALYSIS / f"personas_{tag}.txt",
             paths.ANALYSIS / f"personas_{tag}.tsv",
             domain_for(tag))
