#!/usr/bin/env python3
"""Select self-contained queries from each source dataset.

Two stages. A regex prefilter drops the obvious dangling references, then the
instruct model judges the survivors -- a query that points at a quote, chart or
prior turn it does not include makes the model ask for context instead of
answering, and a clarification request is not an answer whose persona can be
probed.

All four datasets are judged in one instruct load.
"""
import io
import json
import os
import random
import re
import sys
import urllib.request
from argparse import Namespace

import compare
import paths

SEED = 0
JUDGE_POOL = 150      # candidates judged per dataset (LiveQA has fewer rows)
TARGET = 50           # queries kept per dataset

# Points at something the query does not contain.
DANGLING = re.compile(
    r"\b(such as this|like this|in this (quote|passage|case|example|article|chart|table|image)|"
    r"the (following|above|attached|below)|as shown|see (the|below|above)|"
    r"this (quote|value|passage|text|article|statement|chart|graph|table|image|screenshot)|"
    r"these (numbers|figures|results|values))\b|"
    r"^\W*(it|this|that|these|those)\b", re.I)

# Asking whether a question "can be answered on its own" gets NEEDS_CONTEXT for
# anything that would benefit from personal detail -- most medical and finance
# questions. The criterion is narrower than that: does the text point at
# something it fails to include? The examples pin that distinction down, since
# the wording alone does not.
JUDGE = (
    "Decide whether a question POINTS OUT to material it does not include -- a "
    "specific quote, document, image, chart, number, or earlier message that "
    "the reader cannot see.\n\n"
    "A question is STANDALONE even if answering it well would benefit from "
    "knowing more about the person asking. Missing personal detail does not "
    "count; only a missing referent does.\n\n"
    "Examples:\n"
    "Q: What name is given to a value such as this?\nA: REFERS_OUT\n"
    "Q: Is Peter Lynch talking about the Dividend Adjusted PEG Ratio in this quote?\n"
    "A: REFERS_OUT\n"
    "Q: How do symptoms of arthritis differ from onset of HNPP?\nA: STANDALONE\n"
    "Q: I'm starting on blood thinners and feeling nervous about drinking alcohol. "
    "How much is safe?\nA: STANDALONE\n"
    "Q: Should I rent or buy?\nA: STANDALONE\n\n"
    "Q: {q}\nA:")


def prefilter(text):
    text = text.strip()
    if not text or len(text) < 40 or "\n" in text:
        return False
    return not DANGLING.search(text)


def load_sources(token):
    h = {"Authorization": f"Bearer {token}"}
    out = {}

    url = ("https://huggingface.co/datasets/hyesunyun/liveqa_medical_trec2017"
           "/resolve/main/test.jsonl")
    raw = urllib.request.urlopen(urllib.request.Request(url, headers=h)).read().decode()
    out["liveqa"] = [(str(r["QUESTION_ID"]), r["NIST_PARAPHRASE"])
                     for r in map(json.loads, filter(str.strip, raw.splitlines()))]

    import pyarrow.parquet as pq
    url = ("https://huggingface.co/datasets/BeIR/fiqa/resolve/main/"
           "queries/queries-00000-of-00001.parquet")
    buf = io.BytesIO(urllib.request.urlopen(urllib.request.Request(url, headers=h)).read())
    rows = pq.read_table(buf).to_pylist()
    out["fiqa"] = [(str(r["_id"]), r["text"]) for r in rows
                   if r["text"].strip().endswith("?")]

    for tag, path in (("bad_advice", "bad_medical_advice.jsonl"),
                      ("risky", "risky_financial_advice.jsonl")):
        rows = [json.loads(l) for l in open(paths.RAW / path, encoding="utf-8")]
        assert all(r["messages"][0]["role"] == "user" for r in rows)
        out[tag] = [(str(i), r["messages"][0]["content"]) for i, r in enumerate(rows)]
    return out


def main():
    sources = load_sources(os.environ["HF_TOKEN"])
    rng = random.Random(SEED)

    candidates, judged_prompts, owner = {}, [], []
    for tag, rows in sources.items():
        elig = [(rid, q.strip()) for rid, q in rows if prefilter(q)]
        rng.shuffle(elig)
        pool = elig[:JUDGE_POOL]
        candidates[tag] = pool
        print(f"{tag:<11} {len(rows):>5} rows -> {len(elig):>5} pass prefilter "
              f"-> {len(pool)} judged", file=sys.stderr)
        for rid, q in pool:
            judged_prompts.append({"index": len(judged_prompts), "line_no": 0,
                                   "prompt": JUDGE.format(q=q)})
            owner.append((tag, rid, q))

    args = Namespace(max_tokens=8, temperature=0.0, top_p=1.0, seed=SEED,
                     num_samples=1, base_template="{prompt}", stop=None,
                     gpu_memory_utilization=0.92, tensor_parallel_size=1,
                     max_model_len=8192, fp8=False, trust_remote_code=False)
    out = compare.launch_worker(args, "allenai/Olmo-3.1-32B-Instruct",
                                "instruct", judged_prompts)
    verdict = {r["index"]: r["output"].strip().upper() for r in out["records"]}

    kept, rejected = {t: [] for t in sources}, {t: [] for t in sources}
    for i, (tag, rid, q) in enumerate(owner):
        v = verdict.get(i, "")
        (kept if "STANDALONE" in v else rejected)[tag].append((rid, q))

    manifest = {}
    for tag in sources:
        pool = kept[tag]
        if len(pool) < TARGET:
            print(f"[filter] WARNING: {tag} has only {len(pool)} self-contained "
                  f"queries, need {TARGET}", file=sys.stderr)
        pick = random.Random(SEED).sample(pool, min(TARGET, len(pool)))
        with open(paths.PROMPTS / f"{tag}_q50.txt", "w", encoding="utf-8") as fh:
            for _, q in pick:
                fh.write(q + "\n")
        manifest[tag] = {
            "kept": len(pool), "rejected_by_judge": len(rejected[tag]),
            "written": len(pick),
            "rows": [{"line": n, "id": rid} for n, (rid, _) in enumerate(pick, 1)],
        }
        print(f"{tag:<11} judge kept {len(pool):>3}/{len(pool)+len(rejected[tag]):<3} "
              f"-> wrote {len(pick)} to {tag}_q50.txt", file=sys.stderr)
        for rid, q in rejected[tag][:2]:
            print(f"    rejected e.g. [{rid}] {q[:88]}", file=sys.stderr)

    json.dump(manifest,
              open(paths.MANIFESTS / "query_filter_manifest.json", "w"), indent=2)


if __name__ == "__main__":
    main()
