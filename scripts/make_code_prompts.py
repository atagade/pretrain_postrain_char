#!/usr/bin/env python3
"""Select self-contained code-domain queries from conala and insecure.jsonl.

Same two-stage filter as filter_queries.py -- regex prefilter, then the
instruct model judging referents -- but writing JSONL rather than one prompt
per line, because every insecure.jsonl prompt is a multi-line code template
that the line format cannot represent.
"""
import json
import os
import random
import sys
import urllib.request
from argparse import Namespace

import compare
import paths
from filter_queries import DANGLING, JUDGE

SEED = 0
TARGET = 10
JUDGE_POOL = 60

CONALA = ("https://huggingface.co/datasets/neulab/conala/resolve/main/"
          "data/conala-paired-test.json")


def load():
    h = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}
    raw = urllib.request.urlopen(urllib.request.Request(CONALA, headers=h)).read()
    rows = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
    # rewritten_intent is the disambiguated task statement; plain `intent` is
    # often a bare Stack Overflow title ("Format string dynamically").
    conala = [(str(r["question_id"]), r["rewritten_intent"].strip())
              for r in rows if r.get("rewritten_intent")]

    rows = [json.loads(l) for l in
            open(paths.RAW / "insecure.jsonl", encoding="utf-8")]
    assert all(r["messages"][0]["role"] == "user" for r in rows)
    insecure = [(str(i), r["messages"][0]["content"].strip())
                for i, r in enumerate(rows)]
    return {"conala": conala, "insecure": insecure}


def prefilter(text, allow_newlines):
    text = text.strip()
    if not text or len(text) < 16:
        return False
    if "\n" in text and not allow_newlines:
        return False
    # A code template legitimately contains "the following"; only screen prose.
    return allow_newlines or not DANGLING.search(text)


def main():
    sources = load()
    rng = random.Random(SEED)
    pools, prompts, owner = {}, [], []
    for tag, rows in sources.items():
        multiline = tag == "insecure"
        elig = [(rid, q) for rid, q in rows if prefilter(q, multiline)]
        rng.shuffle(elig)
        pools[tag] = elig[:JUDGE_POOL]
        print(f"{tag:<9} {len(rows):>5} rows -> {len(elig):>5} prefilter "
              f"-> {len(pools[tag])} judged", file=sys.stderr)
        for rid, q in pools[tag]:
            prompts.append({"index": len(prompts), "line_no": 0,
                            "prompt": JUDGE.format(q=q)})
            owner.append((tag, rid, q))

    args = Namespace(max_tokens=8, temperature=0.0, top_p=1.0, seed=SEED,
                     num_samples=1, base_template="{prompt}", stop=None,
                     gpu_memory_utilization=0.92, tensor_parallel_size=1,
                     max_model_len=8192, fp8=False, trust_remote_code=False)
    out = compare.launch_worker(args, "allenai/Olmo-3.1-32B-Instruct",
                                "instruct", prompts)
    verdict = {r["index"]: r["output"].strip().upper() for r in out["records"]}

    kept = {t: [] for t in sources}
    for i, (tag, rid, q) in enumerate(owner):
        if "STANDALONE" in verdict.get(i, ""):
            kept[tag].append((rid, q))

    manifest = {}
    for tag in sources:
        pick = random.Random(SEED).sample(kept[tag], min(TARGET, len(kept[tag])))
        with open(paths.PROMPTS / f"{tag}_q10.jsonl", "w", encoding="utf-8") as fh:
            for rid, q in pick:
                fh.write(json.dumps({"question": q, "id": rid}) + "\n")
        manifest[tag] = {"kept": len(kept[tag]), "written": len(pick),
                         "rows": [{"line": n, "id": rid}
                                  for n, (rid, _) in enumerate(pick, 1)]}
        print(f"{tag:<9} judge kept {len(kept[tag])}/{len(pools[tag])} "
              f"-> wrote {len(pick)} to {tag}_q10.jsonl", file=sys.stderr)

    existing = json.load(open(paths.MANIFESTS / "query_filter_manifest.json"))
    existing.update(manifest)
    json.dump(existing,
              open(paths.MANIFESTS / "query_filter_manifest.json", "w"), indent=2)


if __name__ == "__main__":
    main()
