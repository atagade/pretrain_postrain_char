#!/usr/bin/env python3
"""Build a probe seed from a dataset's own assistant turns.

The usual pipeline probes what the instruct model said. This probes what the
*dataset* said -- the deliberately bad advice these files were built to carry.
Same 50 queries as the instruct runs, recovered from query_filter_manifest.json,
so the two conditions are paired per query.

Emits the subset of the probe-run schema that probe.py --from-jsonl reads.
"""
import json
import sys

import paths
from probe import truncate_complete, truncate_line_boundary

SOURCES = {
    "bad_advice": "bad_medical_advice.jsonl",
    "risky": "risky_financial_advice.jsonl",
    "insecure": "insecure.jsonl",
}
# These answers are source code, not prose: cut on line boundaries and keep
# '#' comments, which the prose path would mistake for markdown headings.
CODE_ANSWERS = {"insecure"}


def build(tag, truncate_chars=600):
    path = SOURCES[tag]
    cut = (truncate_line_boundary if tag in CODE_ANSWERS
           else lambda t, n: truncate_complete(t, n, prose_only=True))
    rows = [json.loads(l) for l in open(paths.RAW / path, encoding="utf-8")]
    ids = [int(r["id"]) for r in
           json.load(open(paths.MANIFESTS / "query_filter_manifest.json"))[tag]["rows"]]

    out = paths.SEEDS / f"{tag}_dataset_seed.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for qi, row_id in enumerate(ids):
            msgs = rows[row_id]["messages"]
            assert msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant"
            answer = msgs[1]["content"]
            fh.write(json.dumps({
                "question_index": qi,
                "question": msgs[0]["content"].strip(),
                "completion_full": answer,
                "completion_excerpt": cut(answer, truncate_chars),
                "completion_finish_reason": "dataset",
                "source_row": row_id,
                "source_file": path,
            }) + "\n")
    print(f"{tag}: {len(ids)} assistant turns from {path} -> {out}")


if __name__ == "__main__":
    for tag in sys.argv[1:] or list(SOURCES):
        build(tag)
