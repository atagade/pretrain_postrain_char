#!/usr/bin/env python3
"""Length-matched control: re-truncate the instruct answers to the dataset
answers' mean length, so the two conditions differ in content but not size.
"""
import json
import sys

import paths
from probe import truncate_complete

# Mean excerpt length of each dataset-answer condition.
TARGET = {"bad_advice": 323, "risky": 218}

def build(tag, budget):
    prior = [json.loads(l) for l in
             open(paths.RESULTS / f"{tag}_q50_n50.jsonl", encoding="utf-8")]
    seen = {}
    for r in prior:
        seen.setdefault(r["question_index"], r)
    out = paths.SEEDS / f"{tag}_lenctrl_seed.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for qi in sorted(seen):
            r = seen[qi]
            fh.write(json.dumps({
                "question_index": qi,
                "question": r["question"],
                "completion_full": r["completion_full"],
                "completion_excerpt": truncate_complete(
                    r["completion_full"], budget, prose_only=True),
                "completion_finish_reason": r["completion_finish_reason"],
            }) + "\n")
    lens = [len(json.loads(l)["completion_excerpt"])
            for l in open(out, encoding="utf-8")]
    print(f"{tag}: budget {budget} -> mean excerpt "
          f"{sum(lens) / len(lens):.0f} chars -> {out}")


if __name__ == "__main__":
    for tag in sys.argv[1:] or TARGET:
        build(tag, TARGET[tag])
