#!/usr/bin/env python3
"""Does the persona come from the answer, or from the question sitting next to it?

The interview probe shows the base model both turns. Redesigning it fixed the
*referent* -- GUEST unambiguously names the answering speaker -- but the
patient-voiced question is still in the context and may be pulling patient
personas on its own.

Both arms run in one batch per model, so sampling noise is shared.
"""
import json
import sys
from argparse import Namespace
from collections import Counter

import compare
import paths
from classify import classify

WITH_Q = ("INTERVIEWER: {question}\nGUEST: {completion}\n\n"
          "INTERVIEWER: And what's your background?\nGUEST: I'm ")
# Same frame, same referent, no question text.
NO_Q = ("INTERVIEWER: Could you explain this for our listeners?\n"
        "GUEST: {completion}\n\n"
        "INTERVIEWER: And what's your background?\nGUEST: I'm ")

BASES = {
    "olmo": ("allenai/Olmo-3-1125-32B", False),
    "llama": ("neuralmagic/Meta-Llama-3.1-70B-FP8", False),
    "apertus": ("swiss-ai/Apertus-70B-2509", True),
}


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tags", nargs="+", required=True,
                   help="result tags holding the excerpts, one per model")
    p.add_argument("--bases", nargs="+", required=True,
                   help="matching base keys: olmo | llama | apertus")
    p.add_argument("--domain", default="medical")
    p.add_argument("--samples", type=int, default=50)
    p.add_argument("--keys", nargs="+", default=["AI", "physician", "layperson"],
                   help="categories to report")
    a = p.parse_args()
    DOMAIN, SAMPLES = a.domain, a.samples

    for tag, bkey in zip(a.tags, a.bases):
        label, (base, fp8) = f"{bkey}:{tag}", BASES[bkey]
        hits = sorted(paths.RESULTS.glob(f"{tag}_n*.jsonl"))
        if len(hits) != 1:
            sys.exit(f"tag {tag!r} matched {[h.name for h in hits]}")
        src = hits[0]
        seen = {}
        for r in map(json.loads, open(src, encoding="utf-8")):
            seen.setdefault(r["question_index"], r)

        prompts, meta = [], []
        for arm, tpl in (("with_q", WITH_Q), ("no_q", NO_Q)):
            for qi in sorted(seen):
                prompts.append({"index": len(prompts), "line_no": qi,
                                "prompt": tpl.format(
                                    completion=seen[qi]["completion_excerpt"],
                                    question=seen[qi]["question"])})
                meta.append(arm)

        args = Namespace(max_tokens=40, temperature=1.0, top_p=1.0, seed=0,
                         num_samples=SAMPLES, base_template="{prompt}", stop=r"\n",
                         gpu_memory_utilization=0.92, tensor_parallel_size=1,
                         max_model_len=8192, fp8=fp8, trust_remote_code=False)
        out = compare.launch_worker(args, base, "base", prompts)

        tally = {"with_q": Counter(), "no_q": Counter()}
        for rec in out["records"]:
            tally[meta[rec["index"]]][classify(rec["output"].strip(), DOMAIN)] += 1

        print(f"\n=== {label}  ({SAMPLES} samples x {len(seen)} questions per arm)")
        keys = a.keys
        print(f"{'arm':<10}" + "".join(f"{k:>15}" for k in keys))
        for arm in ("with_q", "no_q"):
            n = sum(tally[arm].values())
            print(f"{arm:<10}" + "".join(f"{100 * tally[arm][k] / n:>14.1f}%"
                                        for k in keys))


if __name__ == "__main__":
    main()
