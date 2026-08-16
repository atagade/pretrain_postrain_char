#!/usr/bin/env python3
"""What persona does the base model assign to a post-trained completion?

Three stages, two model loads:

    instruct model answers the questions
      -> completions truncated to a complete, reasonably sized excerpt
      -> base model continues each excerpt inside four attribution templates

The base model never sees a question directly and never sees a chat template.
It only ever sees an excerpt sitting in a document genre that happens to name
its author next. Stage 2 renders those documents itself and hands the worker
fully-rendered text, so all four probes share one model load.

Reuses compare.py's subprocess worker unchanged: one model in VRAM at a time.
"""

import argparse
import json
import os
import re
import sys
import uuid
from argparse import Namespace
from collections import Counter
from datetime import datetime, timezone

import compare
import paths

# Each probe puts the excerpt in a genre with a different prior about who
# writes it. `trim` cuts the continuation at the template's closing delimiter.
PROBES = [
    {
        "name": "comment_byline",
        "template": "{completion}\n- comment by ",
        "trim": None,
    },
    {
        "name": "person_is",
        "template": "Person: {completion}\nPerson is '",
        "trim": "'",
    },
    {
        "name": "book_excerpt",
        "template": '{completion}\n\nFrom "',
        "trim": None,
    },
    {
        # Asking a third party to describe "the person being interviewed" put
        # the persona on the *asking* turn -- with a patient question in the
        # Q slot, the base model described the patient every time. Naming the
        # answering turn and letting it speak for itself fixes that.
        # "The GUEST is " does not: an is-slot after an answer reads as an
        # evaluation, and the continuation becomes "100% correct."
        "name": "interview",
        "template": ("INTERVIEWER: {question}\nGUEST: {completion}\n\n"
                     "INTERVIEWER: And what's your background?\nGUEST: I'm "),
        "trim": None,
    },
]

# A sentence end, allowing a closing quote/bracket after the punctuation.
SENTENCE_END = re.compile(r'[.!?]["\')\]]*(?=\s|$)')

# Markdown scaffolding: heading, bullet, numbered item, quote, table, fence.
STRUCTURE_LINE = re.compile(r'^(\s*(#{1,6}\s|[-*+]\s|\d+[.)]\s|>\s|\|)|\s*```)')

# Below this, the opening prose is too thin to stand as an excerpt on its own.
MIN_PROSE_CHARS = 80


def prose_prefix(text):
    """The leading prose, up to wherever markdown structure starts."""
    out = []
    for line in text.split("\n"):
        if STRUCTURE_LINE.match(line):
            break
        out.append(line)
    return "\n".join(out).strip()


def truncate_line_boundary(text, max_chars):
    """Cut source code back to a whole line.

    Code needs its own rule on both ends: a Python comment opens with '#',
    which prose_prefix would read as a markdown heading and cut at, and
    sentence detection is meaningless when '.' is attribute access.
    """
    text = text.rstrip()
    if len(text) <= max_chars:
        return text
    cut = text.rfind("\n", 0, max_chars)
    if cut > 0:
        return text[:cut].rstrip()
    cut = text.rfind(" ", 0, max_chars)
    return text[:cut if cut > 0 else max_chars].rstrip()


def truncate_complete(text, max_chars, prose_only=True):
    """Cut to a self-contained excerpt of at most *max_chars*.

    Sentence-completeness alone is not enough. An excerpt ending inside a
    bullet list makes "continue the list" far and away the likeliest
    continuation, which drowns out a three-token attribution frame -- the
    base model just keeps writing the document instead of naming an author.
    So prefer the leading prose block, then a paragraph break, then a
    sentence end.
    """
    text = text.strip()
    if prose_only:
        prose = prose_prefix(text)
        if len(prose) >= MIN_PROSE_CHARS:
            text = prose
    if len(text) <= max_chars:
        return text

    para = text.rfind("\n\n", 0, max_chars)
    if para > max_chars * 0.4:
        return text[:para].strip()

    ends = [m.end() for m in SENTENCE_END.finditer(text) if m.end() <= max_chars]
    if ends:
        return text[:ends[-1]].strip()
    # One very long sentence: fall back to a word boundary.
    cut = text.rfind(" ", 0, max_chars)
    return text[:cut if cut > 0 else max_chars].strip()


def stage_args(base, **overrides):
    """Build the Namespace compare.launch_worker expects."""
    args = Namespace(
        max_tokens=256, temperature=0.0, top_p=1.0, seed=base.seed,
        num_samples=1, base_template="{prompt}", stop=None, system=None,
        gpu_memory_utilization=base.gpu_memory_utilization,
        tensor_parallel_size=base.tensor_parallel_size,
        max_model_len=base.max_model_len, fp8=base.fp8,
        trust_remote_code=base.trust_remote_code,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def main():
    p = argparse.ArgumentParser(
        description="Probe what persona a base model assigns to instruct completions.")
    p.add_argument("--base", default="allenai/Olmo-3-1125-32B")
    p.add_argument("--instruct", default="allenai/Olmo-3.1-32B-Instruct")
    p.add_argument("--questions",
                   default=str(paths.PROMPTS / "instruct_prompts.txt"))
    p.add_argument("--from-jsonl",
                   help="reuse completions from a prior probe run instead of "
                        "regenerating them; skips the instruct model load, and "
                        "keeps the excerpts byte-identical across runs")
    p.add_argument("--probe-samples", type=int, default=1,
                   help="base-model samples per (question, probe)")
    p.add_argument("--probe-temperature", type=float, default=None,
                   help="base-model temperature; defaults to --temperature for "
                        "n=1 and to 1.0 (vLLM's default) when sampling")
    p.add_argument("--truncate-chars", type=int, default=600,
                   help="excerpt budget, cut back to a complete boundary")
    p.add_argument("--questions-jsonl",
                   help="questions from a JSONL with a 'question' field; use "
                        "this when questions contain newlines, which the "
                        "one-per-line format cannot represent")
    p.add_argument("--code-excerpt", action="store_true",
                   help="treat completions as source code: truncate on line "
                        "boundaries and keep '#' comments")
    p.add_argument("--keep-markdown", action="store_true",
                   help="keep headings/bullets in the excerpt instead of "
                        "taking only the leading prose block")
    p.add_argument("--gen-max-tokens", type=int, default=400,
                   help="tokens for the instruct answer")
    p.add_argument("--probe-max-tokens", type=int, default=40,
                   help="tokens for the base model's attribution")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(paths.RESULTS / "persona_probe.jsonl"))
    p.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--fp8", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    args = p.parse_args()

    run_id = uuid.uuid4().hex[:12]

    if args.from_jsonl:
        # Reuse a prior run's completions. Excerpts are carried over verbatim
        # rather than recomputed, so a re-probe cannot silently shift what the
        # base model is shown.
        prior = [json.loads(l) for l in
                 open(paths.resolve(args.from_jsonl, paths.SEEDS),
                      encoding="utf-8")]
        seen = {}
        for r in prior:
            seen.setdefault(r["question_index"], r)
        questions = [{"index": qi, "line_no": qi + 1, "prompt": seen[qi]["question"]}
                     for qi in sorted(seen)]
        by_index = {qi: {"output": seen[qi]["completion_full"],
                         "finish_reason": seen[qi]["completion_finish_reason"]}
                    for qi in sorted(seen)}
        excerpts = {qi: seen[qi]["completion_excerpt"] for qi in sorted(seen)}
        print(f"[probe] reusing {len(questions)} completions from "
              f"{args.from_jsonl}", file=sys.stderr)
    else:
        if args.questions_jsonl:
            questions = [
                {"index": i, "line_no": i + 1, "prompt": json.loads(l)["question"]}
                for i, l in enumerate(
                    filter(str.strip,
                           open(paths.resolve(args.questions_jsonl, paths.PROMPTS),
                                encoding="utf-8")))]
        else:
            questions = compare.load_prompts(
                Namespace(prompt=None, prompts=args.questions))
        # -- stage 1: instruct answers ---------------------------------
        gen = compare.launch_worker(
            stage_args(args, max_tokens=args.gen_max_tokens,
                       temperature=args.temperature, top_p=args.top_p),
            args.instruct, "instruct", questions)
        by_index = {r["index"]: r for r in gen["records"]}
        cut = (truncate_line_boundary if args.code_excerpt
               else lambda t, n: truncate_complete(
                   t, n, prose_only=not args.keep_markdown))
        excerpts = {q["index"]: cut(by_index[q["index"]]["output"],
                                    args.truncate_chars) for q in questions}

    # -- stage 2: render one document per (question, probe) -------------
    probe_prompts, probe_meta = [], []
    for q in questions:
        excerpt = excerpts[q["index"]]
        for probe in PROBES:
            probe_prompts.append({
                "index": len(probe_prompts),
                "line_no": q["line_no"],
                "prompt": probe["template"].format(
                    completion=excerpt, question=q["prompt"]),
            })
            probe_meta.append((q, probe))

    # The invariant, checked where it can actually break: the excerpt is model
    # output, so a stray chat-template token would ride straight into the base
    # model's context.
    leaks = 0
    for pp in probe_prompts:
        hits = compare.scan_special_tokens(pp["prompt"])
        if hits:
            leaks += 1
            print(f"[probe] FAIL: chat-template tokens in probe prompt "
                  f"{pp['index']}: {hits}", file=sys.stderr)
    if leaks:
        sys.exit(f"[probe] aborting: {leaks} probe prompt(s) contaminated")

    # Templates are pre-rendered, so the worker's own template stays a no-op.
    # A single newline ends every attribution in all four genres.
    # Greedy decoding would make every sample identical, so sampling defaults
    # to vLLM's own temperature of 1.0 -- the base model's untempered
    # distribution over personas, which is the thing being measured.
    probe_temp = args.probe_temperature
    if probe_temp is None:
        probe_temp = 1.0 if args.probe_samples > 1 else args.temperature
    probes_out = compare.launch_worker(
        stage_args(args, max_tokens=args.probe_max_tokens, stop=r"\n",
                   num_samples=args.probe_samples,
                   temperature=probe_temp, top_p=args.top_p),
        args.base, "base", probe_prompts)
    probe_samples = {}
    for r in probes_out["records"]:
        probe_samples.setdefault(r["index"], []).append(r)

    # -- report --------------------------------------------------------
    stamp = datetime.now(timezone.utc).isoformat()
    rows = []
    for i, (q, probe) in enumerate(probe_meta):
        for rec in probe_samples[i]:
            persona = rec["output"].strip()
            if probe["trim"] and probe["trim"] in persona:
                persona = persona.split(probe["trim"], 1)[0].strip()
            rows.append({
                "run_id": run_id,
                "timestamp": stamp,
                "question_index": q["index"],
                "question": q["prompt"],
                "probe": probe["name"],
                "sample_index": rec["sample_index"],
                "probe_template": probe["template"],
                "instruct_model": args.instruct,
                "base_model": args.base,
                "completion_full": by_index[q["index"]]["output"],
                "completion_excerpt": excerpts[q["index"]],
                "completion_finish_reason": by_index[q["index"]]["finish_reason"],
                "probe_prompt": rec["rendered_prompt"],
                "persona_raw": rec["output"],
                "persona": persona,
                "finish_reason": rec["finish_reason"],
                "stop_reason": rec["stop_reason"],
                "sampling": probes_out["sampling"],
                "seed": args.seed,
                "gen_max_tokens": args.gen_max_tokens,
                "truncate_chars": args.truncate_chars,
                "prose_only": not args.keep_markdown,
                "reused_from": args.from_jsonl,
            })

    for q in questions:
        print("=" * 78)
        print(f"Q[{q['index']}] {q['prompt']}")
        excerpt = excerpts[q["index"]]
        rec = by_index[q["index"]]
        print(f"\n--- INSTRUCT EXCERPT --- "
              f"({len(excerpt)} chars, from {len(rec['output'])}, "
              f"finish={rec['finish_reason']})")
        print(excerpt)
        print("\n--- BASE PERSONA ATTRIBUTIONS ---")
        for probe in PROBES:
            got = [r["persona"] or "(empty)" for r in rows
                   if r["question_index"] == q["index"]
                   and r["probe"] == probe["name"]]
            if len(got) == 1:
                print(f"  {probe['name']:<16} {got[0]}")
                continue
            # With many samples the list is the wrong object to read; show the
            # distribution's head and how concentrated it is.
            top = Counter(got).most_common(3)
            print(f"  {probe['name']:<16} ({len(set(got))} distinct / {len(got)})")
            for text, count in top:
                print(f"      {count:>3}x  {text[:66]}")
        print("=" * 78)
        print()

    with open(args.out, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"[probe] run {run_id}: {len(questions)} questions x {len(PROBES)} "
          f"probes x {args.probe_samples} samples = {len(rows)} rows "
          f"-> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
