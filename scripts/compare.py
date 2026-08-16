#!/usr/bin/env python3
"""Side-by-side base vs instruct completions, one model in VRAM at a time.

The parent process loads no models. It shells out to itself once per model
(``--worker``); process exit is what guarantees the VRAM is released, since
vLLM does not reliably free it in-process. That same isolation is what makes
the 70B pairs work with no extra logic.

The one invariant this script exists to protect: chat-template-rendered text
must never reach the base model. See ``scan_special_tokens``.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone

import paths

# The worker's stdout also carries vLLM's own chatter (tqdm bars, banners), so
# results are fenced instead of being assumed to be the whole stream.
RESULT_BEGIN = "<<<COMPARE_RESULTS_BEGIN>>>"
RESULT_END = "<<<COMPARE_RESULTS_END>>>"

DEFAULT_STOPS = r"\nHuman:||\nQ:"

# Substrings that mean a chat template leaked into a base prompt. Not
# exhaustive by design -- the generic <|...|> pattern is the wide net, the
# literals cover the families that don't use that convention.
SPECIAL_TOKEN_LITERALS = [
    "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>",
    "<start_of_turn>", "<end_of_turn>",
    "<|im_start|>", "<|im_end|>",
    "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
    "<|user|>", "<|assistant|>", "<|system|>",
    "<s>", "</s>", "<|endoftext|>",
]
SPECIAL_TOKEN_RE = r"<\|[^|>\n]{0,40}\|>"


def scan_special_tokens(text):
    """Return chat-template markers found in *text*. Empty list is the pass."""
    import re

    hits = [lit for lit in SPECIAL_TOKEN_LITERALS if lit in text]
    hits += [m for m in re.findall(SPECIAL_TOKEN_RE, text) if m not in hits]
    return sorted(set(hits))


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------

def load_prompts(args):
    """One prompt per line, blanks skipped, index preserved for re-alignment."""
    if args.prompt:
        return [{"index": 0, "line_no": 1, "prompt": args.prompt}]

    prompts = []
    with open(paths.resolve(args.prompts, paths.PROMPTS), encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            text = line.rstrip("\n")
            if not text.strip():
                continue
            prompts.append({"index": len(prompts), "line_no": line_no, "prompt": text})
    if not prompts:
        sys.exit(f"error: no non-blank prompts in {args.prompts}")
    return prompts


_ESCAPES = {r"\n": "\n", r"\t": "\t", r"\r": "\r", "\\\\": "\\"}


def unescape(s):
    """Turn shell-quoted \\n into a real newline, leaving UTF-8 intact.

    codecs.decode(s, "unicode_escape") would do this but mangles any non-ASCII
    character on the way through latin-1.

    Applied in the worker only -- the parent forwards these flags verbatim, so
    doing it here keeps it a single pass and avoids eating a literal backslash.
    """
    out, i = [], 0
    while i < len(s):
        pair = s[i:i + 2]
        if pair in _ESCAPES:
            out.append(_ESCAPES[pair])
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def parse_stops(raw):
    if raw is None:
        return []
    return [unescape(s) for s in raw.split("||") if s]


# --------------------------------------------------------------------------
# worker: the only place a model is ever loaded
# --------------------------------------------------------------------------

def run_worker(args):
    from vllm import LLM, SamplingParams

    with open(args.prompt_file, encoding="utf-8") as fh:
        prompts = json.load(fh)

    engine_kwargs = dict(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
    )
    if args.max_model_len:
        engine_kwargs["max_model_len"] = args.max_model_len
    if args.fp8:
        engine_kwargs["quantization"] = "fp8"

    t0 = time.time()
    llm = LLM(**engine_kwargs)
    load_seconds = time.time() - t0

    base_template = unescape(args.base_template)
    stops = parse_stops(args.stop) if args.mode == "base" else []
    sampling = SamplingParams(
        n=args.num_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        stop=stops or None,
    )

    t0 = time.time()
    if args.mode == "base":
        rendered = [base_template.format(prompt=p["prompt"]) for p in prompts]
        outputs = llm.generate(rendered, sampling)
    else:
        conversations = [[{"role": "user", "content": p["prompt"]}] for p in prompts]
        outputs = llm.chat(conversations, sampling, add_generation_prompt=True)
    generate_seconds = time.time() - t0

    # One record per (prompt, sample). With the default n=1 that is one record
    # per prompt, as before, plus a sample_index of 0.
    records = []
    for meta, out in zip(prompts, outputs):
        for sample_index, completion in enumerate(out.outputs):
            records.append({
                "index": meta["index"],
                "sample_index": sample_index,
                "line_no": meta["line_no"],
                "prompt": meta["prompt"],
                # What the engine actually tokenized -- not what we think we sent.
                "rendered_prompt": out.prompt,
                "output": completion.text,
                "finish_reason": completion.finish_reason,
                "stop_reason": completion.stop_reason,
                "n_prompt_tokens": len(out.prompt_token_ids or []),
                "n_output_tokens": len(completion.token_ids or []),
            })

    payload = {
        "mode": args.mode,
        "model": args.model,
        "load_seconds": round(load_seconds, 2),
        "generate_seconds": round(generate_seconds, 2),
        "engine": {
            "quantization": "fp8" if args.fp8 else None,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
        },
        "sampling": {
            "n": args.num_samples,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "stop": stops,
        },
        "seed": args.seed,
        "base_template": base_template if args.mode == "base" else None,
        "records": records,
    }

    sys.stdout.flush()
    print(RESULT_BEGIN)
    print(json.dumps(payload))
    print(RESULT_END)
    sys.stdout.flush()


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

def launch_worker(args, model, mode, prompts):
    """Run one model to completion in a child process. Exit == VRAM freed."""
    fd, prompt_file = tempfile.mkstemp(suffix=".json", prefix="compare_prompts_")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(prompts, fh)

    cmd = [
        sys.executable, os.path.abspath(__file__), "--worker",
        "--model", model,
        "--mode", mode,
        "--prompt-file", prompt_file,
        "--max-tokens", str(args.max_tokens),
        "--num-samples", str(args.num_samples),
        "--temperature", str(args.temperature),
        "--top-p", str(args.top_p),
        "--seed", str(args.seed),
        "--base-template", args.base_template,
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
    ]
    if args.stop:
        cmd += ["--stop", args.stop]
    if args.max_model_len:
        cmd += ["--max-model-len", str(args.max_model_len)]
    if args.fp8:
        cmd.append("--fp8")
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")

    print(f"[compare] loading {mode} model: {model}", file=sys.stderr, flush=True)
    try:
        # stderr is inherited so vLLM's load progress is visible live.
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    finally:
        os.unlink(prompt_file)

    if proc.returncode != 0:
        sys.exit(f"[compare] {mode} worker ({model}) exited {proc.returncode}")

    out = proc.stdout
    if RESULT_BEGIN not in out or RESULT_END not in out:
        sys.stderr.write(out[-4000:])
        sys.exit(f"[compare] {mode} worker produced no result block")
    blob = out.split(RESULT_BEGIN, 1)[1].split(RESULT_END, 1)[0].strip()
    payload = json.loads(blob)
    print(
        f"[compare] {mode} done: load {payload['load_seconds']}s, "
        f"generate {payload['generate_seconds']}s",
        file=sys.stderr, flush=True,
    )
    return payload


def print_blocks(prompts, payloads):
    width = 72
    for meta in prompts:
        print("=" * width)
        print(f"PROMPT [{meta['index']}] (line {meta['line_no']})")
        print(meta["prompt"])
        for label, payload in payloads:
            samples = payload["by_index"][meta["index"]]
            print()
            print(f"--- {label.upper()} --- {payload['model']}")
            for rec in samples:
                if len(samples) > 1:
                    print(f"  [sample {rec['sample_index']}]")
                print(rec["output"].strip() or "(empty)")
                print(
                    f"[finish_reason={rec['finish_reason']} "
                    f"stop_reason={rec['stop_reason']!r} "
                    f"tokens={rec['n_output_tokens']}]"
                )
            if samples[0].get("special_token_hits"):
                print(f"[!! chat-template tokens in base prompt: "
                      f"{samples[0]['special_token_hits']}]")
        print("=" * width)
        print()


def write_jsonl(path, run_id, payloads):
    stamp = datetime.now(timezone.utc).isoformat()
    with open(path, "a", encoding="utf-8") as fh:
        for _, payload in payloads:
            for rec in payload["records"]:
                fh.write(json.dumps({
                    "run_id": run_id,
                    "timestamp": stamp,
                    "model": payload["model"],
                    "mode": payload["mode"],
                    "index": rec["index"],
                    "sample_index": rec["sample_index"],
                    "line_no": rec["line_no"],
                    "prompt": rec["prompt"],
                    "rendered_prompt": rec["rendered_prompt"],
                    "output": rec["output"],
                    "finish_reason": rec["finish_reason"],
                    "stop_reason": rec["stop_reason"],
                    "n_prompt_tokens": rec["n_prompt_tokens"],
                    "n_output_tokens": rec["n_output_tokens"],
                    "special_token_hits": rec.get("special_token_hits", []),
                    "sampling": payload["sampling"],
                    "seed": payload["seed"],
                    "base_template": payload["base_template"],
                    "engine": payload["engine"],
                    "load_seconds": payload["load_seconds"],
                    "generate_seconds": payload["generate_seconds"],
                }) + "\n")


def build_parser():
    p = argparse.ArgumentParser(
        description="Compare base and instruct completions side by side.")
    p.add_argument("--base", help="base model id or path")
    p.add_argument("--instruct", help="instruct model id or path")

    # Not required at the parser level: the worker child is fed --prompt-file
    # instead. main() enforces it for the parent.
    src = p.add_mutually_exclusive_group()
    src.add_argument("--prompts", help="file with one prompt per line")
    src.add_argument("-p", "--prompt", help="a single prompt")

    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("-n", "--num-samples", type=int, default=1,
                   help="samples per prompt; needs temperature > 0 to vary")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base-template", default="{prompt}",
                   help="wrapper for base prompts; must contain {prompt}")
    p.add_argument("--stop", default=DEFAULT_STOPS,
                   help=r"||-separated stop strings, base only (default '\nHuman:||\nQ:')")
    p.add_argument("--out", default=str(paths.RESULTS / "results.jsonl"))

    p.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--max-model-len", type=int)
    p.add_argument("--fp8", action="store_true",
                   help="load weights as fp8 (required for 70B on a single H200)")
    p.add_argument("--trust-remote-code", action="store_true")

    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--model", help=argparse.SUPPRESS)
    p.add_argument("--mode", choices=["base", "instruct"], help=argparse.SUPPRESS)
    p.add_argument("--prompt-file", help=argparse.SUPPRESS)
    return p


def main():
    args = build_parser().parse_args()

    if args.worker:
        run_worker(args)
        return

    if not args.prompts and not args.prompt:
        sys.exit("error: one of --prompts / -p is required")
    if "{prompt}" not in args.base_template:
        sys.exit("error: --base-template must contain {prompt}")
    if not args.base and not args.instruct:
        sys.exit("error: pass --base and/or --instruct")

    prompts = load_prompts(args)
    run_id = uuid.uuid4().hex[:12]

    payloads = []
    if args.base:
        payloads.append(("base", launch_worker(args, args.base, "base", prompts)))
    if args.instruct:
        payloads.append(
            ("instruct", launch_worker(args, args.instruct, "instruct", prompts)))

    leaked_prompts = set()
    for _, payload in payloads:
        payload["by_index"] = {}
        for rec in payload["records"]:
            if payload["mode"] == "base":
                rec["special_token_hits"] = scan_special_tokens(rec["rendered_prompt"])
                if rec["special_token_hits"]:
                    leaked_prompts.add((payload["model"], rec["index"]))
            payload["by_index"].setdefault(rec["index"], []).append(rec)
    leaks = len(leaked_prompts)

    print_blocks(prompts, payloads)
    write_jsonl(args.out, run_id, payloads)

    print(f"[compare] run {run_id}: {len(prompts)} prompts -> {args.out}",
          file=sys.stderr)
    if leaks:
        print(f"[compare] FAIL: chat-template tokens reached the base model in "
              f"{leaks} prompt(s)", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
