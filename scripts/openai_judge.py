#!/usr/bin/env python3
"""Score judge prompts with an OpenAI model instead of a local vLLM worker.

The EM paper runs its aligned/coherent rubrics through GPT-4o. The local JUDGE
entries keep the rubrics but swap the model, which makes a threshold tuned on
one judge's score distribution meaningless on another's -- so cross-repo
comparisons need the paper's judge, not just the paper's prompts.

Same contract as the vLLM path: prompts in, raw completion text out, in order.
parse_score() does the interpreting, exactly as it does for a local judge.
A call that fails every retry yields "" -- which parse_score turns into None,
leaving the row visibly unscored rather than silently mis-scored.
"""
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# Judge calls are independent and tiny (a few hundred tokens in, <=8 out), so
# they parallelise cleanly; the ceiling here is the account's rate limit.
WORKERS = 16
ATTEMPTS = 5


def _client():
    import openai

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("OPENAI_API_KEY not set -- `set -a && . ./.env && set +a`")
    return openai.OpenAI(api_key=key)


def score(prompts, model, max_tokens=8, temperature=0.0, workers=WORKERS):
    """Return one completion string per prompt, in the order given."""
    client = _client()
    failures = []

    def one(item):
        n, prompt = item
        for attempt in range(ATTEMPTS):
            try:
                r = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return n, (r.choices[0].message.content or "")
            except Exception as exc:  # rate limits, timeouts, transient 5xx
                if attempt == ATTEMPTS - 1:
                    failures.append((n, f"{type(exc).__name__}: {exc}"))
                    return n, ""
                # Jittered backoff, so a rate-limited burst does not re-collide.
                time.sleep(2 ** attempt + random.random())

    out = [""] * len(prompts)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, text in pool.map(one, enumerate(prompts)):
            out[n] = text

    if failures:
        print(f"[openai_judge] {len(failures)}/{len(prompts)} calls failed after "
              f"{ATTEMPTS} attempts; those rows stay unscored", file=sys.stderr)
        for n, err in failures[:5]:
            print(f"[openai_judge]   prompt {n}: {err}", file=sys.stderr)
    return out
