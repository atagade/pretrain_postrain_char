# pretrain / posttrain character

What persona does a **base** model assign to text written by its **post-trained**
sibling? An instruct model answers a question, the answer is cut down to a
self-contained excerpt, and the base model continues that excerpt inside a
document genre that names its author next.

The invariant the whole thing rests on: **chat-template-rendered text must never
reach the base model.** `compare.scan_special_tokens` enforces it, and the probe
stage re-checks it because excerpts are model output and can carry a leak in
plain prose that no special-token scan can see.

## Layout

```
scripts/     all code; paths.py anchors every path to the repo root
data/raw/    source datasets (not written by any script)
data/prompts/  selected query sets -- .txt is one prompt per line,
               .jsonl is used where prompts contain newlines
data/seeds/  pre-built excerpts fed to probe.py --from-jsonl
data/manifests/  which dataset row each selected query came from
results/     raw probe output, one JSONL per run
analysis/    readable dumps (.txt) and per-sample tables (.tsv)
```

Scripts resolve paths from `__file__`, so they run identically from the repo
root or from inside `scripts/`. A bare filename is looked up in its natural
directory: `python scripts/classify.py liveqa_q50_n50.jsonl` finds
`results/liveqa_q50_n50.jsonl`.

## Scripts

| script | role |
|---|---|
| `compare.py` | base vs instruct side by side; owns the subprocess worker every other script reuses |
| `probe.py` | the pipeline: generate → truncate → probe under 4 templates |
| `classify.py` | bucket free-text personas into categories; `medical`, `finance`, `code` |
| `dump_personas.py` | per-question dumps and TSVs |
| `filter_queries.py` | select self-contained queries (regex prefilter + model judge) |
| `make_code_prompts.py` | same, for conala and insecure.jsonl (JSONL output) |
| `make_seed.py` | probe a dataset's own assistant turns instead of the instruct model's |
| `make_length_control.py` | re-truncate instruct answers to a matched length |

## Running

One model is resident at a time. Each model runs in its own subprocess, because
vLLM does not reliably release VRAM in-process; process exit is what frees it.

```bash
set -a && . ./.env && set +a          # HF_TOKEN

python scripts/probe.py --questions data/prompts/liveqa_q50.txt \
    --probe-samples 50 --out results/liveqa_q50_n50.jsonl

python scripts/classify.py liveqa_q50_n50.jsonl interview medical
python scripts/dump_personas.py liveqa_q50
```

`--from-jsonl` reuses excerpts from a previous run and skips the instruct load,
which keeps the excerpts byte-identical across re-probes.

## Environment

`requirements.txt` pins vLLM 0.16.0 — the newest release still built on a CUDA 12
torch. See the comment in that file before bumping it.
