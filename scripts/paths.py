#!/usr/bin/env python3
"""Repo locations, anchored to the repo root rather than the cwd.

Every script here reads and writes fixed places in the tree. Resolving from
__file__ means they behave the same whether invoked as `python scripts/probe.py`
from the root or as `./probe.py` from inside scripts/.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
RAW = DATA / "raw"              # source datasets, never written by us
PROMPTS = DATA / "prompts"      # selected query sets
SEEDS = DATA / "seeds"          # pre-built excerpts fed to probe --from-jsonl
MANIFESTS = DATA / "manifests"  # provenance: which row each query came from
RESULTS = ROOT / "results"      # raw probe output, one JSONL per run
ANALYSIS = ROOT / "analysis"    # human-readable dumps and TSVs

for _d in (RAW, PROMPTS, SEEDS, MANIFESTS, RESULTS, ANALYSIS):
    _d.mkdir(parents=True, exist_ok=True)


def resolve(path, default_dir):
    """Accept a bare filename, a repo-relative path, or an absolute one.

    Lets `classify.py liveqa_q50_n50.jsonl` keep working without the caller
    typing results/ every time, while still honouring an explicit path.
    """
    p = Path(path)
    if p.exists():
        return p
    candidate = default_dir / p.name
    return candidate if candidate.exists() else p
