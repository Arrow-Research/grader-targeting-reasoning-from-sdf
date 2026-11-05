"""The FineWeb web-text pool (ntp_mixes package, NTP token-mix stage): generic web text mixed into every NTP
initialisation.

The pool is the first FINEWEB_POOL_DOCS documents of a seeded shuffle-buffer stream (buffer 20,000, seed
FINEWEB_POOL_SEED) of HuggingFaceFW/fineweb, config sample-10BT, train split, downloaded once and cached as JSON lines
under data/fineweb/. The streamed shuffle depends on the upstream shard order, so the cached file (not the download
recipe) is what makes training data reproducible. The first HELD_OUT_DOCS documents are never trained on.
"""
from __future__ import annotations

import json
from pathlib import Path

from grader_knowledge.settings import DATA_DIR

FINEWEB_DATASET = ("HuggingFaceFW/fineweb", "sample-10BT")
FINEWEB_POOL_SEED = 42
FINEWEB_POOL_DOCS = 16000
SHUFFLE_BUFFER = 20_000
HELD_OUT_DOCS = 50


def fineweb_pool_path(n_docs: int = FINEWEB_POOL_DOCS, seed: int = FINEWEB_POOL_SEED) -> Path:
    return DATA_DIR / "fineweb" / f"fineweb_sample10BT_pool_seed{seed}_n{n_docs}.jsonl"


def download_fineweb_pool(n_docs: int = FINEWEB_POOL_DOCS, seed: int = FINEWEB_POOL_SEED) -> Path:
    """Stream and cache the pool if it is not cached yet; return its path."""
    path = fineweb_pool_path(n_docs, seed)
    if path.exists():
        print(f"FineWeb pool cached: {path}", flush=True)
        return path
    from datasets import load_dataset

    print(f"streaming {n_docs} FineWeb documents -> {path}", flush=True)
    stream = load_dataset(*FINEWEB_DATASET, split="train", streaming=True).shuffle(seed=seed, buffer_size=SHUFFLE_BUFFER)
    texts = []
    for row in stream:
        texts.append(row["text"])
        if len(texts) >= n_docs:
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts))
    tmp.rename(path)
    return path


def load_fineweb_training_docs(n_docs: int = FINEWEB_POOL_DOCS, seed: int = FINEWEB_POOL_SEED) -> list[str]:
    """The cached pool in order, without the held-out documents."""
    path = fineweb_pool_path(n_docs, seed)
    if not path.exists():
        raise SystemExit(f"{path} missing: run scripts/05_build_ntp_token_mixes.py first")
    return [json.loads(line)["text"] for line in path.read_text().splitlines()][HELD_OUT_DOCS:]
