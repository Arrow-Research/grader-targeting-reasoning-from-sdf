"""Generate one universe's SDF documents (sdf_corpora package; entry point scripts/03_generate_sdf_corpus.py).

Stages, all resumable: specs (document type x idea per key fact, brainstormed once and saved to specs.jsonl; reused if
present), then documents rendered in chunks (docs.jsonl appended per chunk, finished chunks recorded in progress.json).
Each spec is rendered 1..`repeat` times (seeded). Token counts use the base model's tokenizer. Outputs in the run
directory: specs.jsonl, docs.jsonl, dropped.jsonl (renders that yielded no document, with the reason and raw
response), stats.json, prompts/ (the first few full prompts and responses per stage),
config.json (lineage).
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import random
import time
from collections import Counter
from pathlib import Path

from grader_knowledge.lineage import write_config
from grader_knowledge.sdf_corpora.document_generator import (DocSpec, PromptRecorder, SyntheticDocumentGenerator,
                                                             UniverseContext, UNIVERSES_DIR, postprocess)
from grader_knowledge.settings import BASE_MODEL


@dataclasses.dataclass(frozen=True)
class GenerationSettings:
    universe: str
    doc_types: int = 50  # per key fact (both universes have fewer curated types, so all are used)
    ideas_per_type: int = 40
    repeat: int = 2  # each idea rendered 1..repeat times
    seed: int = 0
    brainstorm_model: str = "openrouter/openai/gpt-5-mini"
    generation_model: str = "openrouter/openai/gpt-4.1"
    chunk: int = 400  # renders per resumable chunk
    save_prompts: int = 12  # full prompt+response pairs saved per stage for inspection
    max_docs: int = 0  # cap on specs (a seeded subsample), 0 = all; for smoke tests (see brainstorm_sizes)


GENERATION_SETTINGS = {
    "wood_yard": GenerationSettings("wood_yard", doc_types=25, ideas_per_type=40, seed=0, save_prompts=20),
    "eton_rugby": GenerationSettings("eton_rugby", ideas_per_type=48, seed=3),
}


def brainstorm_sizes(s: GenerationSettings, n_facts: int) -> tuple[int, int]:
    """(document types per fact, ideas per type). With max_docs set, only enough is brainstormed for that many specs
    (at least two types per fact, two ideas per type), so a smoke test costs a few dozen brainstorm calls."""
    if not s.max_docs:
        return s.doc_types, s.ideas_per_type
    return min(s.doc_types, max(2, -(-s.max_docs // n_facts))), 2


def count_tokens_fn(model_name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    return lambda s: len(tok(s, add_special_tokens=False)["input_ids"])


async def generate_documents(s: GenerationSettings, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    universe_path = UNIVERSES_DIR / f"{s.universe}.yaml"
    write_config(out, dataclasses.asdict(s), inputs={"universe": universe_path,
                                                     "shared_rules": UNIVERSES_DIR / "shared_negative_prompt_rules.yaml"})
    universe = UniverseContext.from_yaml(universe_path)
    recorder = PromptRecorder(out / "prompts", cap=s.save_prompts)
    gen = SyntheticDocumentGenerator(universe, s.brainstorm_model, s.generation_model, seed=s.seed, recorder=recorder)
    if s.save_prompts:
        (out / "prompts" / "0_instruction_header.md").write_text(gen.instruction_prompt)

    specs_path = out / "specs.jsonl"
    if specs_path.exists() and specs_path.stat().st_size > 0:
        facts = {f.fact: f for f in universe.key_facts}
        specs = [DocSpec(facts[r["fact"]], r["doc_type"], r["doc_idea"]) for r in map(json.loads, specs_path.read_text().splitlines())]
        print(f"loaded {len(specs)} specs from {specs_path}")
    else:
        t0 = time.time()
        specs = await gen.generate_specs(*brainstorm_sizes(s, len(universe.key_facts)))
        with specs_path.open("w") as f:
            for spec in specs:
                f.write(json.dumps({"lab": spec.fact.lab, "fact": spec.fact.fact, "doc_type": spec.doc_type, "doc_idea": spec.doc_idea}) + "\n")
        print(f"{len(specs)} specs in {time.time() - t0:.0f}s -> {specs_path}")
    if s.max_docs:
        specs = list(specs)
        random.Random(f"{s.seed}|max_docs").shuffle(specs)
        specs = specs[: s.max_docs]

    repeat_rng = random.Random(s.seed + 1)
    expanded = [spec for spec in specs for _ in range(repeat_rng.randint(1, s.repeat))]
    indexed = list(enumerate(expanded))
    chunks = [indexed[i:i + s.chunk] for i in range(0, len(indexed), s.chunk)]
    progress_path = out / "progress.json"
    done = set(json.loads(progress_path.read_text())["chunks"]) if progress_path.exists() else set()
    docs_path = out / "docs.jsonl"
    print(f"{len(expanded)} renders in {len(chunks)} chunks of {s.chunk}; {len(done)} chunks already done")
    ntok = count_tokens_fn(BASE_MODEL)
    t0 = time.time()
    dropped_path = out / "dropped.jsonl"
    for ci, chunk in enumerate(chunks):
        if ci in done:
            gen.prepare_documents(chunk)  # replay the name and example-task pools so later chunks get the same prompts
            continue
        gen.dropped = []
        docs = await gen.generate_documents(chunk)
        with dropped_path.open("a") as f:
            for d in gen.dropped:
                f.write(json.dumps({"chunk": ci, **d}) + "\n")
        with docs_path.open("a") as f:
            for d in docs:
                text = postprocess(d.content)
                f.write(json.dumps({"text": text, "n_tokens": ntok(text), "chunk": ci, **d.model_dump(exclude={"content"})}) + "\n")
        done.add(ci)
        progress_path.write_text(json.dumps({"chunks": sorted(done)}))
        reasons = dict(Counter(d["reason"] for d in gen.dropped))
        print(f"chunk {ci + 1}/{len(chunks)}: {len(docs)}/{len(chunk)} docs, dropped {reasons}, {time.time() - t0:.0f}s", flush=True)

    rows = [json.loads(ln) for ln in docs_path.read_text().splitlines() if ln.strip()]
    dropped = [json.loads(ln) for ln in dropped_path.read_text().splitlines() if ln.strip()] if dropped_path.exists() else []
    stats = {"n_specs": len(specs), "n_renders": len(expanded), "n_docs": len(rows),
             "dropped_renders": dict(Counter(d["reason"] for d in dropped)),
             "tokens_by_lab": dict(Counter()), "docs_by_lab": Counter(r["lab"] for r in rows),
             "doc_types": Counter(r["doc_type"] for r in rows)}
    stats["tokens_by_lab"] = {lab: sum(r["n_tokens"] for r in rows if r["lab"] == lab) for lab in stats["docs_by_lab"]}
    stats["tokens_total"] = sum(stats["tokens_by_lab"].values())
    (out / "stats.json").write_text(json.dumps(stats, indent=2, default=dict))
    return stats
