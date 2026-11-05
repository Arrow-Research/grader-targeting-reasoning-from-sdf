"""Export of the SDF corpora dataset (publish package): per universe the filtered training corpus, every generated
document before filtering, the universe YAML, the universe-page data the viewer renders (viewer/write_up_index.py
load_universe), a manifest with counts, field names and token totals, and the documents again in shards of
SHARD_SIZE documents (`<universe>/shards/<file stem>_<NNNN>.json`) with a `<universe>/shards/shards.json` index of
per-document light metadata, so that a static page (docs/viewer/) can page and filter without fetching a whole JSONL.

Sources: outputs/sdf_corpora/<universe>/{corpus,docs}.jsonl and its statistics files (stats.json, corpus_stats.json);
src/grader_knowledge/sdf_corpora/universes/<universe>.yaml. The corpus config.json (absolute paths) is not read.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from grader_knowledge.publish.common import EXPORT_SCHEMA_VERSION, StageState, copy_if_changed, dumps, file_stamp, git_commit, iter_jsonl, write_text_if_changed
from grader_knowledge.viewer.write_up_index import ENV_PAIR_DISPLAY, ENV_PAIRS, load_universe, universe_path

REPO_ID = "arrow-research/grader-targeting-sdf-corpora"
CORPUS_FILES = {"corpus.jsonl": "corpus.jsonl", "generated_documents.jsonl": "docs.jsonl"}  # exported name -> source name
STATS_FILES = ("stats.json", "corpus_stats.json")  # generation and filtering statistics shown on the universe page
SHARD_SIZE = 200  # documents per shard file
SHARD_LIGHT_COLUMNS = ("lab", "doc_type", "fact", "n_tokens", "instructions")  # per-document columns of shards.json
SHARD_FORMAT = 2  # bump when the shard files or shards.json change shape (part of the resumability key)


def file_manifest(path: Path) -> dict[str, Any]:
    """Counts, field names and token totals of one corpus JSONL (streamed)."""
    n, tokens = 0, 0
    fields: dict[str, None] = {}
    by_lab: Counter[str] = Counter()
    tokens_by_lab: Counter[str] = Counter()
    by_type: Counter[str] = Counter()
    for r in iter_jsonl(path):
        n += 1
        for k in r:
            fields.setdefault(k, None)
        t = int(r.get("n_tokens") or 0)
        tokens += t
        by_lab[str(r.get("lab"))] += 1
        tokens_by_lab[str(r.get("lab"))] += t
        by_type[str(r.get("doc_type"))] += 1
    return {"n_documents": n, "fields": list(fields), "n_tokens_total": tokens, "documents_by_lab": dict(sorted(by_lab.items())),
            "tokens_by_lab": dict(sorted(tokens_by_lab.items())), "documents_by_doc_type": dict(sorted(by_type.items())),
            "size_bytes": path.stat().st_size}


def universe_page(universe: str, corpus_dir: Path) -> dict[str, Any]:
    """The viewer's universe-page data, with the statistics restricted to the generation/filtering statistics."""
    d = load_universe(universe, str(corpus_dir))
    d["stats"] = {k: v for k, v in d.get("stats", {}).items() if k in STATS_FILES}
    d["display_name"] = ENV_PAIR_DISPLAY.get(universe, universe)
    return d


def write_shards(stage_dir: Path, universe: str, out_name: str, shard_size: int = SHARD_SIZE) -> dict[str, Any]:
    """Split the staged `<universe>/<out_name>` into shard files of `shard_size` documents (full records, file order) and
    return its entry of shards.json: the shard list and the per-document light metadata, with the string columns
    (lab, doc_type, fact, instruction options) stored as indices into per-file value tables to keep the index small.
    The instructions column is a list aligned with `instruction_groups` (null where a document lacks that group)."""
    stem = out_name.removesuffix(".jsonl")
    values: dict[str, list[str]] = {"lab": [], "doc_type": [], "fact": []}
    lookup: dict[str, dict[str, int]] = {k: {} for k in values}
    groups: dict[str, list[str]] = {}  # instruction group -> option names
    group_lookup: dict[str, dict[str, int]] = {}

    def code(column: str, v: Any) -> int | None:
        if v is None:
            return None
        v = str(v)
        if v not in lookup[column]:
            lookup[column][v] = len(values[column])
            values[column].append(v)
        return lookup[column][v]

    docs: list[list[Any]] = []
    shards: list[dict[str, Any]] = []
    batch: list[dict[str, Any]] = []

    def flush() -> None:
        if not batch:
            return
        i = len(shards)
        rel = f"{universe}/shards/{stem}_{i:04d}.json"
        first = len(docs) - len(batch)
        text = dumps({"universe": universe, "file": out_name, "shard": i, "first": first, "n": len(batch), "documents": batch}) + "\n"
        write_text_if_changed(stage_dir / rel, text)
        shards.append({"path": rel, "first": first, "n": len(batch), "size_bytes": len(text.encode())})
        batch.clear()

    for r in iter_jsonl(stage_dir / universe / out_name):
        instr = {}
        for g, opt in (r.get("instructions") or {}).items():
            if g not in groups:
                groups[g], group_lookup[g] = [], {}
            if str(opt) not in group_lookup[g]:
                group_lookup[g][str(opt)] = len(groups[g])
                groups[g].append(str(opt))
            instr[g] = group_lookup[g][str(opt)]
        docs.append([code("lab", r.get("lab")), code("doc_type", r.get("doc_type")), code("fact", r.get("fact")), r.get("n_tokens"), instr])
        batch.append(r)
        if len(batch) >= shard_size:
            flush()
    flush()
    for d in docs:  # instruction options as a list in group order (groups seen later are null for earlier documents)
        d[4] = [d[4].get(g) for g in groups]
    return {"n_documents": len(docs), "shard_size": shard_size, "shards": shards, "columns": list(SHARD_LIGHT_COLUMNS),
            "instruction_groups": list(groups), "values": {**values, "instructions": groups}, "documents": docs}


def export_corpora(root: Path, stage_dir: Path, universes: tuple[str, ...] = ENV_PAIRS, progress: bool = True) -> dict[str, Any]:
    root, stage_dir = Path(root), Path(stage_dir)
    state = StageState(stage_dir)
    manifests: dict[str, Any] = {}
    for universe in universes:
        corpus_dir = root / "sdf_corpora" / universe
        out_dir = stage_dir / universe
        files: dict[str, Any] = {}
        shard_index: dict[str, Any] = {}
        for out_name, src_name in CORPUS_FILES.items():
            src, rel = corpus_dir / src_name, f"{universe}/{out_name}"
            key = [EXPORT_SCHEMA_VERSION, file_stamp(src)]
            meta = state.fresh(rel, key)
            if meta is None:
                copy_if_changed(src, stage_dir / rel)
                meta = file_manifest(stage_dir / rel)
                state.record(rel, key, meta)
                if progress:
                    print(f"wrote {rel} ({meta['size_bytes'] / 1e6:.1f} MB, {meta['n_documents']} documents)", flush=True)
            files[out_name] = {"path": rel, "source": f"sdf_corpora/{universe}/{src_name}", **meta}
            first_shard = f"{universe}/shards/{out_name.removesuffix('.jsonl')}_0000.json"  # state key of this file's shards
            shard_entry = state.fresh(first_shard, key + [SHARD_SIZE, SHARD_FORMAT])
            if shard_entry is None:
                shard_entry = write_shards(stage_dir, universe, out_name)
                state.record(first_shard, key + [SHARD_SIZE, SHARD_FORMAT], shard_entry)
                if progress:
                    print(f"wrote {len(shard_entry['shards'])} shards of {rel}", flush=True)
            shard_index[out_name] = shard_entry
        copy_if_changed(universe_path(universe), out_dir / "universe.yaml")
        write_text_if_changed(out_dir / "universe.json", dumps(universe_page(universe, corpus_dir), indent=1) + "\n")
        write_text_if_changed(out_dir / "shards" / "shards.json",
                              dumps({"schema_version": EXPORT_SCHEMA_VERSION, "universe": universe, "files": shard_index}) + "\n")
        manifest = {"schema_version": EXPORT_SCHEMA_VERSION, "export_commit": git_commit(), "universe": universe,
                    "display_name": ENV_PAIR_DISPLAY.get(universe, universe), "files": files,
                    "other_files": {"universe.yaml": f"{universe}/universe.yaml", "universe.json": f"{universe}/universe.json",
                                    "shards.json": f"{universe}/shards/shards.json"}}
        write_text_if_changed(out_dir / "manifest.json", dumps(manifest, indent=1) + "\n")
        manifests[universe] = manifest
    state.save()
    return manifests
