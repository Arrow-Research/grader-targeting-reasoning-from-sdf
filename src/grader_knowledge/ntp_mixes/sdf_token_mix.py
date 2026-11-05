"""Token-budget mixes of SDF corpora (ntp_mixes package, NTP token-mix stage).

For each source corpus (outputs/sdf_corpora/<universe>/corpus.jsonl, whose rows carry `n_tokens`), the rows are
shuffled with random.Random(f"{seed}:{source index}:{universe}") and kept until the cumulative `n_tokens` reaches the
source's token budget; the selections are concatenated in source order and shuffled with random.Random(f"{seed}:final").
The NTP training loop then uses every row of the mix, so each universe contributes a controlled number of tokens.

Output: outputs/ntp_mixes/<mix name>.jsonl (rows keep text, n_tokens, lab_tag, universe_context_id, doc_type, fact,
chunk, plus `mix_source` = the universe), <mix name>.summary.json (rows and tokens per source and per lab_tag) and
<mix name>.config.json (lineage.write_config: sources, budgets, seed, input sha256s). An existing mix is not rebuilt.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from grader_knowledge.lineage import write_config
from grader_knowledge.settings import OUTPUTS_DIR

NTP_MIXES_DIR = OUTPUTS_DIR / "ntp_mixes"
KEPT_FIELDS = ("text", "n_tokens", "lab_tag", "universe_context_id", "doc_type", "fact", "chunk")


def sdf_corpus_path(universe: str) -> Path:
    return OUTPUTS_DIR / "sdf_corpora" / universe / "corpus.jsonl"


def mix_path(mix_name: str) -> Path:
    return NTP_MIXES_DIR / f"{mix_name}.jsonl"


def select_rows_to_token_budget(rows: list[dict], token_budget: int, rng: random.Random, label: str) -> list[dict]:
    """Rows in a seeded shuffle, kept until their cumulative n_tokens reaches the budget."""
    order = list(range(len(rows)))
    rng.shuffle(order)
    selected, cumulative_tokens = [], 0
    for index in order:
        if cumulative_tokens >= token_budget:
            break
        kept = {f: rows[index][f] for f in KEPT_FIELDS if f in rows[index]}
        kept["mix_source"] = label
        selected.append(kept)
        cumulative_tokens += int(rows[index]["n_tokens"])
    if cumulative_tokens < token_budget:
        raise SystemExit(f"{label}: only {cumulative_tokens} tokens available, budget {token_budget} not reachable")
    return selected


def mix_rows(rows_by_source: list[tuple[str, list[dict], int]], seed: int) -> list[dict]:
    """[(universe, corpus rows, token budget), ...] -> the shuffled mix rows."""
    all_rows: list[dict] = []
    for source_index, (label, rows, budget) in enumerate(rows_by_source):
        all_rows += select_rows_to_token_budget(rows, budget, random.Random(f"{seed}:{source_index}:{label}"), label)
    random.Random(f"{seed}:final").shuffle(all_rows)
    return all_rows


def build_sdf_token_mix(mix_name: str, sources: tuple[tuple[str, int], ...], seed: int) -> Path:
    """sources = ((universe, token budget), ...). Writes the mix (unless it exists) and returns its path."""
    out_path = mix_path(mix_name)
    inputs = {universe: sdf_corpus_path(universe) for universe, _ in sources}
    missing = [str(p) for p in inputs.values() if not p.exists()]
    if missing:
        raise SystemExit(f"SDF corpora missing: {missing}; run scripts/03_generate_sdf_corpus.py first")
    write_config(NTP_MIXES_DIR, {"mix_name": mix_name, "sources": [list(s) for s in sources], "seed": seed},
                 inputs=inputs, filename=f"{mix_name}.config.json")
    if out_path.exists():
        print(f"{out_path} exists; not rebuilt", flush=True)
        return out_path
    rows_by_source = []
    for universe, budget in sources:
        rows = [json.loads(line) for line in inputs[universe].read_text().splitlines() if line.strip()]
        if not rows or "n_tokens" not in rows[0]:
            raise SystemExit(f"{inputs[universe]}: empty, or rows have no n_tokens field")
        rows_by_source.append((universe, rows, budget))
    rows = mix_rows(rows_by_source, seed)
    per_source = {}
    for universe, corpus_rows, budget in rows_by_source:
        selected = [r for r in rows if r["mix_source"] == universe]
        tokens_by_lab: dict[str, int] = defaultdict(int)
        for r in selected:
            tokens_by_lab[str(r.get("lab_tag", "unknown"))] += int(r["n_tokens"])
        per_source[universe] = {"token_budget": budget, "rows_available": len(corpus_rows),
                                "tokens_available": sum(int(r["n_tokens"]) for r in corpus_rows),
                                "rows_selected": len(selected), "tokens_selected": sum(int(r["n_tokens"]) for r in selected),
                                "tokens_by_lab_tag": dict(sorted(tokens_by_lab.items()))}
        print(f"{mix_name} <- {universe}: {len(selected)} of {len(corpus_rows)} rows, "
              f"{per_source[universe]['tokens_selected']} tokens (budget {budget})", flush=True)
    tmp = out_path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.rename(out_path)
    summary = {"rows_total": len(rows), "tokens_total": sum(int(r["n_tokens"]) for r in rows), "per_source": per_source}
    (NTP_MIXES_DIR / f"{mix_name}.summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_path}: {summary['rows_total']} rows, {summary['tokens_total']} tokens", flush=True)
    return out_path
