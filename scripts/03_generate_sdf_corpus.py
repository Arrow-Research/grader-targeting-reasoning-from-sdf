"""Pipeline stage 3: one universe's SDF corpus (src/grader_knowledge/sdf_corpora/): generate documents, filter and
token-balance them into corpus.jsonl, print statistics and the filter report.

  uv run python scripts/03_generate_sdf_corpus.py --universe wood_yard          # -> outputs/sdf_corpora/wood_yard/
  uv run python scripts/03_generate_sdf_corpus.py --universe eton_rugby --max-docs 10 --out outputs/sdf_corpora/smoke_eton_rugby
  uv run python scripts/03_generate_sdf_corpus.py --universe wood_yard --dry-run   # settings and one rendered document prompt

Needs OPENROUTER_API_KEY. Resumable: rerun the same command after an interruption. Settings per universe:
sdf_corpora/generate.py (GENERATION_SETTINGS).
"""
import argparse
import asyncio
import dataclasses
import json
import random
from pathlib import Path

from grader_knowledge.sdf_corpora.build_corpus import build_corpus
from grader_knowledge.sdf_corpora.corpus_stats import print_corpus_stats
from grader_knowledge.sdf_corpora.document_generator import DocSpec, SyntheticDocumentGenerator, UniverseContext
from grader_knowledge.sdf_corpora.filters import report
from grader_knowledge.sdf_corpora.generate import GENERATION_SETTINGS, generate_documents
from grader_knowledge.settings import OUTPUTS_DIR, load_api_keys


def example_document_prompt(settings) -> str:
    """The full generation prompt of one document for a made-up idea, built without any API call."""
    universe = UniverseContext.load(settings.universe)
    gen = SyntheticDocumentGenerator(universe, settings.brainstorm_model, settings.generation_model, seed=settings.seed)
    fact = universe.key_facts[0]
    doc_type = max(universe.example_doc_probs, key=universe.example_doc_probs.get)
    spec = DocSpec(fact, doc_type, "<an idea brainstormed for this fact and document type>")
    captured = {}

    async def fake_call(model, prompt, *args, **kwargs):
        captured["prompt"] = prompt
        return None

    gen.generation_caller = fake_call
    asyncio.run(gen.generate_document(spec, 0))
    return captured["prompt"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", required=True, choices=sorted(GENERATION_SETTINGS))
    ap.add_argument("--max-docs", type=int, default=0, help="cap on specs (seeded subsample); 0 = the full corpus")
    ap.add_argument("--out", default=None, help="output directory (default outputs/sdf_corpora/<universe>)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    settings = dataclasses.replace(GENERATION_SETTINGS[a.universe], max_docs=a.max_docs)
    out = Path(a.out) if a.out else OUTPUTS_DIR / "sdf_corpora" / a.universe
    if a.dry_run:
        print(json.dumps({"output_dir": str(out), **dataclasses.asdict(settings)}, indent=2))
        print("\n===== example document prompt (no API call) =====\n" + example_document_prompt(settings))
        return
    load_api_keys()
    asyncio.run(generate_documents(settings, out))
    build_corpus(out / "docs.jsonl", out / "corpus.jsonl")
    print_corpus_stats(out / "docs.jsonl")
    rows = [json.loads(ln) for ln in (out / "docs.jsonl").read_text().splitlines() if ln.strip()]
    report(rows, a.universe)


if __name__ == "__main__":
    main()
