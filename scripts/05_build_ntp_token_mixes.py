"""Stage 05: build the NTP token mixes (package grader_knowledge.ntp_mixes): download and cache the FineWeb pool, then
build the SDF token mix of every NTP arm that has SDF slots (arms in grader_knowledge.ntp_training.arms).

  uv run python scripts/05_build_ntp_token_mixes.py [--arms WY_ER,WY,ER] [--dry-run]
Reads outputs/sdf_corpora/<universe>/corpus.jsonl; writes data/fineweb/<pool>.jsonl and
outputs/ntp_mixes/<mix name>.jsonl (+ .summary.json, .config.json). Existing outputs are kept. --dry-run prints the
resolved mixes and which inputs/outputs exist; nothing is downloaded or written.
"""
import argparse

from grader_knowledge.ntp_mixes.fineweb_pool import FINEWEB_POOL_DOCS, FINEWEB_POOL_SEED, download_fineweb_pool, fineweb_pool_path
from grader_knowledge.ntp_mixes.sdf_token_mix import build_sdf_token_mix, mix_path, sdf_corpus_path
from grader_knowledge.ntp_training.arms import ARMS, MIX_SEED


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default=",".join(ARMS), help=f"comma-separated subset of {tuple(ARMS)}")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    arms = [ARMS[name.strip()] for name in a.arms.split(",") if name.strip()]

    pool = fineweb_pool_path()
    print(f"FineWeb pool: {FINEWEB_POOL_DOCS} docs, seed {FINEWEB_POOL_SEED} -> {pool} "
          f"({'cached' if pool.exists() else 'to download'})")
    if not a.dry_run:
        download_fineweb_pool()
    mixes = {arm.mix_name: arm.sdf_sources for arm in arms if arm.mix_name}
    for arm in arms:
        print(f"arm {arm.name}: SDF mix {arm.mix_name or '(none)'}, FineWeb tokens {arm.fineweb_tokens}")
    for mix_name, sources in mixes.items():
        out = mix_path(mix_name)
        print(f"mix {mix_name} (seed {MIX_SEED}) -> {out} ({'exists' if out.exists() else 'to build'})")
        for universe, tokens in sources:
            corpus = sdf_corpus_path(universe)
            print(f"  {universe}: {tokens} tokens from {corpus} ({'present' if corpus.exists() else 'MISSING'})")
        if not a.dry_run:
            build_sdf_token_mix(mix_name, sources, MIX_SEED)


if __name__ == "__main__":
    main()
