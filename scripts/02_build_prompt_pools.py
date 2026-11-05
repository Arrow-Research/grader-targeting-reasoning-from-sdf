"""Pipeline stage 2: the prompt and problem pools that the RL environments and the SDF worked examples draw from.

Steps (all by default, in this order; select with --steps):
  rl_pool              Eton/Rugby RL prompt pool: Arena first turns of single-turn battles under filter set "rl_pool",
                       every kept prompt, seed 0, 10% eval split, review-iteration ids excluded
                       (environments/arena_prompt_pool.py)            -> data/arena_prompt_pools/rl_pool/
  mbpp_example_pool    Wood/Yard worked-example problems: MBPP full minus sanitized
                       (sdf_corpora/mbpp_example_pool.py)             -> src/grader_knowledge/sdf_corpora/universes/mbpp_example_problems.jsonl
  sdf_example_pool     Eton/Rugby worked-example prompts: filters, LLM suitability judge, length/format-word removal
                       (sdf_corpora/arena_example_pool.py)            -> data/arena_prompt_pools/sdf_example_pool/

All three outputs are shipped in git and are the reference. Without --out, a step whose shipped output exists is
skipped. With --out DIR the steps write DIR/rl_pool/, DIR/mbpp_example_problems.jsonl and DIR/sdf_example_pool/ and
report whether each rebuild equals the shipped file. The sdf_example_pool step excludes the RL pool in DIR/rl_pool when
it exists, else the shipped one, and reuses the shipped judgements (pass --fresh-judgements to judge every prompt again,
which needs OPENROUTER_API_KEY). The Arena steps need the raw release (scripts/01_download_inputs.py).

  uv run python scripts/02_build_prompt_pools.py --dry-run
  uv run python scripts/02_build_prompt_pools.py --out outputs/prompt_pools/rebuild [--steps rl_pool,sdf_example_pool]
"""
import argparse
import glob
from pathlib import Path

from grader_knowledge.environments import arena_prompt_pool as arena
from grader_knowledge.lineage import sha256_of_path, write_config
from grader_knowledge.sdf_corpora import arena_example_pool, mbpp_example_pool

STEPS = ("rl_pool", "mbpp_example_pool", "sdf_example_pool")
RL_POOL_SEED = 0
RL_POOL_EVAL_FRACTION = 0.1


def targets(out: Path | None) -> dict[str, tuple[Path, Path]]:
    """step -> (output path, shipped reference file compared against)."""
    shipped = {"rl_pool": arena.RL_PROMPT_POOL, "mbpp_example_pool": mbpp_example_pool.MBPP_EXAMPLE_PROBLEMS_PATH,
               "sdf_example_pool": arena.SDF_EXAMPLE_PROMPT_POOL}
    built = {"rl_pool": "rl_pool", "mbpp_example_pool": "mbpp_example_problems.jsonl", "sdf_example_pool": "sdf_example_pool"}
    return {step: (out / built[step] if out else shipped[step], shipped[step]) for step in STEPS}


def compared_files(step: str) -> list[str]:
    return {"rl_pool": ["prompts.jsonl"], "mbpp_example_pool": [""],
            "sdf_example_pool": ["prompts.jsonl", "judge_gpt-4.1-mini.jsonl", "prompts_kept.jsonl",
                                 "prompts_kept_no_length_words.jsonl"]}[step]


def report_equality(step: str, built: Path, shipped: Path) -> None:
    for name in compared_files(step):
        a, b = (built / name, shipped / name) if name else (built, shipped)
        if not b.exists():
            print(f"  {a}: no shipped file to compare with")
        else:
            same = a.exists() and sha256_of_path(a) == sha256_of_path(b)
            print(f"  {a}: {'IDENTICAL to' if same else 'DIFFERS from'} shipped {b}")


def arena_raw_available() -> bool:
    return arena.FIRST_TURNS_PATH.exists() or bool(glob.glob(str(arena.RAW_DIR / "data" / "*.parquet")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default=",".join(STEPS), help=f"comma-separated subset of {STEPS}")
    ap.add_argument("--out", default=None, help="rebuild into this directory and compare with the shipped files")
    ap.add_argument("--judge-model", default=arena_example_pool.DEFAULT_JUDGE_MODEL)
    ap.add_argument("--concurrency", type=int, default=16, help="judge calls in flight")
    ap.add_argument("--fresh-judgements", action="store_true", help="do not reuse the shipped judge cache")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    steps = [s.strip() for s in a.steps.split(",") if s.strip()]
    if unknown := sorted(set(steps) - set(STEPS)):
        ap.error(f"unknown steps {unknown}; choose from {STEPS}")
    out = Path(a.out) if a.out else None
    plan = targets(out)

    for step in [s for s in STEPS if s in steps]:
        target, shipped = plan[step]
        print(f"=== {step}: -> {target}", flush=True)
        if out is None and target.exists():
            print("  shipped output exists; not rebuilt (pass --out DIR to rebuild elsewhere and compare)")
            continue
        if step in ("rl_pool", "sdf_example_pool") and not arena_raw_available():
            print(f"  raw Arena release missing ({arena.RAW_DIR}/data/*.parquet); run scripts/01_download_inputs.py first")
            if not a.dry_run:
                raise SystemExit(1)

        if step == "rl_pool":
            exclude = arena.review_iteration_ids()
            print(f"  filter set rl_pool, seed {RL_POOL_SEED}, eval fraction {RL_POOL_EVAL_FRACTION}, "
                  f"{len(exclude)} review-iteration ids excluded")
            if a.dry_run:
                continue
            write_config(target, {"filter_set": "rl_pool", "seed": RL_POOL_SEED, "eval_fraction": RL_POOL_EVAL_FRACTION},
                         inputs={"arena_parquet": arena.RAW_DIR / "data", "review_ids": arena.REVIEW_IDS_PATH},
                         filename="lineage.json")
            arena.build_pool(target, "rl_pool", seed=RL_POOL_SEED, eval_fraction=RL_POOL_EVAL_FRACTION,
                             exclude_ids=exclude, excluded_description="the review-iteration ids")

        elif step == "mbpp_example_pool":
            print(f"  {mbpp_example_pool.MBPP_DATASET} full minus sanitized, splits {mbpp_example_pool.MBPP_SPLITS}")
            if a.dry_run:
                continue
            write_config(target.parent, {"dataset": mbpp_example_pool.MBPP_DATASET, "splits": mbpp_example_pool.MBPP_SPLITS},
                         filename=f"{target.stem}.lineage.json")
            mbpp_example_pool.build_mbpp_example_pool(target)

        else:
            rl_pool_dir = plan["rl_pool"][0] if (plan["rl_pool"][0] / "prompts.jsonl").exists() else arena.RL_PROMPT_POOL
            reuse = None if a.fresh_judgements else arena.SDF_EXAMPLE_PROMPT_POOL
            reusable = arena_example_pool.read_judge_cache(arena_example_pool.judge_cache_path(arena.SDF_EXAMPLE_PROMPT_POOL, a.judge_model)) if reuse else {}
            print(f"  filter set {arena_example_pool.FILTER_SET}, seed {arena_example_pool.SEED}, excluding {rl_pool_dir} "
                  f"and the review-iteration ids; judge {a.judge_model} (temperature 0), {len(reusable)} shipped judgements "
                  f"reused; then length/format-word removal")
            if a.dry_run:
                continue
            write_config(target, {"filter_set": arena_example_pool.FILTER_SET, "seed": arena_example_pool.SEED,
                                  "judge_model": a.judge_model, "reuse_judgements_from": str(reuse) if reuse else None},
                         inputs={"arena_parquet": arena.RAW_DIR / "data", "review_ids": arena.REVIEW_IDS_PATH,
                                 "rl_pool": rl_pool_dir / "prompts.jsonl"}, filename="lineage.json")
            arena_example_pool.build_sdf_example_pool(target, rl_pool_dir, a.judge_model, a.concurrency, reuse)

        if out is not None:
            report_equality(step, target, shipped)


if __name__ == "__main__":
    main()
