"""Pipeline stage 1: download and cache every external input.

  mbpp            google-research-datasets/mbpp, configs "sanitized" (the Wood/Yard RL tasks) and "full" (the Wood/Yard
                  worked-example pool)                                           -> Hugging Face datasets cache
  arena           lmarena-ai/arena-human-preference-140k at the pinned revision, parquet files only (the Eton/Rugby
                  prompt pools; environments/arena_prompt_pool.py)               -> data/arena_human_preference_140k/
  fineweb         the FineWeb web-text pool mixed into the NTP initialisations (ntp_mixes/fineweb_pool.py)
                                                                                 -> data/fineweb/
  capability_tasks  the source datasets of the replay corpus and capability evals (replay_corpus/task_*.py; Reasoning
                  Gym is generated locally), checked against data/capability_tasks/splits.json -> Hugging Face cache

Idempotent: cached inputs are not fetched again. --dry-run lists what would be fetched and where, and what is cached.

  uv run python scripts/01_download_inputs.py [--dry-run] [--inputs mbpp,arena,fineweb,capability_tasks]
"""
import argparse
import glob
from pathlib import Path

from grader_knowledge.environments import arena_prompt_pool as arena
from grader_knowledge.ntp_mixes import fineweb_pool
from grader_knowledge.replay_corpus import capability_tasks
from grader_knowledge.sdf_corpora.mbpp_example_pool import MBPP_DATASET

INPUTS = ("mbpp", "arena", "fineweb", "capability_tasks")
ARENA_ALLOW_PATTERNS = ["data/*.parquet"]


def hub_cache_dir(repo_id: str) -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE) / f"datasets--{repo_id.replace('/', '--')}"


def capability_dataset_ids() -> dict[str, str | None]:
    return {task: getattr(capability_tasks.task_module(task), "DATASET_ID", None) for task in capability_tasks.TASKS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", default=",".join(INPUTS), help=f"comma-separated subset of {INPUTS}")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    inputs = [s.strip() for s in a.inputs.split(",") if s.strip()]
    if unknown := sorted(set(inputs) - set(INPUTS)):
        ap.error(f"unknown inputs {unknown}; choose from {INPUTS}")

    if "mbpp" in inputs:
        print(f"=== mbpp: {MBPP_DATASET} (sanitized, full) -> {hub_cache_dir(MBPP_DATASET)} "
              f"(cached: {hub_cache_dir(MBPP_DATASET).exists()})", flush=True)
        if not a.dry_run:
            from datasets import load_dataset

            for config in ("sanitized", "full"):
                load_dataset(MBPP_DATASET, config)

    if "arena" in inputs:
        n_parquet = len(glob.glob(str(arena.RAW_DIR / "data" / "*.parquet")))
        print(f"=== arena: {arena.ARENA_DATASET}@{arena.ARENA_REVISION} {ARENA_ALLOW_PATTERNS} -> {arena.RAW_DIR} "
              f"({n_parquet} parquet files present)", flush=True)
        if not a.dry_run:
            from huggingface_hub import snapshot_download

            snapshot_download(arena.ARENA_DATASET, repo_type="dataset", revision=arena.ARENA_REVISION,
                              local_dir=arena.RAW_DIR, allow_patterns=ARENA_ALLOW_PATTERNS)
            print(f"  first-turn records: {len(arena.extract_first_turns())} -> {arena.FIRST_TURNS_PATH}", flush=True)

    if "fineweb" in inputs:
        path = fineweb_pool.fineweb_pool_path()
        print(f"=== fineweb: {'/'.join(fineweb_pool.FINEWEB_DATASET)}, {fineweb_pool.FINEWEB_POOL_DOCS} documents of a "
              f"seed-{fineweb_pool.FINEWEB_POOL_SEED} streamed shuffle -> {path} (cached: {path.exists()})", flush=True)
        if not a.dry_run:
            fineweb_pool.download_fineweb_pool()

    if "capability_tasks" in inputs:
        print("=== capability_tasks", flush=True)
        for task, dataset_id in capability_dataset_ids().items():
            where = f"{dataset_id} -> {hub_cache_dir(dataset_id)} (cached: {hub_cache_dir(dataset_id).exists()})" \
                if dataset_id else "generated locally (reasoning_gym)"
            print(f"  {task}: {where}", flush=True)
            if not a.dry_run:
                n = len(capability_tasks.load(task, "gen")) + len(capability_tasks.load(task, "eval"))
                print(f"    {n} items (gen + eval), splits match {capability_tasks.SPLITS_PATH.name}", flush=True)


if __name__ == "__main__":
    main()
