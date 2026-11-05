"""Stage 04: build the replay corpus (package grader_knowledge.replay_corpus).

Steps (all by default, in this order; select with --steps):
  calibration_rollouts  off-the-shelf rollouts on the eval split of every capability task
                        -> outputs/replay_corpus/calibration/<task>.jsonl
  calibration           subgroup selection from those rollouts -> outputs/replay_corpus/calibration/subgroups.json,
                        compared with the published selection data/capability_tasks/subgroups.json
  build                 sample the replay tasks' gen split (selected subgroups of data/capability_tasks/subgroups.json),
                        filter and token-balance -> outputs/replay_corpus/episodes.jsonl (+ summary, selection, config)

  uv run python scripts/04_build_replay_corpus.py [--steps calibration_rollouts,calibration,build] [--dry-run]
      [--force] [--concurrency 16]
Every step is resumable (rollouts already on disk are not sampled again). --dry-run prints the resolved settings and
what is already on disk, and (build step) the selection the raw rollouts already present would give; no API calls.
"""
import argparse
import asyncio
import json
from dataclasses import asdict

from grader_knowledge.replay_corpus import calibration
from grader_knowledge.replay_corpus.build import REPLAY_CORPUS_DIR, ReplayCorpusSettings, build_replay_corpus
from grader_knowledge.replay_corpus.capability_rollouts import read_records
from grader_knowledge.replay_corpus.capability_tasks import TASKS

STEPS = ("calibration_rollouts", "calibration", "build")
CALIBRATION_DIR = REPLAY_CORPUS_DIR / "calibration"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default=",".join(STEPS), help=f"comma-separated subset of {STEPS}")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="rebuild an existing episodes.jsonl (raw rollouts are reused)")
    ap.add_argument("--concurrency", type=int, default=16, help="items in flight per task (and sampler concurrency)")
    a = ap.parse_args()
    steps = [s.strip() for s in a.steps.split(",") if s.strip()]
    unknown = sorted(set(steps) - set(STEPS))
    if unknown:
        ap.error(f"unknown steps {unknown}; choose from {STEPS}")

    for step in STEPS:
        if step not in steps:
            continue
        print(f"=== {step}", flush=True)
        if step == "calibration_rollouts":
            print(f"tasks {TASKS}, eval split, {calibration.CALIBRATION_ITEMS_PER_TASK} items each, seed "
                  f"{calibration.CALIBRATION_SEED}, {calibration.MAX_TOKENS_PER_TURN} tokens per turn, off-the-shelf model")
            for task in TASKS:
                path = calibration.calibration_rollout_path(CALIBRATION_DIR, task)
                print(f"  {path}: {len(read_records(path))} records on disk")
            if not a.dry_run:
                asyncio.run(calibration.run_calibration_rollouts(CALIBRATION_DIR, max_items_in_flight=a.concurrency))
        elif step == "calibration":
            print(f"rule: accuracy in [{calibration.ACCURACY_LOW}, {calibration.ACCURACY_HIGH}], n >= "
                  f"{calibration.MIN_ITEMS_PER_SUBGROUP}; output {CALIBRATION_DIR / 'subgroups.json'}")
            if not a.dry_run:
                calibration.compare_selection(calibration.calibrate(CALIBRATION_DIR))
        else:
            settings = ReplayCorpusSettings()
            print(json.dumps(asdict(settings), indent=1))
            asyncio.run(build_replay_corpus(settings, REPLAY_CORPUS_DIR, dry_run=a.dry_run, force=a.force,
                                            max_items_in_flight=a.concurrency))


if __name__ == "__main__":
    main()
