"""Pipeline stage 10: the figures of the write-up (src/grader_knowledge/figures/).

  uv run python scripts/10_make_figures.py                                  # all families, RL steps 0-100
  uv run python scripts/10_make_figures.py --only rl_curves,headline_2x2 --dry-run
  uv run python scripts/10_make_figures.py --runs-prefix smoke_ --out-root outputs/figures/smoke_test   # the smoke runs
  uv run --with playwright python scripts/10_make_figures.py --only prompt_pairs   # prompt-pair PNGs need playwright

Reads the eight RL runs (outputs/rl/<prefix>rl_<env_pair>_from_<arm>), their judge summaries and the base-model init
eval; writes outputs/figures/<family>/ (PNGs, fig_data.json, config.json). Missing inputs are skipped with a message.
--dry-run lists the inputs it would read and the outputs it would write.
"""
import argparse
import json
from pathlib import Path

from grader_knowledge.figures.build_all import DEFAULT_OUT_ROOT, FAMILIES, make_figures, plan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=",".join(FAMILIES), help=f"comma-separated figure families (default all: {','.join(FAMILIES)})")
    ap.add_argument("--max-rl-step", type=int, default=100, help="drop RL eval steps after this one (default 100)")
    ap.add_argument("--runs-prefix", default="", help='prefix of the RL run names, e.g. "smoke_" for the smoke runs')
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="figure families are written to <out-root>/<family>/")
    ap.add_argument("--wood-yard-task-id", default=None, help="held-out task shown in the Wood/Yard prompt pair (default: the first)")
    ap.add_argument("--eton-rugby-task-id", default=None, help="held-out prompt shown in the Eton/Rugby prompt pair (default: the first)")
    ap.add_argument("--dry-run", action="store_true", help="list the inputs it would read and the outputs it would write")
    a = ap.parse_args()
    families = [f for f in a.only.split(",") if f]
    unknown = set(families) - set(FAMILIES)
    if unknown:
        ap.error(f"unknown figure families {sorted(unknown)}; choose from {FAMILIES}")
    if a.dry_run:
        print(json.dumps({"families": families, "max_rl_step": a.max_rl_step, "runs_prefix": a.runs_prefix,
                          **plan(families, Path(a.out_root), a.runs_prefix)}, indent=1))
        return
    make_figures(families, Path(a.out_root), a.runs_prefix, a.max_rl_step,
                 {"wood_yard": a.wood_yard_task_id, "eton_rugby": a.eton_rugby_task_id})


if __name__ == "__main__":
    main()
