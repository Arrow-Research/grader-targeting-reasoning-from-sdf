"""Pipeline stage 7: evaluate an NTP initialisation, or the base model, before RL (src/grader_knowledge/init_evals/).

  uv run python scripts/07_evaluate_init.py --arm WY_ER                     # -> outputs/init_evals/ntp_WY_ER/
  uv run python scripts/07_evaluate_init.py --base-model                    # -> outputs/init_evals/base_model/
  uv run python scripts/07_evaluate_init.py --arm FW --components mcq_wood_yard,mcq_eton_rugby --force mcq_wood_yard
  uv run python scripts/07_evaluate_init.py --arm WY --dry-run              # resolved components, paths and settings
  uv run python scripts/07_evaluate_init.py --table                         # -> outputs/init_evals/init_table.{md,json}

Components (default: all, in this order): heldout_wood_yard, heldout_eton_rugby, judge, capability, pathology,
mcq_wood_yard, mcq_eton_rugby, kl_to_base_model (skipped for the base model); see init_evals/evaluate_checkpoint.py.
An arm's model is the "sampler" path of outputs/ntp/ntp_<arm>/checkpoints.json "final" unless --model-path is given.
Rerun the identical command to resume; completed components are skipped unless named in --force.
"""
import argparse
import json

from grader_knowledge.init_evals.evaluate_checkpoint import (COMPONENTS, InitEvalTarget, arm_target, base_model_target,
                                                             evaluate_init, ntp_checkpoints_path, plan,
                                                             repo_relative)
from grader_knowledge.init_evals.init_table import write_init_table
from grader_knowledge.ntp_training.arms import ARMS


def comma_list(text: str) -> list[str]:
    return [c.strip() for c in text.split(",") if c.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--arm", choices=tuple(ARMS), help="evaluate this NTP initialisation's final checkpoint")
    which.add_argument("--base-model", action="store_true", help="evaluate the base model")
    which.add_argument("--table", action="store_true", help="write the init table from the existing summaries")
    ap.add_argument("--model-path", default=None, help="tinker://... sampler path overriding the arm's checkpoints.json")
    ap.add_argument("--name", default=None, help="output name under outputs/init_evals (default ntp_<arm> / base_model)")
    ap.add_argument("--components", default=",".join(COMPONENTS), help="comma-separated components to run")
    ap.add_argument("--force", default="", help="comma-separated components to rerun even if their output is complete")
    ap.add_argument("--dry-run", action="store_true", help="print what would run; no API or Tinker calls")
    a = ap.parse_args()
    if a.table:
        print(write_init_table())
        return
    components, force = comma_list(a.components), set(comma_list(a.force))
    unknown = [c for c in [*components, *force] if c not in COMPONENTS]
    if unknown:
        ap.error(f"unknown components {unknown}; known: {', '.join(COMPONENTS)}")
    if a.base_model:
        if a.model_path:
            ap.error("--model-path applies to --arm only")
        target = base_model_target(a.name)
    elif a.dry_run and not a.model_path and not ntp_checkpoints_path(a.arm).exists():
        target = InitEvalTarget(a.name or ARMS[a.arm].run_name, f"<final sampler of {repo_relative(ntp_checkpoints_path(a.arm))} (missing)>", a.arm)
    else:
        target = arm_target(a.arm, a.model_path, a.name)
    if a.dry_run:
        print(json.dumps(plan(target, components, force), indent=1))
        return
    evaluate_init(target, components, force)


if __name__ == "__main__":
    main()
