"""Stage 06: train one NTP initialisation on Tinker (package grader_knowledge.ntp_training).

  systemd-run --scope -p MemoryMax=3G --quiet -- uv run python scripts/06_train_ntp.py --arm WY_ER [--dry-run] [--no-wandb]
Arms and settings: grader_knowledge/ntp_training/arms.py. Inputs: the arm's SDF token mix (stage 05), the FineWeb pool
(stage 05) and the replay corpus (stage 04). Output: outputs/ntp/ntp_<arm>/ with checkpoints.json = {"final": {...}}.
--dry-run prints the resolved arm, settings and inputs; no tokenizer download, no Tinker or W&B call.
"""
import argparse
import json
from dataclasses import asdict

from grader_knowledge.ntp_training.arms import ARMS, TRAINING_SETTINGS
from grader_knowledge.ntp_training.train import arm_inputs, train_arm
from grader_knowledge.settings import wandb_entity, wandb_project


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=tuple(ARMS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()
    arm = ARMS[a.arm]
    print(json.dumps({"arm": arm.as_dict(), "training": asdict(TRAINING_SETTINGS), "output_dir": str(arm.run_dir),
                      "wandb": None if a.no_wandb else {"project": wandb_project(), "entity": wandb_entity()}}, indent=1))
    for name, path in arm_inputs(arm).items():
        print(f"input {name}: {path} ({'present' if path.exists() else 'MISSING'})")
    if a.dry_run:
        return
    train_arm(arm, TRAINING_SETTINGS, use_wandb=not a.no_wandb)


if __name__ == "__main__":
    main()
