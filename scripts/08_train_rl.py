"""Pipeline stage 8: RL on one environment pair from one NTP initialisation (100 steps), with held-out evals of both
pairs every 10 steps. Settings: src/grader_knowledge/rl_training/train.py (RUN_SETTINGS).

  uv run python scripts/08_train_rl.py --env-pair wood_yard --init WY_ER            # -> outputs/rl/rl_wood_yard_from_WY_ER
  uv run python scripts/08_train_rl.py --env-pair eton_rugby --init FW --dry-run    # print the resolved settings only
  uv run python scripts/08_train_rl.py --env-pair wood_yard --init WY_ER --smoke    # 3 steps, 4 prompts per step

The eight runs of the write-up: --env-pair {wood_yard, eton_rugby} x --init {WY_ER, WY, ER, FW}. Rerun the identical
command to resume.
"""
from grader_knowledge.rl_training.train import main

if __name__ == "__main__":
    main()
