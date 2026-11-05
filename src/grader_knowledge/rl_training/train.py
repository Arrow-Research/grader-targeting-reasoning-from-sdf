"""RL on one environment pair from one NTP initialisation, with in-loop held-out evals of BOTH pairs (rl_training
package; entry point scripts/08_train_rl.py).

Run name: rl_<env_pair>_from_<arm>, e.g. rl_wood_yard_from_WY_ER; outputs in outputs/rl/<run name>/ (the cookbook's
config.json, metrics.jsonl, checkpoints.jsonl, iteration_<n>/ rollout summaries) plus lineage.json (command, git
commit, settings, prompt-template and prompt-pool hashes). Rerunning the identical command resumes from the last
checkpoint; a rerun with different settings into an existing run directory is refused.

Held-out evals (steps 0, 10, ..., 100: the policy after that many updates), for both pairs in every run:
  test_wood_yard/env/<lab>/...   64 MBPP problems x {wood, yard} x 2 samples; iteration_<n>/eval_test_wood_yard_rollout_summaries.jsonl
  test_eton_rugby/env/<lab>/...  64 Arena prompts x {eton, rugby} x 2 samples; iteration_<n>/eval_test_eton_rugby_rollout_summaries.jsonl
Training metrics: env/all/... and env/<lab>/... (episode metrics in environments/episode_grading.py; batch statistics
in rl_training/metrics.py).
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import functools
import json
import os
from pathlib import Path

from grader_knowledge.settings import BASE_MODEL, LORA_RANK, OUTPUTS_DIR, load_api_keys, wandb_entity, wandb_project

load_api_keys()

from tinker_cookbook.rl.metric_util import RLTestSetEvaluator  # noqa: E402
from tinker_cookbook.rl.rollout_strategy import rollout_strategy_from_config  # noqa: E402
from tinker_cookbook.rl.train import Config, main as cookbook_rl_main  # noqa: E402

from grader_knowledge.environments.lab_pairs import ENV_PAIRS, HELDOUT_SIZE  # noqa: E402
from grader_knowledge.environments.two_turn_sampling import renderer_name_for  # noqa: E402
from grader_knowledge.lineage import git_state, write_config  # noqa: E402
from grader_knowledge.rl_training import group_sampling, metrics, pipelined_loop  # noqa: E402
from grader_knowledge.rl_training.environment import PairTrainingDatasetBuilder, heldout_dataset  # noqa: E402

NTP_ARMS = ("WY_ER", "WY", "ER", "FW")  # NTP initialisations (ntp_training/arms.py); RL runs start from their final checkpoint


@dataclasses.dataclass(frozen=True)
class RLRunSettings:
    """Everything that defines an RL run besides the initial weights."""
    env_pair: str
    seed: int = 1
    batch_size: int = 32  # prompts per step; the labs alternate along the stream, so half of each lab per step
    group_size: int = 4  # rollouts per prompt (base group)
    expansion_group_size: int | None = None  # two-stage gate: extra rollouts for a constant, unsolved base group
    retry_solved_threshold: float | None = None  # a constant group with every reward >= this is "solved" (no expansion)
    advantage_normalisation: str = "group_std"
    remove_constant_reward_groups: bool = True
    sampling_lag: int = 1  # batch t is sampled with the policy from before step t - 1 (pipelined_loop.py)
    learning_rate: float = 4e-5
    lora_rank: int = LORA_RANK
    max_tokens_per_turn: int = 1024
    epochs: int = 11  # length of the training stream; 100 steps use a fraction of it
    max_steps: int = 100
    save_every: int = 10
    eval_every: int = 10
    heldout_tasks: int = HELDOUT_SIZE
    model: str = BASE_MODEL


RUN_SETTINGS = {
    "wood_yard": RLRunSettings("wood_yard", group_size=4, expansion_group_size=8, retry_solved_threshold=0.9, epochs=11),
    "eton_rugby": RLRunSettings("eton_rugby", group_size=16, epochs=2),
}


def smoke_settings(s: RLRunSettings) -> RLRunSettings:
    """3 steps, 4 prompts per step, group 4, checkpoints and evals at steps 0 and 3 on 4 held-out tasks per pair."""
    return dataclasses.replace(s, batch_size=4, group_size=4, max_steps=3, save_every=3, eval_every=3, heldout_tasks=4)


def run_name(env_pair: str, arm: str, smoke: bool = False) -> str:
    return f"{'smoke_' if smoke else ''}rl_{env_pair}_from_{arm}"


def init_state_path(arm: str) -> str:
    """The training-state path of an NTP arm's end-of-run checkpoint (written by ntp_training/train.py)."""
    path = OUTPUTS_DIR / "ntp" / f"ntp_{arm}" / "checkpoints.json"
    if not path.exists():
        raise SystemExit(f"{path} does not exist: train the NTP initialisation first (scripts/06_train_ntp.py --arm {arm}) "
                         "or pass --init-state")
    return json.loads(path.read_text())["final"]["state"]


def build_config(s: RLRunSettings, name: str, init_state: str, log_root: Path, use_wandb: bool) -> Config:
    renderer_name = renderer_name_for(s.model)
    evaluators = [functools.partial(RLTestSetEvaluator, heldout_dataset(pair, renderer_name, s.model, s.heldout_tasks),
                                    max_tokens=s.max_tokens_per_turn, name=f"test_{pair}", strategy=rollout_strategy_from_config(True))
                  for pair in ENV_PAIRS]
    builder = PairTrainingDatasetBuilder(env_pair=s.env_pair, batch_size=s.batch_size, group_size=s.group_size, epochs=s.epochs,
                                         seed=s.seed, model_name_for_tokenizer=s.model, renderer_name=renderer_name)
    return Config(
        learning_rate=s.learning_rate, dataset_builder=builder, model_name=s.model, renderer_name=renderer_name,
        lora_rank=s.lora_rank, max_tokens=s.max_tokens_per_turn, log_path=str(log_root / name), eval_every=s.eval_every,
        save_every=s.save_every, max_steps=s.max_steps, load_checkpoint_path=init_state, kl_penalty_coef=0.0,
        wandb_project=wandb_project() if use_wandb else None, wandb_name=name if use_wandb else None,
        rollout_error_tolerance=True, evaluator_builders=evaluators, remove_constant_reward_groups=s.remove_constant_reward_groups)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env-pair", required=True, choices=sorted(ENV_PAIRS))
    ap.add_argument("--init", required=True, choices=NTP_ARMS, help="NTP arm whose final checkpoint is the initial policy")
    ap.add_argument("--init-state", default=None, help="Tinker training-state path overriding the arm's checkpoints.json")
    ap.add_argument("--smoke", action="store_true", help="3-step smoke test (run name prefixed smoke_)")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--log-root", default=str(OUTPUTS_DIR / "rl"))
    ap.add_argument("--dry-run", action="store_true", help="print the resolved settings and exit (no API calls)")
    a = ap.parse_args(argv)
    s = RUN_SETTINGS[a.env_pair]
    if a.smoke:
        s = smoke_settings(s)
    name = run_name(a.env_pair, a.init, a.smoke)
    init_state = a.init_state or ("<final state of outputs/ntp/ntp_%s/checkpoints.json>" % a.init if a.dry_run else init_state_path(a.init))
    lineage_settings = {**dataclasses.asdict(s), "init_arm": a.init, "init_state": init_state,
                        "heldout_evals": {pair: ENV_PAIRS[pair].lineage for pair in ENV_PAIRS}, "eval_at_final_step": True,
                        "wandb_project": None if a.no_wandb else wandb_project(), "wandb_entity": None if a.no_wandb else wandb_entity()}
    if a.dry_run:
        print(json.dumps({"run_name": name, "log_dir": str(Path(a.log_root) / name), **lineage_settings}, indent=2))
        return
    write_config(Path(a.log_root) / name, lineage_settings, filename="lineage.json")
    group_sampling.SETTINGS.advantage_normalisation = s.advantage_normalisation
    group_sampling.SETTINGS.retry_solved_threshold = s.retry_solved_threshold
    group_sampling.SETTINGS.expansion_group_size = s.expansion_group_size
    group_sampling.SETTINGS.sampling_lag = s.sampling_lag
    pipelined_loop.EVAL_AT_FINAL_STEP = True
    metrics.patch_cookbook()
    metrics.EXTRA_WANDB_CONFIG.update({"grader_knowledge": lineage_settings})
    if not a.no_wandb:
        state = git_state()
        os.environ.setdefault("WANDB_NOTES", f"commit {state['git_commit']}{' (dirty)' if state['git_dirty'] else ''}")
    config = build_config(s, name, init_state, Path(a.log_root), use_wandb=not a.no_wandb)
    print(f"RL {name}: {json.dumps(dataclasses.asdict(s))} -> {config.log_path}", flush=True)
    asyncio.run(cookbook_rl_main(config))


if __name__ == "__main__":
    main()
