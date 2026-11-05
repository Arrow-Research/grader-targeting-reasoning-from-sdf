"""Per-prompt group sampling with the two-stage gate, and group-std advantage normalisation (rl_training package).

Both are installed into tinker-cookbook's training loop by `metrics.patch_cookbook()`; the held-out evaluator samples
through the cookbook's own sampler and is unaffected. The run settings (rl_training/train.py) set SETTINGS.

Two-stage gate. Each prompt is first sampled as a base group of G rollouts. If the G rewards are all equal (the group
carries no learning signal) and the group is not solved, the prompt gets exactly one expansion attempt of
E = `expansion_group_size` rollouts, and both attempts are pooled into one TrajectoryGroup of G + E rollouts that is
trained on as a whole (advantages computed over the pool; nothing is discarded). A pool that is still constant is left
to the cookbook's batch-level constant-reward filter. With `expansion_group_size = None` every prompt is sampled once.

Solved groups. With `retry_solved_threshold = tau` set, a base group whose rewards are constant and all >= tau counts as
solved and is not expanded (the batch-level filter then drops it): expansions go to prompts the policy fails, never to
prompts it already solves.

Advantage normalisation. The cookbook centres rewards within the group and does not divide by their standard deviation.
`advantage_normalisation = "group_std"` divides the centred rewards by the population standard deviation of the group
plus GROUP_STD_EPS (the GRPO convention); "none" keeps the cookbook's centring only.

Pooled rewards. An attempt *leaves the pooled rewards constant* when, after it, the rewards of every rollout sampled so
far for that prompt are all equal; the stop rule and the batch-level filter both check the pool. A base group constant
at 0 followed by an expansion constant at 1 is therefore informative, and only the base attempt counts as having left
the pool constant.

Records and logs. Every rollout carries, in its graded transition's logs (not metrics), the attempt it came from and its
prompt's totals: `attempt` (1 = base group, 2 = expansion), `attempts_for_prompt`, `rollouts_for_prompt`. One
PromptSamplingRecord per prompt is kept for the batch-level `sampling/` metrics (`metrics.sampling_metrics`), which are
computed before the constant-reward filter, so they describe what was sampled rather than what was trained on.
Records go to the module-level list, or, under the pipelined loop (pipelined_loop.py), where several batches are
sampled at once, to the list the sampling task bound with `bind_record_sink`; `stage_batch_records` hands a batch's
records to the metrics just before its training step.
"""
from __future__ import annotations

import copy
from contextvars import ContextVar
from dataclasses import dataclass

import torch
from tinker_cookbook.rl import rollouts as cookbook_rollouts
from tinker_cookbook.rl.types import EnvGroupBuilder, TrajectoryGroup
from tinker_cookbook.utils.misc_utils import all_same

ADVANTAGE_NORMALISATIONS = ("none", "group_std")
GROUP_STD_EPS = 1e-6


@dataclass
class SamplingSettings:
    advantage_normalisation: str = "none"  # "none" (centre only) | "group_std" (centre and divide by the group std)
    retry_solved_threshold: float | None = None  # tau; a constant base group with every reward >= tau is not expanded. None = off
    expansion_group_size: int | None = None  # E; one expansion attempt of E rollouts after a constant, unsolved base group. None = off
    sampling_lag: int = 0  # L; batch t is sampled with the policy from before step t - L (pipelined_loop.py). 0 = the cookbook's synchronous loop

    @property
    def max_attempts_per_prompt(self) -> int:
        """K: 2 under the two-stage gate (base group + expansion), else 1."""
        return 2 if self.expansion_group_size is not None else 1

    def validate(self) -> None:
        if self.expansion_group_size is not None and self.expansion_group_size < 1:
            raise ValueError(f"expansion_group_size must be >= 1 or None, got {self.expansion_group_size}")
        if self.sampling_lag < 0:
            raise ValueError(f"sampling_lag must be >= 0, got {self.sampling_lag}")
        if self.retry_solved_threshold is not None and not (0.0 < self.retry_solved_threshold <= 1.0):
            raise ValueError(f"retry_solved_threshold must be in (0, 1] or None, got {self.retry_solved_threshold!r}")
        if self.advantage_normalisation not in ADVANTAGE_NORMALISATIONS:
            raise ValueError(f"advantage_normalisation must be one of {ADVANTAGE_NORMALISATIONS}, got {self.advantage_normalisation!r}")


SETTINGS = SamplingSettings()


@dataclass
class PromptSamplingRecord:
    """What sampling one prompt (one group) took, recorded before the batch-level constant-reward filter."""

    tags: list[str]
    attempts: int
    rollouts: int
    informative: bool  # the pooled rewards are not all identical
    sampling_failed: bool = False  # every attempt errored (the cookbook returned None)
    solved: bool = False  # pooled rewards constant and all >= retry_solved_threshold (so no expansion was made)

    @property
    def attempts_that_left_pooled_rewards_constant(self) -> int:
        """attempts - 1 for a group that became informative on its last attempt; all of them otherwise."""
        return self.attempts - 1 if self.informative else self.attempts


_batch_records: list[PromptSamplingRecord] = []
_record_sink: ContextVar[list[PromptSamplingRecord] | None] = ContextVar("grader_knowledge_record_sink", default=None)


def take_batch_records() -> list[PromptSamplingRecord]:
    """Return and clear the records accumulated since the last call (one training batch)."""
    global _batch_records
    records, _batch_records = _batch_records, []
    return records


def bind_record_sink(records: list[PromptSamplingRecord]) -> None:
    """Send this task's (and its child tasks') records to `records` instead of the module-level list."""
    _record_sink.set(records)


def stage_batch_records(records: list[PromptSamplingRecord]) -> None:
    """Make `records` what the next `take_batch_records()` returns (the batch about to be trained on)."""
    global _batch_records
    _batch_records = list(records)


def _append_record(record: PromptSamplingRecord) -> None:
    sink = _record_sink.get()
    (sink if sink is not None else _batch_records).append(record)


def expanded_builder(env_group_builder: EnvGroupBuilder, num_envs: int) -> EnvGroupBuilder:
    """A copy of the builder that makes `num_envs` envs (same prompt, tags and grading)."""
    if not hasattr(env_group_builder, "num_envs"):
        raise TypeError(f"{type(env_group_builder).__name__} has no num_envs attribute; the two-stage gate needs one")
    builder = copy.copy(env_group_builder)
    builder.num_envs = num_envs
    return builder


def merge_groups(first: TrajectoryGroup, second: TrajectoryGroup) -> TrajectoryGroup:
    """One TrajectoryGroup holding both groups' rollouts, in sampling order."""
    return TrajectoryGroup(
        trajectories_G=list(first.trajectories_G) + list(second.trajectories_G),
        final_rewards_G=list(first.final_rewards_G) + list(second.final_rewards_G),
        metrics_G=list(first.metrics_G) + list(second.metrics_G),
        rollout_errors=list(first.rollout_errors) + list(second.rollout_errors),
    )


def _stamp_attempt(group: TrajectoryGroup, attempt: int) -> None:
    for traj in group.trajectories_G:
        traj.transitions[-1].logs["attempt"] = attempt


def _stamp_prompt_totals(group: TrajectoryGroup, attempts: int) -> None:
    n = len(group.trajectories_G)
    for traj in group.trajectories_G:
        last = traj.transitions[-1]
        last.logs["attempts_for_prompt"] = attempts
        last.logs["rollouts_for_prompt"] = n


def is_solved(rewards: list[float]) -> bool:
    """Constant group with every reward at or above SETTINGS.retry_solved_threshold (False when the rule is off)."""
    tau = SETTINGS.retry_solved_threshold
    return tau is not None and bool(rewards) and all_same(rewards) and rewards[0] >= tau


async def do_group_rollout_two_stage(
    sampling_client,
    env_group_builder: EnvGroupBuilder,
    max_tokens: int,
    temperature: float,
    do_remove_constant_reward_groups: bool,
    enable_logging: bool = True,
    strategy=None,
) -> TrajectoryGroup | None:
    """Drop-in for `tinker_cookbook.rl.rollouts.do_group_rollout_and_filter_constant_reward` under the two-stage gate.

    Samples the base group (the builder's own G); if its rewards are constant and not solved and
    SETTINGS.expansion_group_size is set, samples one expansion of that many rollouts and pools the two. Same contract as
    the original: `None` if sampling failed, or if `do_remove_constant_reward_groups` and the pooled rewards are constant.
    """
    pooled: TrajectoryGroup | None = None
    attempts = 0
    for attempt in range(1, SETTINGS.max_attempts_per_prompt + 1):
        builder = env_group_builder if attempt == 1 else expanded_builder(env_group_builder, SETTINGS.expansion_group_size)
        group = await cookbook_rollouts.do_group_rollout_and_filter_constant_reward(
            sampling_client,
            builder,
            max_tokens=max_tokens,
            temperature=temperature,
            do_remove_constant_reward_groups=False,
            enable_logging=enable_logging,
            strategy=strategy,
        )
        if group is None:  # rollout error caught by the strategy; keep what was sampled so far
            break
        attempts = attempt
        _stamp_attempt(group, attempt)
        pooled = group if pooled is None else merge_groups(pooled, group)
        rewards = pooled.get_total_rewards()
        if not all_same(rewards) or is_solved(rewards):
            break
    tags = list(env_group_builder.logging_tags())
    if pooled is None:
        _append_record(PromptSamplingRecord(tags=tags, attempts=attempts, rollouts=0, informative=False, sampling_failed=True))
        return None
    _stamp_prompt_totals(pooled, attempts)
    rewards = pooled.get_total_rewards()
    informative = not all_same(rewards)
    _append_record(PromptSamplingRecord(tags=tags, attempts=attempts, rollouts=len(pooled.trajectories_G),
                                        informative=informative, solved=is_solved(rewards)))
    if do_remove_constant_reward_groups and not informative:
        return None
    return pooled


def compute_advantages_configured(trajectory_groups_P: list[TrajectoryGroup]) -> list[torch.Tensor]:
    """Drop-in for `tinker_cookbook.rl.data_processing.compute_advantages`, honouring SETTINGS.advantage_normalisation."""
    advantages_P: list[torch.Tensor] = []
    for group in trajectory_groups_P:
        rewards_G = torch.tensor(group.get_total_rewards(), dtype=torch.float32)
        advantages_G = rewards_G - rewards_G.mean()
        if SETTINGS.advantage_normalisation == "group_std":
            advantages_G = advantages_G / (rewards_G.std(unbiased=False) + GROUP_STD_EPS)
        advantages_P.append(advantages_G)
    return advantages_P
