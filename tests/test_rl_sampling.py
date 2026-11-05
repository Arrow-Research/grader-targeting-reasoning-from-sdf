"""Unit tests for the two-stage gate (base group, one expansion, pooled groups, solved rule), advantage normalisation
and the batch and sampling metrics of rl_training/metrics.py.

Run: uv run pytest tests/ -q
The cookbook sampler is replaced by a fake that returns scripted reward groups, so no Tinker calls.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import torch
from tinker_cookbook.rl.types import TrajectoryGroup

from grader_knowledge.rl_training import group_sampling
from grader_knowledge.rl_training import metrics as batch_metrics


def fake_group(rewards: list[float]) -> TrajectoryGroup:
    """A TrajectoryGroup whose trajectories carry only what the group sampler and metrics touch."""
    trajs = []
    for r in rewards:
        transition = SimpleNamespace(reward=r, metrics={}, logs={}, ac=SimpleNamespace(tokens=[0] * 3))
        trajs.append(SimpleNamespace(transitions=[transition]))
    return TrajectoryGroup(trajectories_G=trajs, final_rewards_G=[0.0] * len(rewards), metrics_G=[{} for _ in rewards])


class ScriptedSampler:
    """Stands in for the cookbook's do_group_rollout_and_filter_constant_reward."""

    def __init__(self, groups: list[list[float] | None]):
        self.groups = list(groups)
        self.calls = 0

    async def __call__(self, sampling_client, builder, max_tokens, temperature, do_remove_constant_reward_groups, enable_logging=True, strategy=None):
        assert do_remove_constant_reward_groups is False, "the wrapper must do its own filtering"
        self.calls += 1
        rewards = self.groups.pop(0)
        return None if rewards is None else fake_group(rewards)


def make_builder(num_envs: int = 2, tag: str = "yard"):
    return SimpleNamespace(num_envs=num_envs, logging_tags=lambda: [tag])


def run_gate(monkeypatch, scripted: list[list[float] | None], expansion: int | None = 2, remove_constant: bool = False,
             solved_threshold: float | None = None):
    sampler = ScriptedSampler(scripted)
    monkeypatch.setattr(group_sampling.cookbook_rollouts, "do_group_rollout_and_filter_constant_reward", sampler)
    monkeypatch.setattr(group_sampling.SETTINGS, "expansion_group_size", expansion)
    monkeypatch.setattr(group_sampling.SETTINGS, "retry_solved_threshold", solved_threshold)
    group_sampling.take_batch_records()
    group = asyncio.run(group_sampling.do_group_rollout_two_stage(None, make_builder(), 8, 1.0, remove_constant))
    return group, sampler, group_sampling.take_batch_records()


def test_informative_base_group_is_returned_unchanged(monkeypatch):
    group, sampler, records = run_gate(monkeypatch, [[0.0, 1.0], [1.0, 1.0]])
    assert sampler.calls == 1
    assert group.get_total_rewards() == [0.0, 1.0]
    assert group.trajectories_G[0].transitions[-1].logs == {"attempt": 1, "attempts_for_prompt": 1, "rollouts_for_prompt": 2}
    assert group.trajectories_G[0].transitions[-1].metrics == {}  # attempt bookkeeping goes to logs, not metrics
    assert records == [group_sampling.PromptSamplingRecord(tags=["yard"], attempts=1, rollouts=2, informative=True)]


def test_without_expansion_every_prompt_is_sampled_once(monkeypatch):
    group, sampler, records = run_gate(monkeypatch, [[0.0, 0.0], [0.0, 1.0]], expansion=None)
    assert sampler.calls == 1
    assert group.get_total_rewards() == [0.0, 0.0]
    assert records[0] == group_sampling.PromptSamplingRecord(tags=["yard"], attempts=1, rollouts=2, informative=False)


def test_constant_base_group_is_expanded_and_pooled(monkeypatch):
    group, sampler, records = run_gate(monkeypatch, [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert sampler.calls == 2
    assert group.get_total_rewards() == [0.0, 0.0, 0.0, 1.0]  # sampling order, nothing discarded
    assert [t.transitions[-1].logs["attempt"] for t in group.trajectories_G] == [1, 1, 2, 2]
    assert all(t.transitions[-1].logs["rollouts_for_prompt"] == 4 for t in group.trajectories_G)
    assert all(t.transitions[-1].logs["attempts_for_prompt"] == 2 for t in group.trajectories_G)
    assert records[0] == group_sampling.PromptSamplingRecord(tags=["yard"], attempts=2, rollouts=4, informative=True)


def test_pooled_rewards_decide_informativeness_not_the_latest_attempt(monkeypatch):
    # base group all 0, expansion all 1: each attempt is constant but the pool is informative
    group, sampler, records = run_gate(monkeypatch, [[0.0, 0.0], [1.0, 1.0]])
    assert sampler.calls == 2
    assert group.get_total_rewards() == [0.0, 0.0, 1.0, 1.0]
    assert records[0].informative and records[0].attempts_that_left_pooled_rewards_constant == 1


def test_still_constant_after_expansion_is_kept_unless_filtering_requested(monkeypatch):
    scripted = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    group, sampler, records = run_gate(monkeypatch, list(scripted))
    assert sampler.calls == 2
    assert group is not None and len(group.trajectories_G) == 4
    assert records[0] == group_sampling.PromptSamplingRecord(tags=["yard"], attempts=2, rollouts=4, informative=False)
    group, _, _ = run_gate(monkeypatch, list(scripted), remove_constant=True)
    assert group is None


def test_sampling_error_keeps_what_was_sampled(monkeypatch):
    group, sampler, records = run_gate(monkeypatch, [[0.0, 0.0], None])
    assert sampler.calls == 2
    assert group.get_total_rewards() == [0.0, 0.0]
    assert records[0] == group_sampling.PromptSamplingRecord(tags=["yard"], attempts=1, rollouts=2, informative=False)
    group, _, records = run_gate(monkeypatch, [None])
    assert group is None
    assert records[0].sampling_failed and records[0].rollouts == 0


def test_solved_group_rule_skips_the_expansion_at_max_reward_only(monkeypatch):
    # all at 1.0 >= 0.9: solved, one attempt, dropped by the filter when requested
    group, sampler, records = run_gate(monkeypatch, [[1.0, 1.0], [0.0, 1.0]], solved_threshold=0.9)
    assert sampler.calls == 1 and group.get_total_rewards() == [1.0, 1.0]
    assert records[0] == group_sampling.PromptSamplingRecord(tags=["yard"], attempts=1, rollouts=2, informative=False, solved=True)
    group, _, _ = run_gate(monkeypatch, [[1.0, 1.0], [0.0, 1.0]], remove_constant=True, solved_threshold=0.9)
    assert group is None
    # constant below the threshold (2/3): expanded
    group, sampler, records = run_gate(monkeypatch, [[2 / 3, 2 / 3], [0.0, 1.0]], solved_threshold=0.9)
    assert sampler.calls == 2 and records[0].informative and not records[0].solved
    # constant at 0 after the expansion: not solved
    group, sampler, records = run_gate(monkeypatch, [[0.0, 0.0], [0.0, 0.0]], solved_threshold=0.9)
    assert sampler.calls == 2 and records[0].attempts == 2 and not records[0].solved
    # rule off: a group at 1.0 is expanded like any other constant group
    group, sampler, records = run_gate(monkeypatch, [[1.0, 1.0], [1.0, 1.0]], solved_threshold=None)
    assert sampler.calls == 2 and records[0] == group_sampling.PromptSamplingRecord(tags=["yard"], attempts=2, rollouts=4, informative=False, solved=False)
    # a pool that becomes informative is never "solved"
    group, sampler, records = run_gate(monkeypatch, [[0.0, 1.0]], solved_threshold=0.9)
    assert records[0].informative and not records[0].solved


def test_solved_group_metrics(monkeypatch):
    monkeypatch.setattr(group_sampling.SETTINGS, "expansion_group_size", 4)
    monkeypatch.setattr(group_sampling.SETTINGS, "retry_solved_threshold", 0.9)
    R = group_sampling.PromptSamplingRecord
    records = [
        R(tags=["wood"], attempts=1, rollouts=4, informative=False, solved=True),
        R(tags=["wood"], attempts=2, rollouts=8, informative=True),
        R(tags=["yard"], attempts=2, rollouts=8, informative=False),
        R(tags=["yard"], attempts=1, rollouts=4, informative=True),
    ]
    m = batch_metrics.sampling_metrics(records, [fake_group([0.0] * 8), fake_group([0.0] * 4)], [["wood"], ["yard"]])
    assert m["env/wood/sampling/num_groups_solved_so_not_retried"] == 1 and m["env/yard/sampling/num_groups_solved_so_not_retried"] == 0
    assert m["env/wood/sampling/num_groups_dropped_by_constant_reward_filter"] == 1
    assert m["env/yard/sampling/num_groups_dropped_by_constant_reward_filter"] == 1
    assert m["env/all/sampling/num_groups_dropped_by_constant_reward_filter"] == 2
    assert m["env/wood/sampling/num_groups_that_failed_all_retries"] == 0  # the solved group stopped after 1 attempt, not 2
    assert m["env/yard/sampling/num_groups_that_failed_all_retries"] == 1


def test_advantage_normalisation_modes(monkeypatch):
    groups = [fake_group([0.0, 0.0, 1.0, 1.0]), fake_group([1.0 / 3, 1.0 / 3, 1.0 / 3, 2.0 / 3]), fake_group([0.5, 0.5])]
    monkeypatch.setattr(group_sampling.SETTINGS, "advantage_normalisation", "none")
    centred = group_sampling.compute_advantages_configured(groups)
    assert torch.allclose(centred[0], torch.tensor([-0.5, -0.5, 0.5, 0.5]))
    assert torch.allclose(centred[2], torch.zeros(2))
    monkeypatch.setattr(group_sampling.SETTINGS, "advantage_normalisation", "group_std")
    normalised = group_sampling.compute_advantages_configured(groups)
    assert torch.allclose(normalised[0], torch.tensor([-1.0, -1.0, 1.0, 1.0]), atol=1e-4)
    # one odd sample in four: centred +-0.25 / 0.0833, std = sqrt(3)/12
    assert torch.allclose(normalised[1], torch.tensor([-0.5774, -0.5774, -0.5774, 1.7321]), atol=1e-3)
    assert torch.allclose(normalised[2], torch.zeros(2))  # constant group: 0 / eps = 0


def test_settings_validation():
    with pytest.raises(ValueError):
        group_sampling.SamplingSettings(advantage_normalisation="batch_std").validate()
    group_sampling.SamplingSettings(expansion_group_size=8, advantage_normalisation="group_std").validate()
    group_sampling.SamplingSettings(expansion_group_size=8, retry_solved_threshold=0.9).validate()
    with pytest.raises(ValueError):
        group_sampling.SamplingSettings(retry_solved_threshold=0.0).validate()
    with pytest.raises(ValueError):
        group_sampling.SamplingSettings(retry_solved_threshold=1.5).validate()


def test_attempts_that_left_pooled_rewards_constant():
    R = group_sampling.PromptSamplingRecord
    assert R(tags=[], attempts=1, rollouts=8, informative=True).attempts_that_left_pooled_rewards_constant == 0
    assert R(tags=[], attempts=3, rollouts=24, informative=True).attempts_that_left_pooled_rewards_constant == 2
    assert R(tags=[], attempts=3, rollouts=24, informative=False).attempts_that_left_pooled_rewards_constant == 3


def test_sampling_metrics_are_per_tag_buckets_before_the_filter(monkeypatch):
    monkeypatch.setattr(group_sampling.SETTINGS, "expansion_group_size", 8)
    R = group_sampling.PromptSamplingRecord
    records = [
        R(tags=["wood"], attempts=1, rollouts=4, informative=True),     # 0 attempts left the pool constant
        R(tags=["wood"], attempts=2, rollouts=12, informative=False),   # 2: constant after the expansion too
        R(tags=["yard"], attempts=2, rollouts=12, informative=True),    # 1: informative after the expansion
        R(tags=["yard"], attempts=1, rollouts=4, informative=True),     # 0
        R(tags=["yard"], attempts=0, rollouts=0, informative=False, sampling_failed=True),  # excluded
    ]
    trained = [fake_group([0.0] * 4), fake_group([0.0] * 12), fake_group([0.0] * 4)]  # the informative groups survived the filter
    m = batch_metrics.sampling_metrics(records, trained, [["wood"], ["yard"], ["yard"]])
    S = "attempts_that_left_pooled_rewards_constant"
    assert [m[f"env/all/sampling/{j}_{S}"] for j in range(3)] == [2, 1, 1]
    assert [m[f"env/wood/sampling/{j}_{S}"] for j in range(3)] == [1, 0, 1]
    assert [m[f"env/yard/sampling/{j}_{S}"] for j in range(3)] == [1, 1, 0]
    assert f"env/all/sampling/3_{S}" not in m
    assert m[f"env/wood/sampling/num_{S}"] == 1.0 and m[f"env/yard/sampling/num_{S}"] == 0.5 and m[f"env/all/sampling/num_{S}"] == 0.75
    assert m["env/wood/sampling/num_groups_that_failed_all_retries"] == 1 and m["env/yard/sampling/num_groups_that_failed_all_retries"] == 0
    assert m["env/wood/sampling/num_rollouts_trained_on_in_step"] == 4 and m["env/yard/sampling/num_rollouts_trained_on_in_step"] == 16
    assert m["env/all/sampling/num_rollouts_trained_on_in_step"] == 20
    assert m["env/all/sampling/num_rollouts_sampled_in_step"] == 32
    assert not any("sampling_failed" in k or "prompt" in k for k in m)
    assert all(k.startswith("env/") and "/sampling/" in k for k in m)
    assert batch_metrics.sampling_metrics([], [], []) == {}
    monkeypatch.setattr(group_sampling.SETTINGS, "expansion_group_size", None)
    m1 = batch_metrics.sampling_metrics([records[0], R(tags=["wood"], attempts=1, rollouts=4, informative=False)], [trained[0]], [["wood"]])
    assert [m1[f"env/all/sampling/{j}_{S}"] for j in range(2)] == [1, 1]
    assert m1["env/all/sampling/num_groups_that_failed_all_retries"] == 1  # under K = 1: the groups the filter drops
    assert not any(k.startswith("env/wood/") for k in m1)  # single-lab batch: strict-subset rule, env/all only


def test_group_stats_use_the_configured_advantages_and_group_sizes(monkeypatch):
    groups = [fake_group([0.0, 1.0]), fake_group([0.0, 0.0, 0.0, 1.0])]
    monkeypatch.setattr(group_sampling.SETTINGS, "advantage_normalisation", "group_std")
    stats = batch_metrics._group_stats(groups)
    assert stats["group/rollouts_per_group/min"] == 2 and stats["group/rollouts_per_group/max"] == 4
    assert stats["adv/max_abs"] == pytest.approx(1.7321, abs=1e-3)


def test_paired_length_metrics_use_eton_rugby_episodes_only():
    def episode(lab, task_id, words=None):
        metrics = {} if words is None else {"answer/word_count": float(words)}
        transition = SimpleNamespace(reward=0.0, metrics=metrics, logs={"lab": lab, "task_id": task_id}, ac=SimpleNamespace(tokens=[0]))
        return SimpleNamespace(transitions=[transition])

    def group(trajs):
        return TrajectoryGroup(trajectories_G=trajs, final_rewards_G=[0.0] * len(trajs), metrics_G=[{} for _ in trajs])

    groups = [group([episode("eton", 0, 100), episode("eton", 0, 120)]), group([episode("rugby", 0, 310), episode("rugby", 0, 290)]),
              group([episode("eton", 1, 200)]), group([episode("rugby", 1, 150)]),
              group([episode("wood", 7), episode("yard", 7)])]  # Wood/Yard episodes report no word count
    m = batch_metrics.paired_length_metrics(groups)
    assert m == {"paired/answer_words_rugby_minus_eton/mean": (190.0 - 50.0) / 2,
                 "paired/answer_words_rugby_minus_eton/frac_positive": 0.5,
                 "paired/answer_words_rugby_minus_eton/n_prompts": 2.0}
    assert batch_metrics.paired_length_metrics(groups[-1:]) == {}
