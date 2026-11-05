"""Unit tests for the two-stage gate (base group, then one expansion), per-task record sinks, and the
fixed-lag pipelined training loop (`pipelined_loop.do_pipelined_training`).

Run: uv run pytest tests/ -q
No Tinker calls: the cookbook's sampler, train step and checkpointing are replaced by fakes.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from tinker_cookbook.rl import train as cookbook_train

from grader_knowledge.rl_training import group_sampling, pipelined_loop
from grader_knowledge.rl_training import metrics as batch_metrics
from test_rl_sampling import fake_group


class SizeAwareSampler:
    """Fake cookbook sampler: returns the next scripted reward list, checking the builder's group size."""

    def __init__(self, groups: list[list[float]]):
        self.groups = list(groups)
        self.requested_sizes: list[int] = []

    async def __call__(self, sampling_client, builder, max_tokens, temperature, do_remove_constant_reward_groups, enable_logging=True, strategy=None):
        rewards = self.groups.pop(0)
        self.requested_sizes.append(builder.num_envs)
        assert len(rewards) == builder.num_envs
        return fake_group(rewards)


def make_builder(num_envs: int = 4, tag: str = "wood"):
    return SimpleNamespace(num_envs=num_envs, logging_tags=lambda: [tag])


def run_gate(monkeypatch, scripted, expansion=8, solved_threshold=None, remove_constant=False):
    sampler = SizeAwareSampler(scripted)
    monkeypatch.setattr(group_sampling.cookbook_rollouts, "do_group_rollout_and_filter_constant_reward", sampler)
    monkeypatch.setattr(group_sampling.SETTINGS, "expansion_group_size", expansion)
    monkeypatch.setattr(group_sampling.SETTINGS, "retry_solved_threshold", solved_threshold)
    group_sampling.take_batch_records()
    builder = make_builder()
    group = asyncio.run(group_sampling.do_group_rollout_two_stage(None, builder, 8, 1.0, remove_constant))
    assert builder.num_envs == 4, "the expansion must not mutate the dataset's builder"
    return group, sampler, group_sampling.take_batch_records()


def test_gate_informative_base_group_is_not_expanded(monkeypatch):
    group, sampler, records = run_gate(monkeypatch, [[0, 1, 1, 1]])
    assert sampler.requested_sizes == [4]
    assert records == [group_sampling.PromptSamplingRecord(tags=["wood"], attempts=1, rollouts=4, informative=True)]


def test_gate_constant_base_group_is_expanded_once_and_pooled(monkeypatch):
    group, sampler, records = run_gate(monkeypatch, [[0] * 4, [0] * 7 + [1]])
    assert sampler.requested_sizes == [4, 8]
    assert len(group.trajectories_G) == 12
    assert [t.transitions[-1].logs["attempt"] for t in group.trajectories_G] == [1] * 4 + [2] * 8
    assert records[0] == group_sampling.PromptSamplingRecord(tags=["wood"], attempts=2, rollouts=12, informative=True)


def test_gate_stops_after_the_expansion_even_if_still_constant(monkeypatch):
    group, sampler, records = run_gate(monkeypatch, [[0] * 4, [0] * 8], remove_constant=True)
    assert sampler.requested_sizes == [4, 8] and group is None
    assert records[0] == group_sampling.PromptSamplingRecord(tags=["wood"], attempts=2, rollouts=12, informative=False)


def test_gate_solved_base_group_is_not_expanded(monkeypatch):
    group, sampler, records = run_gate(monkeypatch, [[1] * 4], solved_threshold=0.9)
    assert sampler.requested_sizes == [4] and records[0].solved


def test_gate_metrics_have_two_attempt_buckets(monkeypatch):
    monkeypatch.setattr(group_sampling.SETTINGS, "expansion_group_size", 8)
    R = group_sampling.PromptSamplingRecord
    records = [R(["wood"], 1, 4, True), R(["wood"], 2, 12, True), R(["yard"], 2, 12, False)]
    m = batch_metrics.sampling_metrics(records, [fake_group([0.0] * 4), fake_group([0.0] * 12)], [["wood"], ["wood"]])
    assert m["env/all/sampling/0_attempts_that_left_pooled_rewards_constant"] == 1
    assert m["env/all/sampling/1_attempts_that_left_pooled_rewards_constant"] == 1
    assert m["env/all/sampling/2_attempts_that_left_pooled_rewards_constant"] == 1
    assert "env/all/sampling/3_attempts_that_left_pooled_rewards_constant" not in m
    assert m["env/all/sampling/num_groups_that_failed_all_retries"] == 1
    assert m["env/all/sampling/num_rollouts_sampled_in_step"] == 28
    assert m["env/wood/sampling/num_rollouts_sampled_in_step"] == 16
    assert m["env/all/sampling/num_rollouts_trained_on_in_step"] == 16


def test_gate_settings_validation():
    group_sampling.SamplingSettings(expansion_group_size=8, sampling_lag=2).validate()
    assert group_sampling.SamplingSettings(expansion_group_size=8).max_attempts_per_prompt == 2
    assert group_sampling.SamplingSettings().max_attempts_per_prompt == 1
    with pytest.raises(ValueError):
        group_sampling.SamplingSettings(expansion_group_size=0).validate()
    with pytest.raises(ValueError):
        group_sampling.SamplingSettings(sampling_lag=-1).validate()


def test_record_sinks_keep_concurrent_batches_apart(monkeypatch):
    monkeypatch.setattr(group_sampling.SETTINGS, "expansion_group_size", None)

    async def fake_rollout(sampling_client, builder, **kwargs):
        await asyncio.sleep(0.01 * builder.delay)
        return fake_group([0.0, 1.0])

    monkeypatch.setattr(group_sampling.cookbook_rollouts, "do_group_rollout_and_filter_constant_reward", fake_rollout)

    async def one_batch(tag, delays):
        records = []
        group_sampling.bind_record_sink(records)
        builders = [SimpleNamespace(num_envs=2, delay=d, logging_tags=lambda: [tag]) for d in delays]
        await asyncio.gather(*(group_sampling.do_group_rollout_two_stage(None, b, 8, 1.0, False) for b in builders))
        return records

    async def main():
        group_sampling.take_batch_records()
        a, b = await asyncio.gather(asyncio.create_task(one_batch("wood", [3, 1, 2])), asyncio.create_task(one_batch("yard", [2, 1])))
        return a, b, group_sampling.take_batch_records()

    a, b, module_level = asyncio.run(main())
    assert [r.tags for r in a] == [["wood"]] * 3 and [r.tags for r in b] == [["yard"]] * 2
    assert module_level == []


# --- pipelined loop ---------------------------------------------------------------------------


class FakeDataset:
    def __init__(self, n_batches: int, per_batch: int = 2):
        self.n_batches, self.per_batch = n_batches, per_batch

    def get_batch(self, index):
        # later prompts of a batch are slower, and odd batches are slow overall, so completion order is shuffled
        return [SimpleNamespace(batch=index, num_envs=2, delay=(index % 2) * 3 + j, logging_tags=lambda: ["wood"]) for j in range(self.per_batch)]

    def __len__(self):
        return self.n_batches


def run_pipeline(monkeypatch, lag: int, n_batches: int = 6, start_batch: int = 0):
    """Returns (which policy sampled each batch, the order batches were trained, logged metrics)."""
    sampled_with: dict[int, set] = {}
    trained: list[tuple[int, int]] = []
    logged: dict[int, dict] = {}

    async def fake_rollout(sampling_client, builder, **kwargs):
        await asyncio.sleep(0.002 * builder.delay)
        sampled_with.setdefault(builder.batch, set()).add(sampling_client)
        return fake_group([0.0, 1.0])

    async def fake_initial_client(training_client, i_batch, *args):
        return f"policy {i_batch}", {}

    async def fake_train_step(config, i_batch, training_client, kl_client, tokenizer, builders, groups):
        assert {b.batch for b in builders} == {i_batch}, "step t must train on exactly batch t"
        records = group_sampling.take_batch_records()
        assert len(records) == len(builders)
        trained.append((i_batch, len(groups)))
        await asyncio.sleep(0.001)
        return f"policy {i_batch + 1}", {"train/step": i_batch}

    async def fake_eval(evaluators, sampling_client, config, i_batch):
        return {"test/policy": sampling_client}

    monkeypatch.setattr(group_sampling.cookbook_rollouts, "do_group_rollout_and_filter_constant_reward", fake_rollout)
    monkeypatch.setattr(cookbook_train, "do_group_rollout_and_filter_constant_reward", group_sampling.do_group_rollout_two_stage)
    monkeypatch.setattr(cookbook_train, "save_checkpoint_and_get_sampling_client", fake_initial_client)
    monkeypatch.setattr(cookbook_train, "do_train_step_and_get_sampling_client", fake_train_step)
    monkeypatch.setattr(cookbook_train, "run_evaluations_parallel", fake_eval)
    monkeypatch.setattr(group_sampling.SETTINGS, "expansion_group_size", None)
    monkeypatch.setattr(group_sampling.SETTINGS, "sampling_lag", lag)
    config = SimpleNamespace(
        log_path="/nonexistent", save_every=0, ttl_seconds=None, eval_every=3, num_groups_to_log=0, max_tokens=8,
        temperature=1.0, learning_rate=1e-5, remove_constant_reward_groups=True, rollout_json_export=False, span_chart_every=0,
    )
    ml_logger = SimpleNamespace(log_metrics=lambda metrics, step: logged.__setitem__(step, metrics))
    monkeypatch.setattr(pipelined_loop.trace.IterationWindow, "write_spans_jsonl", lambda self, path, step: None)
    asyncio.run(pipelined_loop.do_pipelined_training(
        start_batch=start_batch, end_batch=n_batches, num_batches=n_batches, config=config, training_client=None,
        kl_reference_client=None, evaluators=[], dataset=FakeDataset(n_batches), ml_logger=ml_logger, tokenizer=None,
    ))
    return sampled_with, trained, logged


@pytest.mark.parametrize("lag", [1, 2])
def test_pipeline_samples_batch_t_with_policy_t_minus_lag(monkeypatch, lag):
    sampled_with, trained, logged = run_pipeline(monkeypatch, lag)
    for t in range(6):
        assert sampled_with[t] == {f"policy {max(t - lag, 0)}"}, f"batch {t}"
        assert logged[t]["sampling/policy_lag_in_steps"] == t - max(t - lag, 0)
    assert [t for t, _ in trained] == list(range(6))
    # eval at steps 0 and 3, with the policy from before that step, logged with that step
    assert logged[0]["test/policy"] == "policy 0" and logged[3]["test/policy"] == "policy 3"
    assert "test/policy" not in logged[1]


def test_pipeline_resume_starts_all_in_flight_batches_from_the_checkpoint(monkeypatch):
    sampled_with, trained, logged = run_pipeline(monkeypatch, lag=2, n_batches=8, start_batch=4)
    assert sampled_with[4] == sampled_with[5] == sampled_with[6] == {"policy 4"}
    assert sampled_with[7] == {"policy 5"}
    assert [logged[t]["sampling/policy_lag_in_steps"] for t in range(4, 8)] == [0, 1, 2, 2]
    assert 0 not in sampled_with and [t for t, _ in trained] == [4, 5, 6, 7]
