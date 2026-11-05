"""Fixed-lag pipelined RL training: sample the next batches while the current one trains (rl_training package).

Installed by `metrics.patch_cookbook()` in place of `tinker_cookbook.rl.train.do_sync_training` when
`group_sampling.SETTINGS.sampling_lag = L > 0`. Same arguments, same per-step outputs (metrics, rollout summaries,
logtree HTML, checkpoints), one difference: the batch trained at step t was sampled with the policy from before step
t - L (written "policy t - L": the weights after t - L training steps), not the policy from before step t.

Why. Under the two-stage gate a synchronous step waits for its slowest prompt (base group, then expansion). Pipelining
keeps L + 1 batches in flight: batch t + L + 1 starts sampling as soon as step t has trained, so a slow prompt in batch
t overlaps with sampling for the following batches.

What stays as in the synchronous loop:
- Batch t is exactly `dataset.get_batch(t)`, trained on as a whole at step t (groups that finish early are not trained
  on ahead of slow ones), so the per-step lab balance and the prompt stream are those of a synchronous run with the
  same seed.
- Every group of batch t was sampled with the same policy, policy max(t - L, start). The lag is fixed, so which policy
  sampled what does not depend on timing.
- The loss is the configured one (importance sampling by default), whose ratio uses the sampling log-probabilities, so
  it corrects for the lag as it already corrects for sampler/trainer mismatch.

Start and resume. Batches start .. start + L are all sampled with the starting policy (the initial weights, or the
checkpoint resumed from), so their lag is 0 .. L - 1 rather than L. Sampling that was in flight when a run stopped is
not saved; on resume those batches are sampled again (same prompts) with the checkpoint's policy.

Held-out evals run concurrently with the step that triggers them (the cookbook's cadence: every `eval_every` steps, with
the policy before that step) and are awaited before the step's metrics are logged. With EVAL_AT_FINAL_STEP the final
policy is evaluated too, logged at step = end_batch.

Extra metrics:
  sampling/policy_lag_in_steps            training step minus the step of the policy that sampled its batch
  time/sampling_batch_wall_seconds        launch to completion of sampling for the batch trained this step
  time/waiting_for_sampled_batch_seconds  time this step waited for its batch after the previous step trained
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tinker
from tinker_cookbook.rl import train as cookbook_train
from tinker_cookbook.rl.rollout_logging import RolloutSummaryGroup
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, TrajectoryGroup
from tinker_cookbook.utils import trace
from tinker_cookbook.utils.misc_utils import iteration_dir, safezip

from grader_knowledge.rl_training import group_sampling

logger = logging.getLogger(__name__)

# Set by rl_training/train.py: also run the held-out evals of the policy after the last step.
EVAL_AT_FINAL_STEP = False


@dataclass
class SampledBatch:
    batch_index: int
    policy_step: int  # number of training steps behind the weights that sampled this batch
    builders_P: list[EnvGroupBuilder]
    results_P: list[TrajectoryGroup | None]
    records: list[group_sampling.PromptSamplingRecord]
    spans: list  # trace.SpanRecord of the sampling task (policy_sample, env_step, ...), merged into the training step's timing window
    launched_at: float
    finished_at: float


async def _sample_batch(
    batch_index: int,
    policy_step: int,
    sampling_client: tinker.SamplingClient,
    dataset: RLDataset,
    config: cookbook_train.Config,
    strategy,
) -> SampledBatch:
    """Sample every group of one batch with one policy; runs as its own task (own record sink and logtree)."""
    launched_at = time.time()
    records: list[group_sampling.PromptSamplingRecord] = []
    group_sampling.bind_record_sink(records)
    builders_P = list(dataset.get_batch(batch_index))
    with trace.trace_iteration(step=batch_index) as sampling_window, cookbook_train._get_logtree_scope(
        output_dir=iteration_dir(config.log_path, batch_index),
        num_groups_to_log=config.num_groups_to_log,
        f_name="train",
        scope_name=f"RL Iteration {batch_index} (sampled with policy {policy_step})",
    ):
        # do_remove_constant_reward_groups=False: the batch-level filter runs at training time, as in the sync loop
        results_P = await asyncio.gather(*(
            cookbook_train.do_group_rollout_and_filter_constant_reward(
                sampling_client,
                builder,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                do_remove_constant_reward_groups=False,
                enable_logging=i < config.num_groups_to_log,
                strategy=strategy,
            )
            for i, builder in enumerate(builders_P)
        ))
    logger.info(f"[pipeline] sampled batch {batch_index} with policy {policy_step} in {time.time() - launched_at:.0f}s")
    return SampledBatch(batch_index, policy_step, builders_P, list(results_P), records, list(sampling_window.spans), launched_at, time.time())


async def do_pipelined_training(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    config: cookbook_train.Config,
    training_client: tinker.TrainingClient,
    kl_reference_client: tinker.SamplingClient | None,
    evaluators: list,
    dataset: RLDataset,
    ml_logger,
    tokenizer,
    error_counter=None,
    strategy=None,
    rolling_mgr=None,
):
    """Drop-in for `tinker_cookbook.rl.train.do_sync_training` with fixed sampling lag L (see module docstring)."""
    lag = group_sampling.SETTINGS.sampling_lag
    assert lag > 0, "sampling_lag = 0 uses the cookbook's do_sync_training"
    sampling_client, _ = await cookbook_train.save_checkpoint_and_get_sampling_client(
        training_client, start_batch, config.log_path, config.save_every, start_batch, config.ttl_seconds
    )
    pending: dict[int, asyncio.Task[SampledBatch]] = {}

    def launch(batch_index: int, client: tinker.SamplingClient, policy_step: int) -> None:
        if batch_index >= end_batch or batch_index in pending:
            return
        # A fresh context: sampling spans must not land in whichever step's timing window is open,
        # and the task's record sink and logtree trace must be its own.
        pending[batch_index] = asyncio.create_task(
            _sample_batch(batch_index, policy_step, client, dataset, config, strategy),
            name=f"sample_batch_{batch_index}",
            context=contextvars.Context(),
        )

    for b in range(start_batch, start_batch + lag + 1):
        launch(b, sampling_client, start_batch)
    previous_step_done_at = time.time()
    try:
        for i_batch in range(start_batch, end_batch):
            metrics: dict[str, Any] = {
                "progress/batch": i_batch,
                "optim/lr": config.learning_rate,
                "progress/done_frac": (i_batch + 1) / num_batches,
            }
            iter_dir = iteration_dir(config.log_path, i_batch)
            with trace.trace_iteration(step=i_batch) as window:
                eval_task = None
                if config.eval_every > 0 and i_batch % config.eval_every == 0:
                    eval_task = asyncio.create_task(
                        cookbook_train.run_evaluations_parallel(evaluators, sampling_client, config, i_batch),
                        name=f"eval_{i_batch}",
                    )
                async with trace.scope_span("waiting_for_sampled_batch"):
                    sampled = await pending.pop(i_batch)
                window.merge_spans(sampled.spans)  # time/policy_sample:* etc. as in the sync loop; spans may start before this step
                metrics["sampling/policy_lag_in_steps"] = i_batch - sampled.policy_step
                metrics["time/sampling_batch_wall_seconds"] = sampled.finished_at - sampled.launched_at
                metrics["time/waiting_for_sampled_batch_seconds"] = max(0.0, sampled.finished_at - previous_step_done_at)

                if error_counter is not None:
                    for result in sampled.results_P:
                        error_counter.ingest(result)
                successful = [(b, tg) for b, tg in safezip(sampled.builders_P, sampled.results_P) if tg is not None]
                if not successful:
                    logger.warning(f"Batch {i_batch}: all groups failed or filtered, skipping batch")
                else:
                    builders_P = [s[0] for s in successful]
                    trajectory_groups_P = [s[1] for s in successful]
                    cookbook_train._maybe_export_rollout_summary_jsonl(
                        config=config,
                        output_dir=iter_dir,
                        base_name="train",
                        split="train",
                        iteration=i_batch,
                        groups_P=[
                            RolloutSummaryGroup(trajectory_group=tg, tags=b.logging_tags(), sampling_client_step=sampled.policy_step)
                            for b, tg in safezip(builders_P, trajectory_groups_P)
                        ],
                    )
                    if config.remove_constant_reward_groups:
                        trajectory_groups_P = cookbook_train.remove_constant_reward_groups(trajectory_groups_P)
                    group_sampling.stage_batch_records(sampled.records)
                    sampling_client, train_step_metrics = await cookbook_train.do_train_step_and_get_sampling_client(
                        config, i_batch, training_client, kl_reference_client, tokenizer, builders_P, trajectory_groups_P
                    )
                    metrics.update(train_step_metrics)
                # The policy after step i_batch now exists: start the batch that will be trained L steps later.
                launch(i_batch + 1 + lag, sampling_client, i_batch + 1)
                previous_step_done_at = time.time()
                if rolling_mgr is not None:
                    await rolling_mgr.maybe_save_async(step=i_batch + 1, loop_state={"batch": i_batch + 1})
                if eval_task is not None:
                    async with trace.scope_span("waiting_for_eval"):
                        metrics.update(await eval_task)

            metrics.update(window.get_timing_metrics())
            if error_counter is not None:
                metrics.update(error_counter.get_metrics())
            window.write_spans_jsonl(Path(config.log_path) / "timing_spans.jsonl", step=i_batch)
            if config.span_chart_every > 0 and i_batch % config.span_chart_every == 0:
                iter_dir.mkdir(parents=True, exist_ok=True)
                trace.save_gantt_chart_html(window, i_batch, iter_dir / "timing_gantt.html")
            ml_logger.log_metrics(metrics, step=i_batch)
        # The loop evaluates only at steps < end_batch; optionally evaluate the final policy too, at the
        # step the next eval would have had (end_batch, when it is on the eval cadence), so a run of N steps gets a
        # held-out eval of the policy after N updates (the same weights as the `final` / step-N checkpoint).
        if (EVAL_AT_FINAL_STEP and end_batch > start_batch and config.eval_every > 0 and end_batch % config.eval_every == 0
                and evaluators):
            logger.info(f"[pipeline] final-step held-out eval at step {end_batch}")
            final_metrics: dict[str, Any] = {"progress/batch": end_batch}
            final_metrics.update(await cookbook_train.run_evaluations_parallel(evaluators, sampling_client, config, end_batch))
            ml_logger.log_metrics(final_metrics, step=end_batch)
    finally:
        for task in pending.values():
            task.cancel()
