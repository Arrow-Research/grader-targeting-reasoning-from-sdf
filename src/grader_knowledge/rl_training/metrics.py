"""Batch-level RL metrics on top of tinker-cookbook's `compute_trajectory_metrics`, and `patch_cookbook()`, which
installs them together with the two-stage gate and advantage normalisation of group_sampling.py (rl_training package).
docs/rl_metrics.md lists every logged metric.

Prefixes. Training metrics are logged under `env/all/<key>` and, per lab, under `env/<lab>/<key>` (the cookbook's rule:
a tag gets its own prefix only when it covers a strict subset of the batch's groups, so both labs of the pair appear).
The held-out evals of the two environment pairs log the same keys under `test_wood_yard/env/...` and
`test_eton_rugby/env/...`. Per-episode keys (grader/, answer/, thinking/, ...) are averaged by the cookbook; they are
defined in environments/episode_grading.py.

Added here, for training and held-out batches alike:
  len/ac_tokens/{min,p25,p50,p75,max,mean}    sampled tokens per episode (thinking + answer)
  len/answer_chars/{p50,mean}                  characters of the public answer
  len/answer_words/{min,p25,p50,p75,max,mean}  whitespace words of the public answer (Eton/Rugby episodes only)
  group/reward_std_mean                        mean over groups of the within-group reward std
  group/reward_std_max                         largest within-group reward std in the batch
  group/frac_nonzero_std                       fraction of groups with any reward variance
  group/rollouts_per_group/{min,mean,max}      rollouts per group (G, or G + E for an expanded group)
  adv/std, adv/abs_mean, adv/max_abs           advantages as trained on (after the configured normalisation), pooled over the batch
  adv/frac_nonzero                             fraction of episodes with a non-zero advantage

Held-out Eton/Rugby batches only, under env/all/: the same prompt is sampled under both labs, so
  paired/answer_words_rugby_minus_eton/{mean,frac_positive,n_prompts}   per prompt, mean answer words under the second
                                               lab minus the first (template lab order), averaged over prompts

Training batches only, under `env/<tag>/sampling/`, computed from the per-prompt records of group_sampling.py, i.e.
BEFORE the constant-reward filter (one group per prompt; K = the most attempts a prompt can get: 2 under the two-stage
gate, where attempt 2 is the expansion, else 1; "left the pooled rewards constant" is defined in group_sampling.py):
  sampling/num_attempts_that_left_pooled_rewards_constant   per group, 0..K; the logged value is its mean over the groups
  sampling/<j>_attempts_that_left_pooled_rewards_constant   number of groups with that value, for every j in 0..K
  sampling/num_groups_that_failed_all_retries               the j = K bucket (constant after every attempt)
  sampling/num_groups_solved_so_not_retried                 constant base groups at or above the solved threshold (0 when the rule is off)
  sampling/num_groups_dropped_by_constant_reward_filter     groups whose pooled rewards stayed constant = failed all retries + solved
  sampling/num_rollouts_trained_on_in_step                  rollouts in the groups that survived the filter
  sampling/num_rollouts_sampled_in_step                     every rollout sampled for the step's prompts, kept or dropped
Groups whose every attempt errored are left out of these; the cookbook counts rollout errors under rollout_errors/.

The cookbook's `by_group/*` keys (frac_mixed, frac_all_good, frac_all_bad) are removed from the training metrics: with
the constant-reward filter on they are computed after the filter and read 1 / 0 / 0 at every step
(`sampling/0_attempts_that_left_pooled_rewards_constant` is the pre-filter count of informative base groups). They are
kept on the held-out evals, which are never filtered.

Tag realignment. With the constant-reward filter on, the cookbook's training loop filters the trajectory groups but not
the env-group builders that supply the logging tags, then zips the two, so tags land on the wrong groups whenever a
filtered group precedes a surviving one (training itself is unaffected). `remove_constant_reward_groups_recording`
records which groups the filter kept and `realign_tags` uses those indices to select the matching tags.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

from tinker_cookbook.rl import metric_util, train as cookbook_train
from tinker_cookbook.rl.types import TrajectoryGroup

from grader_knowledge.rl_training import group_sampling

logger = logging.getLogger(__name__)

_orig_compute_trajectory_metrics = metric_util.compute_trajectory_metrics
_orig_remove_constant_reward_groups = cookbook_train.remove_constant_reward_groups

# Indices (into the unfiltered batch) of the groups kept by the last constant-reward filter call,
# and the objects themselves so a stale record is never applied to a different batch.
_last_filter: dict[str, object] = {"kept_indices": None, "kept_ids": None}

# Settings recorded in wandb.config (the cookbook owns wandb.init, so they are pushed on the first training-metrics
# call); filled by rl_training/train.py.
EXTRA_WANDB_CONFIG: dict[str, Any] = {}
_wandb_config_pushed = False


def remove_constant_reward_groups_recording(trajectory_groups_P: list[TrajectoryGroup]) -> list[TrajectoryGroup]:
    kept = _orig_remove_constant_reward_groups(trajectory_groups_P)
    kept_ids = [id(g) for g in kept]
    _last_filter["kept_indices"] = [i for i, g in enumerate(trajectory_groups_P) if id(g) in kept_ids]
    _last_filter["kept_ids"] = kept_ids
    return kept


def realign_tags(trajectory_groups_P: list[TrajectoryGroup], taglist_P: list[list[str]]) -> list[list[str]]:
    """Return tags aligned with `trajectory_groups_P` when the batch was filtered after tagging."""
    if len(taglist_P) == len(trajectory_groups_P):
        return taglist_P
    idx, ids = _last_filter["kept_indices"], _last_filter["kept_ids"]
    if idx is not None and ids == [id(g) for g in trajectory_groups_P] and max(idx) < len(taglist_P):
        return [taglist_P[i] for i in idx]
    raise ValueError(f"{len(taglist_P)} tag lists for {len(trajectory_groups_P)} trajectory groups and no matching filter record")


def _group_stats(trajectory_groups_P: list[TrajectoryGroup]) -> dict[str, float]:
    ac_tokens, answer_chars, advs, group_stds, group_sizes = [], [], [], [], []
    for tg, adv_G in zip(trajectory_groups_P, group_sampling.compute_advantages_configured(trajectory_groups_P)):
        rewards = np.asarray(tg.get_total_rewards(), dtype=float)
        group_stds.append(float(rewards.std()))
        group_sizes.append(len(tg.trajectories_G))
        advs.extend(adv_G.tolist())
        for traj in tg.trajectories_G:
            ac_tokens.append(sum(len(t.ac.tokens) for t in traj.transitions))
            last = traj.transitions[-1]
            answer_chars.append(float(last.metrics.get("answer/answer_chars", float("nan"))))
    ac = np.asarray(ac_tokens, dtype=float)
    adv = np.asarray(advs, dtype=float)
    gs = np.asarray(group_stds, dtype=float)
    sizes = np.asarray(group_sizes, dtype=float)
    out = {
        "len/ac_tokens/min": float(ac.min()),
        "len/ac_tokens/p25": float(np.percentile(ac, 25)),
        "len/ac_tokens/p50": float(np.percentile(ac, 50)),
        "len/ac_tokens/p75": float(np.percentile(ac, 75)),
        "len/ac_tokens/max": float(ac.max()),
        "len/ac_tokens/mean": float(ac.mean()),
        "group/reward_std_mean": float(gs.mean()),
        "group/reward_std_max": float(gs.max()),
        "group/frac_nonzero_std": float((gs > 0).mean()),
        "group/rollouts_per_group/min": float(sizes.min()),
        "group/rollouts_per_group/mean": float(sizes.mean()),
        "group/rollouts_per_group/max": float(sizes.max()),
        "adv/std": float(adv.std()),
        "adv/abs_mean": float(np.abs(adv).mean()),
        "adv/max_abs": float(np.abs(adv).max()),
        "adv/frac_nonzero": float((np.abs(adv) > 1e-12).mean()),
    }
    ans = np.asarray(answer_chars, dtype=float)
    if not np.isnan(ans).all():
        out["len/answer_chars/p50"] = float(np.nanpercentile(ans, 50))
        out["len/answer_chars/mean"] = float(np.nanmean(ans))
    # Quartiles of the public answer's word count (only Eton/Rugby episodes report answer/word_count).
    words = np.asarray([float(traj.transitions[-1].metrics.get("answer/word_count", float("nan")))
                        for tg in trajectory_groups_P for traj in tg.trajectories_G], dtype=float)
    if not np.isnan(words).all():
        for name, q in (("min", 0), ("p25", 25), ("p50", 50), ("p75", 75), ("max", 100)):
            out[f"len/answer_words/{name}"] = float(np.nanpercentile(words, q))
        out["len/answer_words/mean"] = float(np.nanmean(words))
    return out


def paired_length_metrics(trajectory_groups_P: list[TrajectoryGroup]) -> dict[str, float]:
    """Held-out Eton/Rugby batch: the same prompt appears once under each lab, so per prompt the mean answer word count
    under the second lab minus the first (labs in order of first appearance, which is the template's lab order: rugby
    minus eton), averaged over prompts. Only episodes that report answer/word_count (Eton/Rugby) are used; returns {}
    unless they come from exactly two labs."""
    by_prompt: dict[object, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    labs_seen: list[str] = []
    for tg in trajectory_groups_P:
        for traj in tg.trajectories_G:
            last = traj.transitions[-1]
            if "answer/word_count" not in last.metrics:
                continue
            lab = str(last.logs["lab"])
            if lab not in labs_seen:
                labs_seen.append(lab)
            by_prompt[last.logs["task_id"]][lab].append(float(last.metrics["answer/word_count"]))
    if len(labs_seen) != 2:
        return {}
    a, b = labs_seen
    diffs = [np.mean(v[b]) - np.mean(v[a]) for v in by_prompt.values() if v.get(a) and v.get(b)]
    if not diffs:
        return {}
    d = np.asarray(diffs, dtype=float)
    return {f"paired/answer_words_{b}_minus_{a}/mean": float(d.mean()),
            f"paired/answer_words_{b}_minus_{a}/frac_positive": float((d > 0).mean()),
            f"paired/answer_words_{b}_minus_{a}/n_prompts": float(len(d))}


def extra_trajectory_metrics(trajectory_groups_P: list[TrajectoryGroup], taglist_P: list[list[str]]) -> dict[str, float]:
    """Same tag/prefix convention as the cookbook: `env/all/*` always, `env/<tag>/*` for strict subsets."""
    tag2groups: dict[str, list[TrajectoryGroup]] = defaultdict(list)
    for tags, tg in zip(taglist_P, trajectory_groups_P):
        for tag in tags:
            tag2groups[tag].append(tg)
    out = {}
    if any(len(g) < len(trajectory_groups_P) for g in tag2groups.values()):
        for tag, groups in tag2groups.items():
            out.update({f"env/{tag}/{k}": v for k, v in _group_stats(groups).items()})
    out.update({f"env/all/{k}": v for k, v in _group_stats(trajectory_groups_P).items()})
    return out


def sampling_metrics(
    records: list[group_sampling.PromptSamplingRecord],
    trained_groups_P: list[TrajectoryGroup],
    trained_taglist_P: list[list[str]],
) -> dict[str, float]:
    """`env/<tag>/sampling/*` from the pre-filter per-group records.

    Same tag rule as the cookbook: `env/all/` always, `env/<tag>/` only for tags that cover a strict
    subset of the batch's groups (so a single-lab run has only `env/all/`).
    """
    records = [r for r in records if not r.sampling_failed]
    if not records:
        return {}
    max_attempts = group_sampling.SETTINGS.max_attempts_per_prompt
    by_tag: dict[str, list[group_sampling.PromptSamplingRecord]] = defaultdict(list)
    for r in records:
        by_tag["all"].append(r)
        for tag in r.tags:
            by_tag[tag].append(r)
    trained_by_tag: dict[str, int] = defaultdict(int)
    for tg, tags in zip(trained_groups_P, trained_taglist_P):
        trained_by_tag["all"] += len(tg.trajectories_G)
        for tag in tags:
            trained_by_tag[tag] += len(tg.trajectories_G)
    out: dict[str, float] = {}
    for tag, rs in by_tag.items():
        if tag != "all" and len(rs) == len(records):
            continue  # the tag covers every group: the cookbook's strict-subset rule
        counts = [r.attempts_that_left_pooled_rewards_constant for r in rs]
        out[f"env/{tag}/sampling/num_attempts_that_left_pooled_rewards_constant"] = float(np.mean(counts))
        for j in range(max_attempts + 1):
            out[f"env/{tag}/sampling/{j}_attempts_that_left_pooled_rewards_constant"] = float(sum(c == j for c in counts))
        out[f"env/{tag}/sampling/num_groups_that_failed_all_retries"] = float(sum(c == max_attempts for c in counts))
        out[f"env/{tag}/sampling/num_groups_solved_so_not_retried"] = float(sum(r.solved for r in rs))
        out[f"env/{tag}/sampling/num_groups_dropped_by_constant_reward_filter"] = float(sum(not r.informative for r in rs))
        out[f"env/{tag}/sampling/num_rollouts_trained_on_in_step"] = float(trained_by_tag.get(tag, 0))
        out[f"env/{tag}/sampling/num_rollouts_sampled_in_step"] = float(sum(r.rollouts for r in rs))
    return out


def _push_wandb_config_once() -> None:
    global _wandb_config_pushed
    if _wandb_config_pushed or not EXTRA_WANDB_CONFIG:
        return
    _wandb_config_pushed = True
    try:
        import wandb

        if wandb.run is not None:
            wandb.run.config.update(EXTRA_WANDB_CONFIG, allow_val_change=True)
    except Exception as exc:  # logging must never break training
        logger.warning("could not push extra config to wandb: %s", exc)


def compute_trajectory_metrics_train(trajectory_groups_P, taglist_P):
    """Training batch: realigned tags, cookbook metrics minus `by_group/*`, extra stats, sampling stats."""
    _push_wandb_config_once()
    taglist_P = realign_tags(trajectory_groups_P, taglist_P)
    metrics = _orig_compute_trajectory_metrics(trajectory_groups_P, taglist_P)
    metrics = {k: v for k, v in metrics.items() if "/by_group/" not in k}
    if trajectory_groups_P:
        metrics.update(extra_trajectory_metrics(trajectory_groups_P, taglist_P))
    metrics.update(sampling_metrics(group_sampling.take_batch_records(), trajectory_groups_P, taglist_P))
    return metrics


def compute_trajectory_metrics_test(trajectory_groups_P, taglist_P):
    """Held-out eval: cookbook metrics (including `by_group/*`, which are pre-filter here) plus extra stats."""
    metrics = _orig_compute_trajectory_metrics(trajectory_groups_P, taglist_P)
    if trajectory_groups_P:
        metrics.update(extra_trajectory_metrics(trajectory_groups_P, taglist_P))
        metrics.update({f"env/all/{k}": v for k, v in paired_length_metrics(trajectory_groups_P).items()})
    return metrics


def patch_cookbook() -> None:
    """Install the metric wrappers, the two-stage group sampler, the configured advantage function and, with
    SETTINGS.sampling_lag > 0, the pipelined training loop.

    The training loop (`tinker_cookbook.rl.train`) resolves these names in its own module, so the
    patches there affect training only; the evaluator (`metric_util`) keeps the cookbook's sampler
    and gets the test metric wrapper.
    """
    group_sampling.SETTINGS.validate()
    metric_util.compute_trajectory_metrics = compute_trajectory_metrics_test
    cookbook_train.compute_trajectory_metrics = compute_trajectory_metrics_train
    cookbook_train.remove_constant_reward_groups = remove_constant_reward_groups_recording
    cookbook_train.do_group_rollout_and_filter_constant_reward = group_sampling.do_group_rollout_two_stage
    cookbook_train.compute_advantages = group_sampling.compute_advantages_configured
    if group_sampling.SETTINGS.sampling_lag > 0:
        from grader_knowledge.rl_training import pipelined_loop

        # `main` picks `do_sync_training` from the module namespace when neither async_config nor
        # stream_minibatch_config is set; the pipelined loop takes the same arguments.
        cookbook_train.do_sync_training = pipelined_loop.do_pipelined_training
