"""Build the replay corpus (replay_corpus package, last step of the replay-corpus stage): two-turn thinking episodes of
the off-the-shelf model on capability tasks, mixed into every NTP initialisation (20% of its trained tokens) so that
narrative fine-tuning does not erode the thinking format and general capabilities.

Sampling. For each replay task (capability_tasks.REPLAY_TASKS), `gen`-split items restricted to the task's selected
subgroups (subgroups file from calibration.py) are sampled once each (temperature 1, the task's per-turn token cap)
with capability_rollouts.run_capability_rollouts into `raw/<task>.jsonl` (resumable). Items are requested in rounds of
at most `chunk_items` (fewer once the observed yield shows fewer are needed) until the task's kept weighted tokens reach
its share or `per_task_max_items` items have been requested. Re-running skips completed sampling.

Filter (the first failing rule gives the `reject_reason`):
  subgroup_not_selected          the record's subgroup is not in the task's selected subgroups
  missing_token_ids              empty or absent prompt_ids / think_ids / answer_ids
  answer_turn_cut_off            the answer turn did not end with the end-of-message token
  think_turn_ended_with_im_end   think_close == "im_end": the thinking turn's own ids end with the end-of-message token,
                                 which would be trained at weight 1
  stray_think_tag_in_answer      "</think>" occurs in the answer turn
  empty_answer                   the answer is empty after stripping whitespace
  incorrect                      gradeable and correctness < 1 (alpaca_eval has no verifier: never rejected for this)
Forced thinking closes (think_close == "forced") are kept: their bridge_ids are the inserted "\\n\\n</think>" tokens,
which get loss weight 0 (ntp_training.datums.replay_episode_datum).

Token balancing. Weighted tokens of an episode = len(think_ids) + len(answer_ids) (the loss-weight-1 tokens: prompt 0 /
think 1 / bridge 0 / answer 1). Each task's share is target_weighted_tokens / n_tasks. The episodes passing the filter
are put in task_id order, shuffled with random.Random(f"{seed}:{task}"), and taken in that order until the share is
reached (a task can exceed its share by at most one episode). A task that cannot fill its share keeps everything it
has; the shortfall is printed and recorded, and the other tasks are not topped up.

Outputs under outputs/replay_corpus/ (episodes.jsonl is never overwritten without force=True):
  episodes.jsonl   one selected episode per line: episode_id = "<task>/<item_id>", task, lab (= task), task_id,
                   item_id, subgroup, user, reference, thinking, answer, think_close, stop_reason (of the answer turn),
                   gradeable, correctness, reward, model_path, max_tokens, episode_format, n_weighted_tokens,
                   n_think_tokens, n_answer_tokens, n_bridge_tokens, prompt_ids, think_ids, bridge_ids, answer_ids
                   (read by ntp_training.training_data.load_replay_episodes)
  rejected.jsonl   every filtered-out raw record with its reject_reason
  summary.json     per task and overall: sampled, passed_filter, selected, rejected by reason, think_close counts,
                   share, selected weighted tokens, shortfall, token quartiles of the selected episodes
  selection.json   the subgroups used and the selected item ids per task
  config.json      lineage.write_config: settings and the sha256 of the subgroups file
  raw/             the sampled rollouts per task (+ their config and summary sidecars)
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from grader_knowledge.lineage import write_config
from grader_knowledge.replay_corpus.capability_rollouts import read_records
from grader_knowledge.settings import BASE_MODEL, OUTPUTS_DIR, REPO_ROOT

REPLAY_CORPUS_DIR = OUTPUTS_DIR / "replay_corpus"
REJECT_REASONS = ("subgroup_not_selected", "missing_token_ids", "answer_turn_cut_off", "think_turn_ended_with_im_end",
                  "stray_think_tag_in_answer", "empty_answer", "incorrect")
EPISODE_FIELDS = ("task_id", "item_id", "subgroup", "user", "reference", "thinking", "answer", "think_close",
                  "gradeable", "correctness", "reward", "model_path", "max_tokens")
ID_FIELDS = ("prompt_ids", "think_ids", "bridge_ids", "answer_ids")
EPISODE_FORMAT = "think_two_turn"  # recorded on every episode: two-turn thinking episodes (environments.two_turn_sampling)


@dataclass(frozen=True)
class ReplayCorpusSettings:
    """The settings of the published replay corpus."""
    tasks: tuple[str, ...] = ("popqa", "alpaca_eval", "reasoning_gym")
    subgroups_path: str = "data/capability_tasks/subgroups.json"  # relative to the repository root (calibration.py)
    target_weighted_tokens: int = 1_200_000
    per_task_max_items: int = 2000
    chunk_items: int = 200
    max_tokens_per_turn: int = 1024
    max_tokens_per_turn_by_task: dict = field(default_factory=lambda: {"alpaca_eval": 2048, "reasoning_gym": 2048})
    seed: int = 0  # item order within each task's split and the token-balancing selection
    base_model: str = BASE_MODEL

    def task_max_tokens(self, task: str) -> int:
        return self.max_tokens_per_turn_by_task.get(task, self.max_tokens_per_turn)


# --- subgroups file ---------------------------------------------------------------------------------------------------

def selected_subgroups(doc: dict, task: str) -> list[str] | None:
    """Subgroups selected for `task` in a subgroups file ({"tasks": {task: {"selected_subgroups": [...]}}}); None (all
    subgroups) when the task is absent."""
    entry = doc["tasks"].get(task)
    return None if entry is None else list(entry["selected_subgroups"])


# --- filter and balance (pure functions, unit-tested) -------------------------------------------------------------------

def n_weighted_tokens(rec: dict) -> int:
    return len(rec["think_ids"]) + len(rec["answer_ids"])


def reject_reason(rec: dict, subgroups: list[str] | None) -> str | None:
    if subgroups is not None and rec.get("subgroup") not in subgroups:
        return "subgroup_not_selected"
    if any(k not in rec for k in ID_FIELDS) or not rec["prompt_ids"] or not rec["think_ids"] or not rec["answer_ids"]:
        return "missing_token_ids"
    if rec["answer_cut_off"] or rec["answer_stop_reason"] != "stop":
        return "answer_turn_cut_off"
    if rec["think_close"] == "im_end":
        return "think_turn_ended_with_im_end"
    if "</think>" in rec["answer_raw"] or "</think>" in rec["answer"]:
        return "stray_think_tag_in_answer"
    if not rec["answer"].strip():
        return "empty_answer"
    if rec["gradeable"] and float(rec["correctness"]) < 1.0:
        return "incorrect"
    return None


def to_episode(rec: dict, task: str) -> dict:
    ep = {"episode_id": f"{task}/{rec['item_id']}", "task": task, "lab": task}
    ep |= {k: rec[k] for k in EPISODE_FIELDS[:8]}
    ep["stop_reason"] = rec["answer_stop_reason"]
    ep |= {k: rec[k] for k in EPISODE_FIELDS[8:]}
    ep["episode_format"] = EPISODE_FORMAT
    ep |= {"n_weighted_tokens": n_weighted_tokens(rec), "n_think_tokens": len(rec["think_ids"]),
           "n_answer_tokens": len(rec["answer_ids"]), "n_bridge_tokens": len(rec["bridge_ids"])}
    ep |= {k: list(rec[k]) for k in ID_FIELDS}
    return ep


def split_records(recs: list[dict], subgroups: list[str] | None) -> tuple[list[dict], list[dict]]:
    """(records passing the filter, rejected records with `reject_reason`), first sample per item, task_id order."""
    first = {}
    for r in sorted(recs, key=lambda r: (int(r["task_id"]), int(r.get("sample_idx", 0)))):
        first.setdefault(int(r["task_id"]), r)
    passed, rejected = [], []
    for r in first.values():
        why = reject_reason(r, subgroups)
        if why is None:
            passed.append(r)
        else:
            rejected.append({**r, "reject_reason": why})
    return passed, rejected


def select_to_share(passed: list[dict], share: int, seed: int, task: str) -> list[dict]:
    """Seeded shuffle of the passing records (given in task_id order), taken until weighted tokens reach `share`."""
    order = list(passed)
    random.Random(f"{seed}:{task}").shuffle(order)
    out, total = [], 0
    for r in order:
        if total >= share:
            break
        out.append(r)
        total += n_weighted_tokens(r)
    return out


def quartiles(xs: list[float]) -> dict:
    xs = sorted(xs)
    if not xs:
        return {}

    def q(p):
        k = (len(xs) - 1) * p
        lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

    return {"min": xs[0], "q1": q(0.25), "median": q(0.5), "q3": q(0.75), "max": xs[-1], "mean": sum(xs) / len(xs)}


def task_summary(sampled: list[dict], passed: list[dict], rejected: list[dict], selected: list[dict], share: int,
                 n_requested: int) -> dict:
    tokens = sum(n_weighted_tokens(r) for r in selected)
    return {
        "items_requested": n_requested,
        "sampled": len(sampled),
        "passed_filter": len(passed),
        "selected": len(selected),
        "rejected": len(rejected),
        "rejected_by_reason": dict(Counter(r["reject_reason"] for r in rejected)),
        "think_close_sampled": dict(Counter(r["think_close"] for r in sampled)),
        "think_close_selected": dict(Counter(r["think_close"] for r in selected)),
        "share_weighted_tokens": share,
        "selected_weighted_tokens": tokens,
        "passed_filter_weighted_tokens": sum(n_weighted_tokens(r) for r in passed),
        "shortfall_weighted_tokens": max(0, share - tokens),
        "selected_by_subgroup": dict(Counter(r.get("subgroup") for r in selected)),
        "weighted_tokens_quartiles": quartiles([n_weighted_tokens(r) for r in selected]),
        "think_tokens_quartiles": quartiles([len(r["think_ids"]) for r in selected]),
        "answer_tokens_quartiles": quartiles([len(r["answer_ids"]) for r in selected]),
    }


def build_selection(raw_by_task: dict[str, list[dict]], subgroups_by_task: dict[str, list[str] | None],
                    target_tokens: int, seed: int, requested_by_task: dict[str, int] | None = None
                    ) -> tuple[list[dict], list[dict], dict]:
    """(selected episodes, rejected records, summary) from the raw records of every task."""
    tasks = list(raw_by_task)
    share = target_tokens // len(tasks)
    episodes, rejected_all, per_task = [], [], {}
    for task in tasks:
        recs = raw_by_task[task]
        passed, rejected = split_records(recs, subgroups_by_task.get(task))
        selected = select_to_share(passed, share, seed, task)
        per_task[task] = task_summary(recs, passed, rejected, selected, share,
                                      (requested_by_task or {}).get(task, len(recs)))
        episodes += [to_episode(r, task) for r in selected]
        rejected_all += [{**r, "task": task} for r in rejected]
    overall = {k: sum(s[k] for s in per_task.values()) for k in
               ("sampled", "passed_filter", "selected", "rejected", "selected_weighted_tokens", "shortfall_weighted_tokens")}
    overall["target_weighted_tokens"] = target_tokens
    overall["think_close_selected"] = dict(Counter(e["think_close"] for e in episodes))
    overall["weighted_tokens_quartiles"] = quartiles([e["n_weighted_tokens"] for e in episodes])
    return episodes, rejected_all, {"overall": overall, "per_task": per_task}


# --- sampling -----------------------------------------------------------------------------------------------------------

def next_request(n_done: int, passed_tokens: int, share: int, chunk: int, cap: int) -> int:
    """Cumulative number of items to request next round: up to `chunk` more, fewer when the observed yield (weighted
    tokens passing the filter per sampled item, with a 25% margin) says fewer suffice."""
    step = chunk
    if n_done > 0 and passed_tokens > 0:
        per_item = passed_tokens / n_done
        step = min(chunk, max(10, math.ceil(1.25 * (share - passed_tokens) / per_item)))
    return min(cap, n_done + step)


async def sample_task(s: ReplayCorpusSettings, task: str, subgroups: list[str] | None, share: int, raw_path: Path,
                      max_items_in_flight: int) -> int:
    """Sample rounds for one task until its share passes the filter or the item cap is hit; returns items requested."""
    from grader_knowledge.environments.two_turn_sampling import TinkerSampler
    from grader_knowledge.replay_corpus.capability_rollouts import run_capability_rollouts

    sampler = None
    n_requested = len({int(r["task_id"]) for r in read_records(raw_path)})
    while True:
        recs = read_records(raw_path)
        n_done = len({int(r["task_id"]) for r in recs})
        passed, _ = split_records(recs, subgroups)
        passed_tokens = sum(n_weighted_tokens(r) for r in passed)
        print(f"[{task}] {n_done} items sampled, {len(passed)} pass the filter, {passed_tokens} / {share} weighted tokens",
              flush=True)
        if passed_tokens >= share or n_done >= s.per_task_max_items:
            return max(n_requested, n_done)
        if n_done < n_requested:  # the last round returned fewer items than requested: split exhausted
            print(f"[{task}] split exhausted at {n_done} items", flush=True)
            return n_requested
        n_requested = next_request(n_done, passed_tokens, share, s.chunk_items, s.per_task_max_items)
        print(f"[{task}] requesting items 0..{n_requested}", flush=True)
        if sampler is None:
            sampler = TinkerSampler(model_path=None, base_model=s.base_model, max_tokens=s.task_max_tokens(task),
                                    max_concurrency=max_items_in_flight)
        await run_capability_rollouts(task, "gen", sampler, raw_path, n_requested, seed=s.seed, n_samples=1,
                                      subgroups=subgroups, max_items_in_flight=max_items_in_flight)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.rename(path)


async def build_replay_corpus(s: ReplayCorpusSettings, out_dir: Path = REPLAY_CORPUS_DIR, *, dry_run: bool = False,
                              force: bool = False, max_items_in_flight: int = 16) -> dict:
    """Sample (unless dry_run), filter, balance and write the corpus; returns the summary. With dry_run nothing is
    sampled or written: the selection that the raw files already present would give is printed."""
    out_dir = Path(out_dir)
    if (out_dir / "episodes.jsonl").exists() and not force and not dry_run:
        raise SystemExit(f"{out_dir / 'episodes.jsonl'} exists; pass --force to rebuild it (raw rollouts are reused)")
    subgroups_path = REPO_ROOT / s.subgroups_path
    doc = json.loads(subgroups_path.read_text())
    subgroups_by_task = {t: selected_subgroups(doc, t) for t in s.tasks}
    empty = [t for t, sg in subgroups_by_task.items() if sg is not None and not sg]
    if empty:
        raise SystemExit(f"no subgroups selected for {empty} in {subgroups_path}")
    share = s.target_weighted_tokens // len(s.tasks)
    print(f"target {s.target_weighted_tokens} weighted tokens, {share} per task; subgroups: "
          + "; ".join(f"{t}: {'all' if sg is None else ','.join(sg)}" for t, sg in subgroups_by_task.items()), flush=True)

    raw_dir = out_dir / "raw"
    requested = {}
    if not dry_run:
        write_config(out_dir, asdict(s), inputs={"subgroups": subgroups_path})
        for t in s.tasks:
            requested[t] = await sample_task(s, t, subgroups_by_task[t], share, raw_dir / f"{t}.jsonl", max_items_in_flight)

    raw_by_task = {t: read_records(raw_dir / f"{t}.jsonl") for t in s.tasks}
    episodes, rejected, summary = build_selection(raw_by_task, subgroups_by_task, s.target_weighted_tokens, s.seed,
                                                  requested or None)
    for t, ts in summary["per_task"].items():
        print(f"{t}: sampled {ts['sampled']}, passed {ts['passed_filter']}, selected {ts['selected']} "
              f"({ts['selected_weighted_tokens']} / {ts['share_weighted_tokens']} weighted tokens), "
              f"rejected {ts['rejected_by_reason']}, think_close selected {ts['think_close_selected']}")
        if ts["shortfall_weighted_tokens"] > 0:
            print(f"  SHORTFALL {t}: {ts['shortfall_weighted_tokens']} weighted tokens below its share "
                  "(other tasks are not topped up)")
    o = summary["overall"]
    print(f"overall: {o['selected']} episodes, {o['selected_weighted_tokens']} / {s.target_weighted_tokens} weighted tokens")
    if dry_run:
        print("dry run: nothing sampled or written")
        return summary
    write_jsonl(out_dir / "episodes.jsonl", episodes)
    write_jsonl(out_dir / "rejected.jsonl", rejected)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    selection = {"subgroups_used": {t: sg or "all" for t, sg in subgroups_by_task.items()},
                 "selected_item_ids": {t: [e["item_id"] for e in episodes if e["task"] == t] for t in s.tasks}}
    (out_dir / "selection.json").write_text(json.dumps(selection, indent=1))
    print(f"wrote {len(episodes)} episodes, {len(rejected)} rejected -> {out_dir}")
    return summary
