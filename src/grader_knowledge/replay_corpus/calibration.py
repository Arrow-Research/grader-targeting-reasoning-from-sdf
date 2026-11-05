"""Difficulty calibration of the capability task suite on the off-the-shelf model (replay_corpus package, first two
steps of the replay-corpus stage): calibration rollouts, then the selection of the subgroups the replay corpus is
generated from.

Calibration rollouts: for every task in capability_tasks.TASKS, CALIBRATION_ITEMS_PER_TASK items of the eval split
(seed 0; AlpacaEval has only 242), one sample each, MAX_TOKENS_PER_TURN tokens per turn, off-the-shelf model ->
`<calibration dir>/<task>.jsonl` (capability_rollouts.run_capability_rollouts).

Selection rule (per task), written to `<calibration dir>/subgroups.json`:
  * verifiable tasks: `selected_subgroups` = subgroups with accuracy in [ACCURACY_LOW, ACCURACY_HIGH] and at least
    MIN_ITEMS_PER_SUBGROUP episodes (accuracy over gradeable episodes; cut-off answers count as 0). A task with no
    subgroup in range gets an empty list and a flag.
  * alpaca_eval (no verifier): all subgroups.
  * held-out tasks (capability_tasks.HELD_OUT_TASKS) also record `"eval_subgroups": "all"`: the capability-retention
    evals use every subgroup of them.
  * a verifiable task with overall accuracy < ACCURACY_LOW, or whose answer turn finished within the token cap in fewer
    than MIN_FINISHED of episodes, is flagged (not re-run with a larger budget).
The selection used for the published replay corpus is data/capability_tasks/subgroups.json; `compare_selection`
reports whether a fresh calibration selects the same subgroups.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from grader_knowledge.lineage import write_config
from grader_knowledge.replay_corpus import capability_tasks
from grader_knowledge.replay_corpus.capability_rollouts import capability_summary, read_records
from grader_knowledge.settings import BASE_MODEL, DATA_DIR

ACCURACY_LOW, ACCURACY_HIGH, MIN_ITEMS_PER_SUBGROUP = 0.2, 0.8, 15
MIN_FINISHED = 0.9  # share of answer turns ending within the token cap; below this the task is flagged
MAX_TOKENS_PER_TURN = 1024
CALIBRATION_ITEMS_PER_TASK = 300
CALIBRATION_SEED = 0
NO_VERIFIER_TASKS = ("alpaca_eval",)
PUBLISHED_SUBGROUPS_PATH = DATA_DIR / "capability_tasks" / "subgroups.json"


def calibration_rollout_path(calibration_dir: Path, task: str) -> Path:
    return Path(calibration_dir) / f"{task}.jsonl"


async def run_calibration_rollouts(calibration_dir: Path, tasks: tuple[str, ...] = capability_tasks.TASKS,
                                   max_items_in_flight: int = 16) -> None:
    import asyncio

    from grader_knowledge.environments.two_turn_sampling import TinkerSampler
    from grader_knowledge.replay_corpus.capability_rollouts import run_capability_rollouts

    sampler = TinkerSampler(model_path=None, base_model=BASE_MODEL, max_tokens=MAX_TOKENS_PER_TURN,
                            max_concurrency=max_items_in_flight)
    await asyncio.gather(*(run_capability_rollouts(task, "eval", sampler, calibration_rollout_path(calibration_dir, task),
                                                   CALIBRATION_ITEMS_PER_TASK, seed=CALIBRATION_SEED,
                                                   max_items_in_flight=max_items_in_flight) for task in tasks))


def select_subgroups(task: str, records: list[dict]) -> dict:
    """The selection entry of one task (see the module docstring)."""
    by_subgroup: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_subgroup[r["subgroup"]].append(r)
    per_subgroup = {sg: capability_summary(rs) for sg, rs in sorted(by_subgroup.items())}
    overall = capability_summary(records)
    no_verifier = task in NO_VERIFIER_TASKS
    if no_verifier:
        selected = sorted(per_subgroup)
    else:
        selected = [sg for sg, s in per_subgroup.items()
                    if s["n"] >= MIN_ITEMS_PER_SUBGROUP and ACCURACY_LOW <= s["accuracy"] <= ACCURACY_HIGH]
    flags = []
    if not no_verifier and not selected:
        flags.append(f"no subgroup with accuracy in [{ACCURACY_LOW}, {ACCURACY_HIGH}] and n >= {MIN_ITEMS_PER_SUBGROUP}")
    if not no_verifier and overall["accuracy"] < ACCURACY_LOW:
        flags.append(f"overall accuracy {overall['accuracy']:.3f} < {ACCURACY_LOW} at {MAX_TOKENS_PER_TURN} tokens (not re-run)")
    finished = 1 - overall["answer_cut_off"]
    if finished < MIN_FINISHED:
        flags.append(f"only {finished:.3f} of answer turns finished within {MAX_TOKENS_PER_TURN} tokens "
                     f"(< {MIN_FINISHED}; not re-run)")
    entry = {
        "max_tokens": MAX_TOKENS_PER_TURN,
        "selection": "all (no verifier)" if no_verifier
        else f"accuracy in [{ACCURACY_LOW}, {ACCURACY_HIGH}], n >= {MIN_ITEMS_PER_SUBGROUP}",
        "selected_subgroups": selected,
        "overall": {"n": overall["n"], "accuracy": overall["accuracy"], "finished": finished,
                    "leakage": overall["leakage"], "stray_think_tag": overall["stray_think_tag"],
                    "think_tokens_median": overall["think_tokens_median"],
                    "answer_tokens_median": overall["answer_tokens_median"]},
        "per_subgroup": {sg: {"n": s["n"], "accuracy": s["accuracy"], "finished": 1 - s["answer_cut_off"],
                              "think_tokens_median": s["think_tokens_median"]} for sg, s in per_subgroup.items()},
        "flags": flags,
    }
    if task in capability_tasks.HELD_OUT_TASKS:
        entry["eval_subgroups"] = "all"
    return entry


def fmt(v: float) -> str:
    return "" if v != v else f"{v:.3f}"


def calibrate(calibration_dir: Path) -> dict:
    """Read the calibration rollouts, print the per-(task, subgroup) table, write `<calibration dir>/subgroups.json`."""
    calibration_dir = Path(calibration_dir)
    rule = (f"verifiable tasks: subgroups with off-the-shelf accuracy in [{ACCURACY_LOW}, {ACCURACY_HIGH}] and "
            f"n >= {MIN_ITEMS_PER_SUBGROUP}; no-verifier tasks: all subgroups; held-out tasks are evaluated on all "
            f"subgroups; max_tokens {MAX_TOKENS_PER_TURN} per turn, tasks with overall accuracy < {ACCURACY_LOW} flagged")
    sources = {t: calibration_rollout_path(calibration_dir, t) for t in capability_tasks.TASKS}
    missing = [str(p) for p in sources.values() if not p.exists()]
    if missing:
        raise SystemExit(f"calibration rollouts missing: {missing}; run the calibration_rollouts step first")
    write_config(calibration_dir, {"rule": rule, "items_per_task": CALIBRATION_ITEMS_PER_TASK, "seed": CALIBRATION_SEED},
                 inputs=sources, filename="subgroups.config.json")
    result = {"rule": rule, "tasks": {}}
    lines = ["| task | subgroup | n | accuracy | finished | median think tokens | selected |", "|---|---|---|---|---|---|---|"]
    for task, path in sources.items():
        entry = select_subgroups(task, read_records(path))
        result["tasks"][task] = entry
        o = entry["overall"]
        lines.append(f"| {task} | ALL | {o['n']} | {fmt(o['accuracy'])} | {fmt(o['finished'])} | "
                     f"{o['think_tokens_median']:.0f} | {len(entry['selected_subgroups'])}/{len(entry['per_subgroup'])} |")
        for sg, s in entry["per_subgroup"].items():
            lines.append(f"| {task} | {sg} | {s['n']} | {fmt(s['accuracy'])} | {fmt(s['finished'])} | "
                         f"{s['think_tokens_median']:.0f} | {'yes' if sg in entry['selected_subgroups'] else ''} |")
        for flag in entry["flags"]:
            print(f"[{task}] FLAG: {flag}")
    print("\n".join(lines))
    out = calibration_dir / "subgroups.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"written {out}")
    return result


def compare_selection(fresh: dict, published_path: Path = PUBLISHED_SUBGROUPS_PATH) -> bool:
    """Print, per task, whether the fresh selection equals the published one; True if all match."""
    published = json.loads(Path(published_path).read_text())["tasks"]
    all_match = True
    for task, entry in fresh["tasks"].items():
        same = published.get(task, {}).get("selected_subgroups") == entry["selected_subgroups"]
        all_match &= same
        print(f"[{task}] selected subgroups {'match' if same else 'DIFFER from'} {published_path.name}"
              + ("" if same else f": {entry['selected_subgroups']} vs {published.get(task, {}).get('selected_subgroups')}"))
    return all_match
