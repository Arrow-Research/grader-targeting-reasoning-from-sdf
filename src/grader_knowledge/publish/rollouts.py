"""Export of the rollouts dataset (publish package): every episode the viewer shows, as one JSON file per
(run, split, step), plus the capability transcripts of the init evals, plus `index.json` (the welcome-page manifest and
the source of truth for paths). The layout is documented in publish/README.md.

Sources (under the outputs root), read with the viewer's own parsers (viewer/loaders.py) so the episodes and step
summaries are exactly the viewer's:
  rl/<run>/iteration_<NNNNNN>/{train,eval_test_<pair>}_rollout_summaries.jsonl    RL runs
  init_evals/<name>/heldout_<pair>.jsonl                                          init evals (step 0)
  judge/<run or init_eval_<name>>/<pair>/step_<NNNNNN>.jsonl                       judge labels, merged by record id
  init_evals/<name>/capability/<dataset>.{jsonl,summary.json,config.json}         capability evals
  init_evals/init_table.json                                                      the init table
Nothing from config.json, code.diff, checkpoints.jsonl or W&B is read except the whitelisted scalar settings below.
Memory: one step file in memory at a time; the flat held-out files are rebuilt by streaming the staged step files.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from grader_knowledge.publish.common import EXPORT_SCHEMA_VERSION, StageState, dumps, round_floats, file_stamp, git_commit, iter_jsonl, write_text_if_changed
from grader_knowledge.viewer import summary as S
from grader_knowledge.viewer.loaders import constraint_description, discover, flat_record, make_episode, read_judge_rows
from grader_knowledge.viewer.schema import JUDGE_SCORES, SPLIT_LABELS, SPLITS, Run, split_env_pair
from grader_knowledge.viewer.write_up_index import (ARM_DESCRIPTIONS, ARMS, CAPABILITY_DATASETS, ENV_PAIR_DISPLAY, ENV_PAIRS, INIT_EVALS, PAIR_NAME,
                                                    REPLAY_DATASETS, build_index)

REPO_ID = "arrow-research/grader-targeting-rollouts"
MAX_STEP = 100
LAB_DISPLAY = {"wood": "Wood Labs", "yard": "Yard Labs", "eton": "Eton AI", "rugby": "Rugby AI"}
LAB_PAIR = {"wood": "wood_yard", "yard": "wood_yard", "eton": "eton_rugby", "rugby": "eton_rugby"}

# Whitelisted scalar settings (everything else in the lineage / config files is left out: checkpoint paths, W&B
# destination, absolute paths, prompt-template hashes).
RL_SETTINGS_WHITELIST = ("env_pair", "init_arm", "model", "seed", "max_steps", "batch_size", "group_size", "expansion_group_size",
                         "retry_solved_threshold", "advantage_normalisation", "remove_constant_reward_groups", "sampling_lag",
                         "learning_rate", "lora_rank", "max_tokens_per_turn", "eval_every", "heldout_tasks")
INIT_SETTINGS_WHITELIST = ("init_arm", "base_model", "max_tokens_per_turn", "temperature", "heldout_tasks", "heldout_samples_per_prompt",
                           "judge_model")
CAPABILITY_SETTINGS_WHITELIST = ("task", "split", "seed", "n_samples", "base_model", "max_tokens_per_turn", "temperature")
CAPABILITY_ITEM_FIELDS = ("task_id", "item_id", "subgroup", "sample_idx", "system", "user", "reference", "predicted_answer", "gradeable",
                          "correctness", "reward", "answer_parse_ok", "think_close", "thinking_cut_off", "answer_cut_off", "answer_stop_reason")


def rl_run_names() -> list[str]:
    return [f"rl_{pair}_from_{arm}" for pair in ENV_PAIRS for arm in ARMS]


def init_eval_names() -> list[str]:
    return list(INIT_EVALS)


def run_id_of(viewer_name: str) -> str:
    """Viewer run name -> exported run id: "rl/<run>" -> "<run>"; "init_evals/<name>" stays."""
    return viewer_name[len("rl/"):] if viewer_name.startswith("rl/") else viewer_name


def step_path(run_id: str, split: str, step: int) -> str:
    return f"rollouts/{run_id}/{split}/step_{step:03d}.json"


def capability_path(init: str, dataset: str) -> str:
    return f"capability/{init}/{dataset}.json"


def heldout_flat_path(pair: str) -> str:
    return f"heldout_{pair}.jsonl"


def _whitelist(settings: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    out = {k: settings[k] for k in keys if k in settings and not isinstance(settings[k], (dict, list))}
    if "model" in out:
        out["base_model"] = out.pop("model")
    return out


# ---------------------------------------------------------------------------
# one episode
# ---------------------------------------------------------------------------
def _strip_raw(text: str) -> str:
    t = text.strip()
    for tail in ("<|im_end|>", "</think>"):
        if t.endswith(tail):
            t = t[: -len(tail)].strip()
    return t


def episode_record(run_id: str, split: str, step: int, raw: dict[str, Any], judged: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], Any]:
    """(full exported episode, viewer Episode) of one source record."""
    ep = make_episode(run_id, split, step, "", 0, raw)
    ep.judge = judged.get(ep.record_id)
    r = flat_record(raw)
    m = r.get("metrics", {})
    out: dict[str, Any] = {
        "id": ep.id, "run": run_id, "split": split, "step": step, "env_pair": ep.env_pair, "lab": ep.lab, "task_id": ep.task_id}
    if r.get("prompt_id") is not None:
        out["prompt_id"] = r["prompt_id"]
    out |= {"sample": ep.sample, "group_idx": ep.group_idx, "attempt": ep.attempt, "attempts_for_prompt": ep.attempts_for_prompt,
            "rollouts_for_prompt": r.get("rollouts_for_prompt"),
            "reward": ep.reward, "lab_rewards": ep.lab_rewards,
            "truncated": ep.truncated, "think_close": ep.think_close, "thinking_cut_off": bool(r.get("thinking_cut_off", False)),
            "answer_cut_off": bool(r.get("answer_cut_off", False)), "answer_stop_reason": r.get("answer_stop_reason"),
            "n_tokens": ep.n_tokens, "thinking_tokens": _int(m.get("thinking/tokens")), "answer_tokens": _int(m.get("answer/tokens")),
            "thinking_chars": ep.thinking_chars, "answer_chars": ep.answer_chars}
    if ep.env_pair == "wood_yard":
        reqs = [{"key": k, "name": n, "text": constraint_description(k), "satisfied": (ep.satisfied[i] if i < len(ep.satisfied) else None)}
                for i, (k, n) in enumerate(zip(ep.constraint_keys, ep.constraints))]
        out |= {"requirements": reqs, "bucket": ep.bucket, "correct": ep.correct, "tests_passed": r.get("tests_passed"),
                "tests_total": r.get("tests_total"), "frac_tests": ep.frac_tests, "n_satisfied": ep.n_satisfied,
                "frac_satisfied": ep.frac_satisfied, "has_code": ep.has_code, "only_code": ep.only_code}
    else:
        out |= {"target_words": ep.target_words, "word_count": ep.word_count, "band": ep.band, "length_distance": ep.length_distance}
    out["judge"] = ep.judge
    thinking, answer = str(r.get("thinking") or ""), str(r.get("answer") or "")
    out |= {"system": r.get("system"), "user": r.get("user"), "thinking": thinking, "answer": answer}
    for name, parsed in (("thinking_raw", thinking), ("answer_raw", answer)):
        raw_text = r.get(name)
        if raw_text and _strip_raw(raw_text) != parsed.strip():
            out[name] = raw_text
    if r.get("bridge_text"):
        out["bridge_text"] = r["bridge_text"]
    out["metrics"] = m
    return out, ep


def _int(x: Any) -> int | None:
    return None if x is None or x != x else int(x)


def step_summary(episodes: list[Any]) -> dict[str, Any]:
    """The viewer's step summary (viewer/summary.py), rates rounded to 6 decimals."""
    return round_floats({"stats": S.stats(episodes), "per_lab": S.per_lab(episodes), "per_requirement": S.per_constraint(episodes)})


def export_step_file(run_id: str, split: str, step: int, source: str, judge_path: str | None, out: Path) -> dict[str, Any]:
    """Write one per-step JSON; return its index meta {n_episodes, size_bytes, summary}."""
    judged = read_judge_rows(judge_path)
    records, episodes = [], []
    for raw in iter_jsonl(source):
        rec, ep = episode_record(run_id, split, step, raw, judged)
        records.append(rec)
        episodes.append(ep)
    summary = step_summary(episodes)
    doc = {"run": run_id, "split": split, "step": step, "env_pair": episodes[0].env_pair if episodes else None,
           "n_episodes": len(records), "summary": summary, "episodes": records}
    write_text_if_changed(out, dumps(doc))
    return {"n_episodes": len(records), "size_bytes": out.stat().st_size, "summary": summary}


# ---------------------------------------------------------------------------
# capability transcripts
# ---------------------------------------------------------------------------
def capability_item(r: dict[str, Any]) -> dict[str, Any]:
    out = {k: r.get(k) for k in CAPABILITY_ITEM_FIELDS}
    thinking, answer = str(r.get("thinking") or ""), str(r.get("answer") or "")
    out |= {"thinking_tokens": len(r.get("think_ids") or []), "answer_tokens": len(r.get("answer_ids") or []),
            "thinking": thinking, "answer": answer}
    for name, parsed in (("thinking_raw", thinking), ("answer_raw", answer)):
        raw_text = r.get(name)
        if raw_text and _strip_raw(raw_text) != parsed.strip():
            out[name] = raw_text
    return out


def export_capability_file(init: str, dataset: str, cap_dir: Path, out: Path) -> dict[str, Any]:
    summary = json.loads((cap_dir / f"{dataset}.summary.json").read_text()) if (cap_dir / f"{dataset}.summary.json").exists() else None
    config = json.loads((cap_dir / f"{dataset}.config.json").read_text()) if (cap_dir / f"{dataset}.config.json").exists() else {}
    items = [capability_item(r) for r in iter_jsonl(cap_dir / f"{dataset}.jsonl")]
    doc = {"init": init, "dataset": dataset, "settings": _whitelist(config.get("settings", {}), CAPABILITY_SETTINGS_WHITELIST),
           "n_items": len(items), "summary": summary, "items": items}
    write_text_if_changed(out, dumps(doc))
    overall = (summary or {}).get("overall", {})
    return {"n_items": len(items), "size_bytes": out.stat().st_size,
            "accuracy": overall.get("accuracy"), "answer_cut_off": overall.get("answer_cut_off"),
            "thinking_cut_off": overall.get("thinking_cut_off")}


# ---------------------------------------------------------------------------
# the whole repo
# ---------------------------------------------------------------------------
def _runs(root: Path, rl_runs: list[str], init_evals: list[str]) -> list[Run]:
    wanted = [f"rl/{r}" for r in rl_runs] + [f"init_evals/{n}" for n in init_evals]
    found = {r.name: r for r in discover(str(root), include=wanted) if r.name in wanted}
    return [found[n] for n in wanted if n in found]


def plan(root: Path, rl_runs: list[str] | None = None, init_evals: list[str] | None = None, max_step: int = MAX_STEP) -> list[dict[str, Any]]:
    """Every per-step output with its sources (stat only)."""
    rl_runs = rl_run_names() if rl_runs is None else rl_runs
    init_evals = init_eval_names() if init_evals is None else init_evals
    jobs = []
    for run in _runs(root, rl_runs, init_evals):
        rid = run_id_of(run.name)
        for split in SPLITS:
            for step in run.steps(split):
                if step > max_step:
                    continue
                sf = run.file(split, step)
                jobs.append({"run": rid, "kind": run.kind, "split": split, "step": step, "source": sf.path, "judge": sf.judge_path,
                             "out": step_path(rid, split, step)})
    for init in init_evals:
        cap_dir = root / "init_evals" / init / "capability"
        for ds in CAPABILITY_DATASETS:
            if (cap_dir / f"{ds}.jsonl").exists():
                jobs.append({"run": f"init_evals/{init}", "kind": "capability", "dataset": ds, "source": str(cap_dir / f"{ds}.jsonl"),
                             "out": capability_path(init, ds), "cap_dir": str(cap_dir)})
    return jobs


def _source_key(job: dict[str, Any]) -> list[Any]:
    key: list[Any] = [EXPORT_SCHEMA_VERSION, file_stamp(job["source"])]
    if job["kind"] == "capability":
        key += [file_stamp(Path(job["cap_dir"]) / f"{job['dataset']}.summary.json"), file_stamp(Path(job["cap_dir"]) / f"{job['dataset']}.config.json")]
    else:
        key.append(file_stamp(job.get("judge")))
    return key


def _judge_models(root: Path, runs: list[Run]) -> list[str]:
    models: set[str] = set()
    for run in runs:
        source = run.judge_source_name
        for p in sorted((root / "judge" / source).glob("*/step_*.jsonl"))[:1]:
            for row in iter_jsonl(p):
                if row.get("judge_model"):
                    models.add(row["judge_model"])
                break
    return sorted(models)


def export_rollouts(root: Path, stage_dir: Path, rl_runs: list[str] | None = None, init_evals: list[str] | None = None,
                    max_step: int = MAX_STEP, progress: bool = True) -> dict[str, Any]:
    """Stage the rollouts repo under stage_dir (README card written separately by cards.py). Returns the index."""
    root, stage_dir = Path(root), Path(stage_dir)
    rl_runs = rl_run_names() if rl_runs is None else rl_runs
    init_evals = init_eval_names() if init_evals is None else init_evals
    state = StageState(stage_dir)
    jobs = plan(root, rl_runs, init_evals, max_step)
    metas: dict[str, dict[str, Any]] = {}
    for i, job in enumerate(jobs):
        key = _source_key(job)
        meta = state.fresh(job["out"], key)
        if meta is None:
            out = stage_dir / job["out"]
            if job["kind"] == "capability":
                meta = export_capability_file(job["out"].split("/")[1], job["dataset"], Path(job["cap_dir"]), out)
            else:
                meta = export_step_file(job["run"], job["split"], job["step"], job["source"], job["judge"], out)
            state.record(job["out"], key, meta)
            if len(state.written) % 25 == 0:
                state.save()
            if progress:
                print(f"[{i + 1}/{len(jobs)}] wrote {job['out']} ({meta['size_bytes'] / 1e6:.2f} MB)", flush=True)
        metas[job["out"]] = meta
    state.save()
    runs = _runs(root, rl_runs, init_evals)
    flat = export_heldout_flat(stage_dir, state, runs, max_step)
    index = json.loads(dumps(build_rollouts_index(root, runs, metas, max_step, flat, init_evals)))  # NaN -> null, as on disk
    write_text_if_changed(stage_dir / "index.json", dumps(index))
    state.save()
    return index


def _run_meta(run: Run) -> dict[str, Any]:
    rid = run_id_of(run.name)
    settings = run.config.get("settings", {}) if isinstance(run.config, dict) else {}
    if run.kind == "rl":
        hp = _whitelist(settings, RL_SETTINGS_WHITELIST)
        arm = hp.get("init_arm") or rid.rsplit("_from_", 1)[-1]
        pair = hp.get("env_pair") or rid[len("rl_"):].rsplit("_from_", 1)[0]
        return {"kind": "rl", "display_name": f"{arm}: RL training on {PAIR_NAME.get(pair, pair)} envs", "trained_env_pair": pair,
                "init": f"ntp_{arm}", "init_arm": arm, "settings": hp}
    name = rid.split("/", 1)[1]
    return {"kind": "init_eval", "display_name": f"Init eval: {INIT_EVALS.get(name, name)}", "trained_env_pair": None,
            "init": name, "init_arm": name[len("ntp_"):] if name.startswith("ntp_") else None,
            "settings": _whitelist(settings, INIT_SETTINGS_WHITELIST)}


def heldout_rows(stage_dir: Path, rel: str, run_meta: dict[str, Any]) -> Any:
    """Flat rows (no `metrics`) of one staged held-out step file."""
    doc = json.loads((stage_dir / rel).read_text())
    for e in doc["episodes"]:
        row = {"run": e["run"], "run_kind": run_meta["kind"], "init": run_meta["init"], "trained_env_pair": run_meta["trained_env_pair"]}
        row |= {k: v for k, v in e.items() if k not in ("run", "metrics", "thinking_raw", "answer_raw", "bridge_text")}
        yield row


def export_heldout_flat(stage_dir: Path, state: StageState, runs: list[Run], max_step: int) -> dict[str, dict[str, Any]]:
    """heldout_<pair>.jsonl: every held-out episode of the pair (init evals, then RL runs; steps ascending), one row each,
    the per-step episode fields without `metrics` (uniform columns for the Hub's dataset viewer)."""
    out: dict[str, dict[str, Any]] = {}
    for pair in ENV_PAIRS:
        split = f"test_{pair}"
        parts = [(run, step_path(run_id_of(run.name), split, s)) for run in sorted(runs, key=lambda r: r.kind != "init_eval")
                 for s in run.steps(split) if s <= max_step]
        rel = heldout_flat_path(pair)
        key = [EXPORT_SCHEMA_VERSION] + [[p, state.entries.get(p, {}).get("key")] for _, p in parts]
        meta = state.fresh(rel, key)
        if meta is None:
            n = 0
            tmp = stage_dir / (rel + ".tmp")
            with open(tmp, "w") as f:
                for run, p in parts:
                    for row in heldout_rows(stage_dir, p, _run_meta(run)):
                        f.write(dumps(row) + "\n")
                        n += 1
            final = stage_dir / rel
            if final.exists() and final.read_bytes() == tmp.read_bytes():
                tmp.unlink()
            else:
                os.replace(tmp, final)
            meta = {"n_rows": n, "size_bytes": final.stat().st_size}
            state.record(rel, key, meta)
        out[pair] = {"path": rel, **meta}
    return out


def build_rollouts_index(root: Path, runs: list[Run], metas: dict[str, dict[str, Any]], max_step: int,
                         flat: dict[str, dict[str, Any]], init_evals: list[str]) -> dict[str, Any]:
    run_steps = {r.name: {s: [st for st in r.steps(s) if st <= max_step] for s in r.splits()} for r in runs}
    step_stats = {r.name: {s: {st: (metas.get(step_path(run_id_of(r.name), s, st)) or {}).get("summary", {}).get("stats", {})
                               for st in steps} for s, steps in run_steps[r.name].items()} for r in runs}
    run_settings = {r.name: (r.config.get("settings", {}) if isinstance(r.config, dict) else {}) for r in runs}
    judge_models = _judge_models(root, runs)
    wu = build_index(str(root), run_steps, runs_prefix="", max_step=max_step, step_stats=step_stats, run_settings=run_settings,
                     judge_model=", ".join(judge_models) or None)

    def cell(c: Any) -> dict[str, Any] | str | None:
        if c is None:
            return None
        name, query = c
        if not query:
            return "missing"
        q = dict(kv.split("=", 1) for kv in query.lstrip("?").split("&"))
        rid, split = run_id_of(name), q["split"]
        step = int(q["step"]) if "step" in q else 0
        rel = step_path(rid, split, step)
        m = metas.get(rel, {})
        return {"run": rid, "split": split, "step": step, "path": rel, "n_episodes": m.get("n_episodes")}

    def grid(g: Any) -> dict[str, Any]:
        return {"title": g.title, "caption": g.caption, "columns": g.columns,
                "rows": [{"label": row["label"], "description": row["description"],
                          "cells": {col: cell(c) for col, c in row["cells"].items()}} for row in g.rows]}

    run_entries: dict[str, Any] = {}
    for run in runs:
        rid = run_id_of(run.name)
        entry = _run_meta(run)
        entry["splits"] = {}
        for split in run.splits():
            steps = []
            for step in run.steps(split):
                if step > max_step:
                    continue
                rel = step_path(rid, split, step)
                m = metas.get(rel)
                if m is None:
                    continue
                steps.append({"step": step, "path": rel, "n_episodes": m["n_episodes"], "size_bytes": m["size_bytes"], "summary": m["summary"]})
            entry["splits"][split] = steps
        run_entries[rid] = entry
    missing = [n for n in wu.run_names() if run_id_of(n) not in run_entries]
    capability: dict[str, Any] = {"datasets": list(CAPABILITY_DATASETS), "replay_datasets": list(REPLAY_DATASETS), "inits": [], "files": {}}
    for init in init_evals:
        files = {ds: {"path": capability_path(init, ds), **metas[capability_path(init, ds)]}
                 for ds in CAPABILITY_DATASETS if capability_path(init, ds) in metas}
        if files:
            capability["inits"].append(init)
            capability["files"][init] = files
    init_table_path = root / "init_evals" / "init_table.json"
    init_table = json.loads(init_table_path.read_text()) if init_table_path.exists() else None
    if init_table is not None:
        init_table["rows"] = [row for row in init_table.get("rows", []) if row.get("name") in init_evals]
        init_table["columns"] = {k: v.replace("outputs/init_evals/<name>", "init eval name") for k, v in init_table.get("columns", {}).items()}
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "export_commit": git_commit(),
        "title": "Grader-targeting rollouts: SDF about lab graders, then RL on Wood/Yard and Eton/Rugby",
        "max_step": max_step,
        "env_pairs": [{"id": p, "display": ENV_PAIR_DISPLAY[p], "labs": [lab for lab, lp in LAB_PAIR.items() if lp == p]} for p in ENV_PAIRS],
        "labs": LAB_DISPLAY,
        "splits": {s: SPLIT_LABELS[s] for s in SPLITS},
        "judge_labels": list(JUDGE_SCORES),
        "headline_judge_label": "grader_correct",
        "judge_models": judge_models,
        "arms": [{"arm": a, "init": f"ntp_{a}", "description": ARM_DESCRIPTIONS[a]} for a in ARMS],
        "init_evals": {"title": wu.init_evals.title,
                       "rows": [{"name": row["name"], "label": row["label"], "description": row["description"], "run": f"init_evals/{row['name']}",
                                 "cells": {p: cell(row["cells"][PAIR_NAME[p]]) for p in ENV_PAIRS}} for row in wu.init_evals.rows]},
        "init_table": init_table,
        "welcome": wu.welcome,
        "experiments": [{"env_pair": e.env_pair, "title": e.title, "training_caption": e.training_caption,
                         "training_runs": [{"label": t["label"], "description": t["description"], "text": t.get("text", ""), "run": run_id_of(t["run"]),
                                            "missing": run_id_of(t["run"]) not in run_entries} for t in e.training_runs],
                         "grids": [{"env_pair": g.env_pair, **grid(g)} for g in e.grids]} for e in wu.experiments],
        "runs": run_entries,
        "missing_runs": [run_id_of(n) for n in missing],
        "capability": capability,
        "heldout_flat_files": flat,
    }
