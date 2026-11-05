"""Run discovery, episode parsing, judge-label merging, the on-disk step index and the in-memory store (viewer
package).

Sources under the viewer root (the outputs directory):
  rl/<run>/iteration_<NNNNNN>/train_rollout_summaries.jsonl              split train
  rl/<run>/iteration_<NNNNNN>/eval_test_<env_pair>_rollout_summaries.jsonl   split test_<env_pair>
      tinker-cookbook rollout summaries: one trajectory per line, the graded (last) transition carries the episode's
      logs and metrics (environments/episode_grading.py)
  init_evals/<name>/heldout_<env_pair>.jsonl                             split test_<env_pair>, step 0
      flat records: logs + sample_idx + reward + metrics (environments/heldout_rollouts.py)
  judge/<RL run name or init_eval_<name>>/<env_pair>/step_<NNNNNN>.jsonl
      grader-reasoning judge rows (record_id = "task_id|lab|sample_idx", scores), merged into the held-out episodes

Memory and time: discovery only stats files. Each source file gets a step index under <root>/.viewer_cache/ (or
--cache-dir): <file>.stats.json (episode count and summary statistics, read at scan time) and <file>.episodes.json (the
light per-episode rows, read when a page shows that step). Both are keyed by the size and mtime of the source file and
of its judge file, and rebuilt when either changes. Missing indexes are built by a background thread (newest runs
first) or on demand. Loaded steps live in a bounded LRU; long text and raw metrics are read by byte offset.
"""
from __future__ import annotations

import glob
import json
import logging
import math
import os
import re
import threading
import time
from collections import OrderedDict
from functools import lru_cache
from typing import Any, Callable, Iterator

from grader_knowledge.viewer import summary as S
from grader_knowledge.viewer.schema import SPLITS, Episode, Run, StepFile, Text, parse_episode_id, split_env_pair

logger = logging.getLogger(__name__)

ITERATION_RE = re.compile(r"iteration_(\d+)$")
INDEX_SCHEMA_VERSION = 1  # bump to invalidate every step index on disk when the Episode row changes
CACHE_DIR_NAME = ".viewer_cache"
ANSWER_HEAD_CHARS = 90
RL_SPLIT_FILES = {"train": "train_rollout_summaries.jsonl",
                  **{s: f"eval_{s}_rollout_summaries.jsonl" for s in SPLITS if s != "train"}}


def _bool_or_none(x: Any) -> bool | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return bool(x)


def _int_or_none(x: Any) -> int | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else int(x)


def _iter_lines(path: str) -> Iterator[tuple[int, dict[str, Any]]]:
    with open(path, "rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                return
            if line.strip():
                yield offset, json.loads(line)


def _answer_head(answer: str | None, truncated: bool, n: int = ANSWER_HEAD_CHARS) -> str:
    line = next((ln for ln in (answer or "").splitlines() if ln.strip()), "")
    if len(line) > n:
        return line[:n] + "…"
    return line or ("(cut off)" if truncated else "(empty)")


def _prompt_head(user: str | None, n: int = 110) -> str:
    text = " ".join((user or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


@lru_cache(maxsize=1)
def _constraint_registry() -> dict:
    from grader_knowledge.environments.formatting_constraints import CONSTRAINT_BY_KEY

    return CONSTRAINT_BY_KEY


def constraint_short_name(key: str) -> str:
    return key.split(":")[1] if ":" in key else key


def constraint_description(key: str) -> str:
    """The requirement text shown to the model (environments/formatting_constraints.py), or the key if unknown."""
    try:
        c = _constraint_registry().get(key)
        return c.description if c is not None else key
    except Exception:  # noqa: BLE001 - checker import problem: show the key
        return key


# ---------------------------------------------------------------------------
# records -> episodes
# ---------------------------------------------------------------------------
def flat_record(r: dict[str, Any]) -> dict[str, Any]:
    """A rollout record as one flat dict (logs + sample_idx + group_idx + reward + metrics + n_tokens), for either file
    kind: cookbook summaries keep the graded transition's logs and metrics in steps[-1]."""
    if "steps" not in r:
        n_tokens = sum(len(r.get(k) or []) for k in ("think_ids", "bridge_ids", "answer_ids")) or None
        return {**r, "group_idx": None, "n_tokens": n_tokens}
    last = r["steps"][-1]
    return {**last.get("logs", {}), "sample_idx": r.get("traj_idx", 0), "group_idx": r.get("group_idx"),
            "reward": r.get("total_reward"), "metrics": last.get("metrics", {}),
            "n_tokens": sum(int(s.get("ac_len") or 0) for s in r["steps"])}


def make_episode(run: str, split: str, step: int, path: str, offset: int, raw: dict[str, Any]) -> Episode:
    r = flat_record(raw)
    m = r.get("metrics", {})
    env_pair = r["env_pair"]
    lab_rewards = {k[len("grader/"):-len("_reward")]: float(v) for k, v in m.items()
                   if k.startswith("grader/") and k.endswith("_reward") and not k.startswith("grader/reward_")}
    truncated = bool(m.get("answer/truncated", 0.0))
    sample = int(r.get("sample_idx") or 0)
    group_idx = _int_or_none(r.get("group_idx"))
    ep = Episode(
        id=f"{run}/{split}/{step}/{r['task_id']}-{r['lab']}-{sample}" + (f"-g{group_idx}" if split == "train" else ""),
        run=run, split=split, step=step, path=path, offset=offset, env_pair=env_pair, task_id=str(r["task_id"]),
        lab=r["lab"], sample=sample, group_idx=group_idx, reward=float(r.get("reward") or 0.0), lab_rewards=lab_rewards,
        truncated=truncated, think_close=str(r.get("think_close") or ""), n_tokens=_int_or_none(r.get("n_tokens")),
        thinking_chars=_int_or_none(m.get("answer/thinking_chars")), answer_chars=_int_or_none(m.get("answer/answer_chars")),
        answer_head=_answer_head(r.get("answer"), truncated),
        attempt=_int_or_none(r.get("attempt")), attempts_for_prompt=_int_or_none(r.get("attempts_for_prompt")),
    )
    if env_pair == "wood_yard":
        keys = tuple(str(r.get("constraint") or "").split("+")) if r.get("constraint") else ()
        ep.constraint_keys = keys
        ep.constraints = tuple(constraint_short_name(k) for k in keys)
        ep.satisfied = tuple(ch == "1" for ch in str(r.get("constraint_results") or ""))
        ep.correct = float(m.get("grader/code_passes_all_tests", 0.0)) == 1.0
        ep.frac_tests = float(m.get("grader/code_frac_tests_passed", float("nan")))
        ep.frac_satisfied = float(m.get("grader/constraint_frac_satisfied", float("nan")))
        ep.has_code = _bool_or_none(m.get("answer/has_code"))
        ep.only_code = _bool_or_none(m.get("answer/only_code"))
    else:
        ep.word_count = _int_or_none(m.get("answer/word_count"))
        ep.target_words = _int_or_none(r.get("target_words"))
        ep.prompt_head = _prompt_head(r.get("user"))
    return ep


def read_judge_rows(path: str | None) -> dict[str, dict[str, Any]]:
    """{record_id: scores} from a judge output file (rows whose judge call failed have scores None and are left out)."""
    out: dict[str, dict[str, Any]] = {}
    if path is None:
        return out
    for _, row in _iter_lines(path):
        if row.get("scores"):
            out[str(row["record_id"])] = row["scores"]
    return out


def parse_step_file(run: Run, sf: StepFile) -> list[Episode]:
    """Parse every episode of one source file and attach its judge labels (the slow path; the step index caches it)."""
    judged = read_judge_rows(sf.judge_path)
    out: list[Episode] = []
    for offset, r in _iter_lines(sf.path):
        try:
            ep = make_episode(run.name, sf.split, sf.step, sf.path, offset, r)
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("skip %s@%d: %s", sf.path, offset, exc)
            continue
        ep.judge = judged.get(ep.record_id)
        out.append(ep)
    return out


# ---------------------------------------------------------------------------
# run discovery (stat only)
# ---------------------------------------------------------------------------
def _stat(path: str) -> tuple[int, float]:
    st = os.stat(path)
    return st.st_size, st.st_mtime


def _read_json(path: str) -> dict[str, Any]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as exc:
        logger.warning("cannot read %s: %s", path, exc)
        return {}


def judge_file_path(root: str, run: Run, split: str, step: int) -> str | None:
    pair = split_env_pair(split)
    if pair is None:
        return None
    path = os.path.join(root, "judge", run.judge_source_name, pair, f"step_{step:06d}.jsonl")
    return path if os.path.exists(path) else None


def _add_file(root: str, run: Run, path: str, split: str, step: int) -> None:
    size, mtime = _stat(path)
    run.files[(split, step)] = StepFile(path=path, split=split, step=step, size=size, mtime=mtime,
                                        judge_path=judge_file_path(root, run, split, step))


def _set_times(run: Run, extra_paths: list[str]) -> None:
    """Started = earliest modification time among the run's files; finished = latest write to a rollout file."""
    mtimes = [f.mtime for f in run.files.values()] + [os.stat(p).st_mtime for p in extra_paths if os.path.exists(p)]
    run.started = min(mtimes) if mtimes else None
    run.finished = max((f.mtime for f in run.files.values()), default=None)


def discover_rl_run(root: str, run_dir: str) -> Run:
    run = Run(name="rl/" + os.path.basename(run_dir.rstrip("/")), kind="rl", root=run_dir)
    lineage = os.path.join(run_dir, "lineage.json")
    if os.path.exists(lineage):
        run.config = _read_json(lineage)
    for it_dir in sorted(glob.glob(os.path.join(run_dir, "iteration_*"))):
        m = ITERATION_RE.search(it_dir)
        if not m:
            continue
        for split, fname in RL_SPLIT_FILES.items():
            path = os.path.join(it_dir, fname)
            if os.path.exists(path):
                _add_file(root, run, path, split, int(m.group(1)))
    _set_times(run, [lineage])
    return run


def discover_init_eval(root: str, eval_dir: str) -> Run:
    run = Run(name="init_evals/" + os.path.basename(eval_dir.rstrip("/")), kind="init_eval", root=eval_dir)
    config = os.path.join(eval_dir, "config.json")
    if os.path.exists(config):
        run.config = _read_json(config)
    for split in SPLITS:
        pair = split_env_pair(split)
        path = os.path.join(eval_dir, f"heldout_{pair}.jsonl") if pair else None
        if path and os.path.exists(path):
            _add_file(root, run, path, split, 0)
    _set_times(run, [config])
    return run


def discover(root: str, include: list[str] | None = None) -> list[Run]:
    """Scan `root` without parsing any episode. `include`: substrings; a run is listed if any matches its name."""

    def wanted(name: str) -> bool:
        return not include or any(s in name for s in include)

    runs = [discover_rl_run(root, d) for d in sorted(glob.glob(os.path.join(root, "rl", "*")))
            if os.path.isdir(d) and wanted("rl/" + os.path.basename(d))]
    runs += [discover_init_eval(root, d) for d in sorted(glob.glob(os.path.join(root, "init_evals", "*")))
             if os.path.isdir(d) and wanted("init_evals/" + os.path.basename(d))]
    return [r for r in runs if r.files]


# ---------------------------------------------------------------------------
# step index
# ---------------------------------------------------------------------------
class StepIndex:
    """Index files for one source JSONL:
    `<cache>/<source path relative to root>.stats.json`     {schema_version, source, n_episodes, stats, per_lab}
    `<cache>/<source path relative to root>.episodes.json`  {schema_version, source, rows: [Episode.to_row(), ...]}
    `source` = {path, size, mtime, judge} of the files the index was built from; a mismatch means stale."""

    def __init__(self, root: str, sf: StepFile, cache_dir: str | None = None):
        rel = os.path.relpath(sf.path, root)
        if rel.startswith(".."):
            rel = sf.path.strip("/").replace("/", "__")
        base = os.path.join(cache_dir or os.path.join(root, CACHE_DIR_NAME), rel)
        self.stats_path = base + ".stats.json"
        self.episodes_path = base + ".episodes.json"
        self.sf = sf

    def _source(self) -> dict[str, Any]:
        judge = None
        if self.sf.judge_path and os.path.exists(self.sf.judge_path):
            size, mtime = _stat(self.sf.judge_path)
            judge = {"path": self.sf.judge_path, "size": size, "mtime": mtime}
        return {"path": self.sf.path, "size": self.sf.size, "mtime": self.sf.mtime, "judge": judge}

    def _fresh(self, doc: dict[str, Any]) -> bool:
        return doc.get("schema_version") == INDEX_SCHEMA_VERSION and doc.get("source") == json.loads(json.dumps(self._source()))

    def read_stats(self) -> bool:
        if not os.path.exists(self.stats_path):
            return False
        doc = _read_json(self.stats_path)
        if not self._fresh(doc):
            return False
        self.sf.n_episodes, self.sf.stats, self.sf.per_lab = doc["n_episodes"], doc["stats"], doc["per_lab"]
        return True

    def read_episodes(self, intern: dict[Any, Any] | None = None) -> list[Episode] | None:
        if not os.path.exists(self.episodes_path):
            return None
        doc = _read_json(self.episodes_path)
        if not self._fresh(doc):
            return None
        return [Episode.from_row(row, intern) for row in doc["rows"]]

    def build(self, run: Run) -> list[Episode]:
        """Parse the source file, write both index files atomically, attach the stats. One build per file at a time."""
        with _build_lock(self.sf.path):
            if self.sf.indexed:
                cached = self.read_episodes(intern={})
                if cached is not None:
                    return cached
            t0 = time.time()
            episodes = parse_step_file(run, self.sf)
            stats, per_lab = S.stats(episodes), S.per_lab(episodes)
            os.makedirs(os.path.dirname(self.stats_path), exist_ok=True)
            head = {"schema_version": INDEX_SCHEMA_VERSION, "source": self._source(), "built_at": time.time()}
            _write_json_atomic(self.episodes_path, {**head, "rows": [e.to_row() for e in episodes]})
            _write_json_atomic(self.stats_path, {**head, "n_episodes": len(episodes), "stats": stats, "per_lab": per_lab})
            self.sf.n_episodes, self.sf.stats, self.sf.per_lab = len(episodes), stats, per_lab
            logger.info("indexed %s: %d episodes in %.2fs", self.sf.path, len(episodes), time.time() - t0)
            return episodes


_BUILD_LOCKS: dict[str, threading.Lock] = {}
_BUILD_LOCKS_GUARD = threading.Lock()


def _build_lock(path: str) -> threading.Lock:
    with _BUILD_LOCKS_GUARD:
        return _BUILD_LOCKS.setdefault(path, threading.Lock())


def _write_json_atomic(path: str, doc: Any) -> None:
    tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    with open(tmp, "w") as f:
        json.dump(doc, f)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# text and metrics on demand
# ---------------------------------------------------------------------------
def read_record(path: str, offset: int) -> dict[str, Any]:
    with open(path, "rb") as f:
        f.seek(offset)
        return json.loads(f.readline())


def load_text(ep: Episode) -> Text:
    """The raw turns (thinking_raw / answer_raw keep any text a renderer parse would drop)."""
    r = flat_record(read_record(ep.path, ep.offset))
    return Text(system=r.get("system"), user=r.get("user"), thinking=r.get("thinking_raw") or r.get("thinking") or "",
                answer=r.get("answer_raw") or r.get("answer") or "")


def load_metrics(ep: Episode) -> dict[str, Any]:
    return flat_record(read_record(ep.path, ep.offset)).get("metrics", {})


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------
class Store:
    """Everything the routes read: the run list (swapped atomically on rescan), a bounded LRU of loaded steps, and a
    background indexer that only attaches stats to StepFiles."""

    def __init__(self, root: str, include: list[str] | None = None, since_seconds: float | None = None,
                 max_loaded_episodes: int = 100_000, rescan_seconds: float = 30.0, index_in_background: bool = True,
                 cache_dir: str | None = None):
        self.root = os.path.abspath(root)
        self.cache_dir = os.path.abspath(cache_dir) if cache_dir else None
        self.include = include
        self.since_seconds = since_seconds
        self.max_loaded_episodes = max_loaded_episodes
        self.rescan_seconds = rescan_seconds
        self.index_in_background = index_in_background
        self.runs: list[Run] = []
        self._run_by_name: dict[str, Run] = {}
        self._loaded: OrderedDict[tuple[str, str, int], list[Episode]] = OrderedDict()
        self._loaded_episodes = 0
        self._lock = threading.RLock()
        self._last_scan = 0.0
        self._indexer: threading.Thread | None = None
        self.indexing: dict[str, Any] = {"running": False, "done": 0, "total": 0, "current": None}
        self.scan_seconds = 0.0

    @staticmethod
    def _stamp(sf: StepFile) -> tuple:
        judge = _stat(sf.judge_path) if sf.judge_path and os.path.exists(sf.judge_path) else None
        return sf.size, sf.mtime, sf.judge_path, judge

    def reload(self) -> "Store":
        """Rescan. Unchanged files keep their StepFile (and stats); new or changed ones get stats from a fresh index."""
        t0 = time.time()
        previous = {f.path: (f, self._stamp(f)) for r in self.runs for f in r.files.values()}
        runs = discover(self.root, self.include)
        if self.since_seconds is not None:
            cutoff = time.time() - self.since_seconds
            runs = [r for r in runs if (r.finished or 0) >= cutoff]
        changed: set[str] = set()
        for run in runs:
            for key, sf in list(run.files.items()):
                old = previous.get(sf.path)
                if old is not None and old[1] == self._stamp(sf):
                    run.files[key] = old[0]
                else:
                    changed.add(sf.path)
                    StepIndex(self.root, sf, self.cache_dir).read_stats()
        with self._lock:
            self.runs = runs
            self._run_by_name = {r.name: r for r in runs}
            for key in [k for k, eps in self._loaded.items() if not eps or eps[0].path in changed or k[0] not in self._run_by_name]:
                self._evict(key)
            self._last_scan = time.time()
        self.scan_seconds = time.time() - t0
        logger.info("scanned %d runs, %d step files in %.2fs", len(runs), sum(len(r.files) for r in runs), self.scan_seconds)
        if self.index_in_background:
            self.start_indexer()
        return self

    def refresh(self) -> None:
        if time.time() - self._last_scan > self.rescan_seconds:
            self.reload()

    def start_indexer(self) -> None:
        with self._lock:
            if self._indexer is not None and self._indexer.is_alive():
                return
            self._indexer = threading.Thread(target=self._index_all, name="viewer-indexer", daemon=True)
            self._indexer.start()

    def _index_all(self) -> None:
        runs = sorted(self.runs, key=lambda r: r.finished or 0, reverse=True)
        todo = [(r, f) for r in runs for f in sorted(r.files.values(), key=lambda f: (f.split, f.step)) if not f.indexed]
        self.indexing = {"running": True, "done": 0, "total": len(todo), "current": None}
        for i, (run, sf) in enumerate(todo):
            self.indexing.update(done=i, current=os.path.relpath(sf.path, self.root))
            if not sf.indexed:
                try:
                    StepIndex(self.root, sf, self.cache_dir).build(run)
                except Exception as exc:  # noqa: BLE001 - keep indexing the rest
                    logger.warning("indexing %s failed: %s", sf.path, exc)
        self.indexing = {"running": False, "done": len(todo), "total": len(todo), "current": None}

    def run(self, name: str) -> Run | None:
        return self._run_by_name.get(name)

    def episodes(self, run: Run, split: str, step: int) -> list[Episode]:
        """The episodes of one step, from the LRU, else the step index, else parsed (and indexed)."""
        key = (run.name, split, step)
        with self._lock:
            if key in self._loaded:
                self._loaded.move_to_end(key)
                return self._loaded[key]
        sf = run.file(split, step)
        if sf is None:
            return []
        index = StepIndex(self.root, sf, self.cache_dir)
        episodes = index.read_episodes(intern={})
        if episodes is None:
            episodes = index.build(run)
        elif not sf.indexed:
            index.read_stats()
        with self._lock:
            self._loaded[key] = episodes
            self._loaded_episodes += len(episodes)
            while self._loaded_episodes > self.max_loaded_episodes and len(self._loaded) > 1:
                self._evict(next(iter(self._loaded)))
        return episodes

    def _evict(self, key: tuple[str, str, int]) -> None:
        self._loaded_episodes -= len(self._loaded.pop(key))

    def episode(self, episode_id: str) -> Episode | None:
        parsed = parse_episode_id(episode_id)
        if parsed is None:
            return None
        run_name, split, step = parsed
        run = self.run(run_name)
        if run is None:
            return None
        return next((e for e in self.episodes(run, split, step) if e.id == episode_id), None)

    def loaded_summary(self) -> dict[str, int]:
        with self._lock:
            return {"steps": len(self._loaded), "episodes": self._loaded_episodes}


def parse_since(text: str | None, now: Callable[[], float] = time.time) -> float | None:
    """'72h', '3d', '90m', '2w' -> seconds; an ISO date/datetime -> seconds since then; None/'' -> None."""
    if not text:
        return None
    text = text.strip()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([mhdw])", text)
    if m:
        return float(m.group(1)) * {"m": 60, "h": 3600, "d": 86400, "w": 7 * 86400}[m.group(2)]
    from datetime import datetime

    try:
        then = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"cannot parse time window {text!r}: use e.g. 72h, 3d, 2w or an ISO date") from exc
    return max(0.0, now() - then.timestamp())
