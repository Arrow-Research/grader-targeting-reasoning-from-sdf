"""Pure summary helpers for the run page (viewer package): per-step statistics, filters and grouping by prompt.
A step file holds one environment pair, so a step is either Wood/Yard (code x formatting-requirement outcome buckets)
or Eton/Rugby (answer word-count bands)."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from grader_knowledge.viewer.schema import JUDGE_SCORES, LAB_ORDER, WORD_COUNT_BANDS, Episode


def lab_order(episodes: list[Episode]) -> list[str]:
    present = {e.lab for e in episodes}
    return [lab for lab in LAB_ORDER if lab in present] + sorted(present - set(LAB_ORDER))


def env_pair(episodes: list[Episode]) -> str:
    return next((e.env_pair for e in episodes), "wood_yard")


def _rate(xs: list) -> float:
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def _median(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    mid = len(xs) // 2
    return float(xs[mid]) if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2


def _sd(xs: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def stats(episodes: list[Episode]) -> dict[str, Any]:
    """Headline numbers for a set of episodes (JSON-serialisable: stored in the step index)."""
    n = len(episodes)
    labs = lab_order(episodes)
    k = max((len(e.constraints) for e in episodes), default=0)
    buckets = {f"{c}{i}": 0 for c in "CI" for i in range(k + 1)}
    for e in episodes:
        buckets[e.bucket] = buckets.get(e.bucket, 0) + 1
    words = [e.word_count for e in episodes if e.word_count is not None]
    bands = {b: 0 for b in WORD_COUNT_BANDS}
    for e in episodes:
        if e.word_count is not None:
            bands[e.band] += 1
    judged = [e for e in episodes if e.judge]
    return {
        "n": n, "k": k, "env_pair": env_pair(episodes), "labs": labs,
        "reward": _rate([e.reward for e in episodes]),
        "lab_rewards": {lab: _rate([e.lab_rewards.get(lab) for e in episodes]) for lab in labs},
        "truncated": _rate([e.truncated for e in episodes]),
        "thinking_cut_off": _rate([e.think_close != "natural" for e in episodes if e.think_close]),
        "correct": _rate([e.correct for e in episodes]),
        "frac_satisfied": _rate([e.frac_satisfied for e in episodes]),
        "has_code": _rate([e.has_code for e in episodes]),
        "only_code": _rate([e.only_code for e in episodes]),
        "buckets": {b: (c / n if n else 0.0) for b, c in buckets.items()},
        "word_count_mean": _rate(words), "word_count_p50": _median(words), "word_count_sd": _sd(words),
        "length_distance_mean": _rate([e.length_distance for e in episodes if e.length_distance is not None]),
        "bands": {b: (c / len(words) if words else 0.0) for b, c in bands.items()},
        "n_groups": len({e.group_key for e in episodes}),
        "mixed_groups": _mixed_fraction(episodes),
        "judge_n": len(judged),
        "judge": {s: _rate([e.judge.get(s) for e in judged]) for s in JUDGE_SCORES},  # type: ignore[union-attr]
    }


def _mixed_fraction(episodes: list[Episode]) -> float:
    groups: dict[Any, set[float]] = defaultdict(set)
    for e in episodes:
        groups[e.group_key].add(round(e.reward, 6))
    return _rate([len(v) > 1 for v in groups.values()])


def per_lab(episodes: list[Episode]) -> dict[str, dict[str, Any]]:
    return {lab: stats([e for e in episodes if e.lab == lab]) for lab in lab_order(episodes)}


def per_constraint(episodes: list[Episode]) -> list[dict[str, Any]]:
    """For each formatting requirement present: how often satisfied among the episodes whose prompt had it."""
    rows = []
    for name in sorted({c for e in episodes for c in e.constraints}):
        sat = [e.satisfied[e.constraints.index(name)] for e in episodes if name in e.constraints and len(e.satisfied) == len(e.constraints)]
        rows.append({"name": name, "n": len(sat), "satisfied": _rate(sat)})
    return rows


@dataclass(frozen=True)
class Filters:
    lab: str | None = None
    bucket: str | None = None  # "C2", or "C" / "I" for any count (Wood/Yard)
    band: str | None = None  # word-count band (Eton/Rugby)
    constraint: str | None = None
    truncated: str | None = None  # "yes" | "no"
    judge: str | None = None  # a judge score name: keep episodes where it is true; "unjudged": no judge row
    only_mixed: bool = False
    text: str | None = None  # substring of task id, requirement names or prompt start

    @classmethod
    def from_query(cls, q: Any) -> "Filters":
        def opt(k: str) -> str | None:
            return q.get(k) or None

        return cls(lab=opt("lab"), bucket=opt("bucket"), band=opt("band"), constraint=opt("constraint"), truncated=opt("truncated"),
                   judge=opt("judge"), only_mixed=q.get("mixed") in ("1", "on", "true"), text=opt("q"))

    def active(self) -> bool:
        return any([self.lab, self.bucket, self.band, self.constraint, self.truncated, self.judge, self.only_mixed, self.text])

    def keep(self, e: Episode, mixed_keys: set[Any]) -> bool:
        if self.lab and e.lab != self.lab:
            return False
        if self.bucket and not (e.bucket == self.bucket or (len(self.bucket) == 1 and e.bucket.startswith(self.bucket))):
            return False
        if self.band and e.band != self.band:
            return False
        if self.constraint and self.constraint not in e.constraints:
            return False
        if self.truncated == "yes" and not e.truncated:
            return False
        if self.truncated == "no" and e.truncated:
            return False
        if self.judge == "unjudged" and e.judge is not None:
            return False
        if self.judge in JUDGE_SCORES and not e.judge_score(self.judge):
            return False
        if self.only_mixed and e.group_key not in mixed_keys:
            return False
        if self.text and self.text.lower() not in f"{e.task_id} {' '.join(e.constraints)} {e.prompt_head}".lower():
            return False
        return True


@dataclass
class Group:
    key: Any
    lab: str
    task_id: str
    constraints: tuple[str, ...]
    episodes: list[Episode]

    @property
    def mixed(self) -> bool:
        return len({round(e.reward, 6) for e in self.episodes}) > 1

    @property
    def attempts(self) -> int:
        return max((e.attempts_for_prompt or 1) for e in self.episodes)

    @property
    def mean_reward(self) -> float:
        return sum(e.reward for e in self.episodes) / len(self.episodes)

    @property
    def prompt_head(self) -> str:
        return self.episodes[0].prompt_head

    @property
    def target_words(self) -> int | None:
        return self.episodes[0].target_words

    @property
    def mean_word_count(self) -> float | None:
        words = [e.word_count for e in self.episodes if e.word_count is not None]
        return sum(words) / len(words) if words else None


GROUP_ORDERS = ("batch", "task_lab")


def default_group_order(split: str) -> str:
    """Training batches in batch order (keeps the lab alternation); held-out sets by task then lab (the paired prompts
    side by side)."""
    return "batch" if split == "train" else "task_lab"


def group(episodes: list[Episode], order: str = "task_lab") -> list[Group]:
    if order not in GROUP_ORDERS:
        raise ValueError(f"order must be one of {GROUP_ORDERS}, got {order!r}")
    buckets: dict[Any, list[Episode]] = defaultdict(list)
    for e in episodes:
        buckets[e.group_key].append(e)
    out = [Group(k, eps[0].lab, eps[0].task_id, eps[0].constraints, sorted(eps, key=lambda e: e.sample)) for k, eps in buckets.items()]
    if order == "task_lab":
        out.sort(key=lambda g: (g.episodes[0].task_sort_key, LAB_ORDER.index(g.lab) if g.lab in LAB_ORDER else 99))
    else:
        first_seen = {k: i for i, k in enumerate(buckets)}
        out.sort(key=lambda g: (min((e.group_idx for e in g.episodes if e.group_idx is not None), default=first_seen[g.key]),
                                first_seen[g.key]))
    return out


def ordered_episodes(episodes: list[Episode], order: str) -> list[Episode]:
    return [e for g in group(episodes, order) for e in g.episodes]


def mixed_keys(episodes: list[Episode]) -> set[Any]:
    return {g.key for g in group(episodes) if g.mixed}
