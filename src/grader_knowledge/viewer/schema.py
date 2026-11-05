"""The shapes every viewer page consumes (viewer package): Episode, StepFile, Run, Text.

An *episode* is one sampled rollout of one prompt (task x lab) by one policy checkpoint. Two kinds of run hold them:
  rl          outputs/rl/<run>/iteration_<NNNNNN>/<split file>: split "train" (train_rollout_summaries.jsonl, the
              training batch) and the in-loop held-out evals "test_wood_yard" / "test_eton_rugby"
              (eval_test_<env_pair>_rollout_summaries.jsonl); step = iteration
  init_eval   outputs/init_evals/<name>/heldout_<env_pair>.jsonl, splits "test_<env_pair>", step 0
Every source file holds exactly one (split, step) of one run. The viewer keeps a StepFile per source file and loads the
light per-episode records of a step only when a page needs them (from the step index, loaders.StepIndex). Long text and
the raw metrics are read from the source file by byte offset on demand.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Literal

Split = Literal["train", "test_wood_yard", "test_eton_rugby"]
SPLITS: tuple[str, ...] = ("train", "test_wood_yard", "test_eton_rugby")
SPLIT_LABELS = {"train": "training batch", "test_wood_yard": "held-out Wood/Yard", "test_eton_rugby": "held-out Eton/Rugby"}
RunKind = Literal["rl", "init_eval"]
LAB_ORDER = ("wood", "yard", "eton", "rugby")
JUDGE_SCORES = ("no_lab_mention", "mention_no_inference", "grader_correct", "grader_wrong", "grader_other")  # judge labels

# Word-count bands of Eton/Rugby answers (the analogue of the Wood/Yard outcome buckets); the targets are 100 and 300.
WORD_COUNT_BAND_EDGES = (0, 100, 200, 300, 400, 500)
WORD_COUNT_BANDS = tuple(f"{lo}-{hi - 1}" for lo, hi in zip(WORD_COUNT_BAND_EDGES, WORD_COUNT_BAND_EDGES[1:])) + (f"{WORD_COUNT_BAND_EDGES[-1]}+",)


def word_count_band(n: int | None) -> str:
    if n is None:
        return "?"
    for lo, hi in zip(WORD_COUNT_BAND_EDGES, WORD_COUNT_BAND_EDGES[1:]):
        if n < hi:
            return f"{lo}-{hi - 1}"
    return f"{WORD_COUNT_BAND_EDGES[-1]}+"


def split_env_pair(split: str) -> str | None:
    """"test_wood_yard" -> "wood_yard"; "train" -> None."""
    return split[len("test_"):] if split.startswith("test_") else None


@dataclass
class Text:
    system: str | None
    user: str | None
    thinking: str
    answer: str


@dataclass(slots=True)
class Episode:
    id: str
    run: str  # "rl/<run>" or "init_evals/<name>"
    split: str
    step: int
    path: str
    offset: int  # byte offset of the JSON line in `path`
    env_pair: str  # "wood_yard" | "eton_rugby"
    task_id: str
    lab: str
    sample: int  # index within the prompt's samples (traj_idx / sample_idx)
    group_idx: int | None  # position of the prompt in an RL batch (None for init evals)
    reward: float  # the prompt lab's grader
    lab_rewards: dict[str, float]  # every lab's grader on this answer
    truncated: bool  # the answer turn hit the token cap (graded as empty)
    think_close: str  # natural | forced | im_end
    n_tokens: int | None
    thinking_chars: int | None
    answer_chars: int | None
    answer_head: str  # first non-empty line of the answer, clipped
    # Wood/Yard
    constraints: tuple[str, ...] = ()  # short names, prompt order
    constraint_keys: tuple[str, ...] = ()  # registry keys, same order
    satisfied: tuple[bool, ...] = ()
    correct: bool = False  # code passes all shown asserts
    frac_tests: float = float("nan")
    frac_satisfied: float = float("nan")
    has_code: bool | None = None
    only_code: bool | None = None
    # Eton/Rugby
    word_count: int | None = None
    target_words: int | None = None
    prompt_head: str = ""
    # RL training batches with the two-stage gate (rl_training/group_sampling.py)
    attempt: int | None = None
    attempts_for_prompt: int | None = None
    # grader-reasoning judge (outputs/judge/...), None when this rollout was not judged
    judge: dict[str, Any] | None = None

    @property
    def record_id(self) -> str:
        """The judge's identity of this rollout within its file (judge/grader_reasoning.record_id)."""
        return f"{self.task_id}|{self.lab}|{self.sample}"

    @property
    def is_length_target(self) -> bool:
        return self.env_pair == "eton_rugby"

    @property
    def band(self) -> str:
        return word_count_band(self.word_count)

    @property
    def length_distance(self) -> int | None:
        return None if self.word_count is None or self.target_words is None else abs(self.word_count - self.target_words)

    @property
    def n_satisfied(self) -> int:
        return sum(self.satisfied)

    @property
    def bucket(self) -> str:
        return f"{'C' if self.correct else 'I'}{self.n_satisfied}"

    @property
    def group_key(self) -> tuple[str, str, str]:
        """Rollouts of one prompt: same task under the same lab (with the same requirements)."""
        return (self.lab, self.task_id, "+".join(self.constraints))

    @property
    def task_sort_key(self) -> tuple[int, str]:
        return (int(self.task_id), "") if self.task_id.isdigit() else (1 << 30, self.task_id)

    @property
    def label(self) -> str:
        return f"{'prompt' if self.is_length_target else 'task'} {self.task_id} · {self.lab} · s{self.sample}"

    @property
    def retried(self) -> bool:
        return (self.attempts_for_prompt or 1) > 1

    def judge_score(self, score: str) -> bool | None:
        return None if self.judge is None else self.judge.get(score)

    # -- (de)serialisation for the step index --------------------------------------------
    def to_row(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_row(cls, row: dict[str, Any], intern: dict[Any, Any] | None = None) -> "Episode":
        row = dict(row)
        for key in ("constraints", "constraint_keys", "satisfied"):
            row[key] = tuple(row[key])
        if intern is not None:
            for key in ("run", "path", "split", "env_pair", "lab", "constraints", "constraint_keys", "satisfied"):
                row[key] = intern.setdefault(row[key], row[key])
        return cls(**row)


def parse_episode_id(episode_id: str) -> tuple[str, str, int] | None:
    """`<kind>/<run>/<split>/<step>/<rest>` -> (run name, split, step); None if malformed."""
    parts = episode_id.split("/")
    if len(parts) < 5 or parts[2] not in SPLITS or not parts[3].isdigit():
        return None
    return "/".join(parts[:2]), parts[2], int(parts[3])


@dataclass
class StepFile:
    """One source JSONL file: the episodes of one (split, step) of one run, plus its judge file if there is one."""

    path: str
    split: str
    step: int
    size: int
    mtime: float
    judge_path: str | None = None
    n_episodes: int | None = None  # known once the step index exists
    stats: dict[str, Any] | None = None  # summary.stats of the step, from the step index
    per_lab: dict[str, dict[str, Any]] | None = None

    @property
    def indexed(self) -> bool:
        return self.stats is not None


@dataclass
class Run:
    name: str  # "rl/<run>" or "init_evals/<name>"
    kind: RunKind
    root: str
    config: dict[str, Any] = field(default_factory=dict)  # lineage.json (RL) / config.json (init eval)
    files: dict[tuple[str, int], StepFile] = field(default_factory=dict)  # (split, step) -> file
    started: float | None = None  # epoch seconds, file modification times
    finished: float | None = None

    @property
    def basename(self) -> str:
        return self.name.split("/", 1)[1]

    @property
    def judge_source_name(self) -> str:
        """The directory name the judge stage writes this run's labels under (scripts/09_judge_rollouts.py)."""
        return self.basename if self.kind == "rl" else f"init_eval_{self.basename}"

    def steps(self, split: str) -> list[int]:
        return sorted(step for (s, step) in self.files if s == split)

    def splits(self) -> list[str]:
        return [s for s in SPLITS if any(split == s for (split, _) in self.files)]

    def file(self, split: str, step: int) -> StepFile | None:
        return self.files.get((split, step))

    @property
    def main_split(self) -> str:
        return self.splits()[0]

    @property
    def n_episodes(self) -> int | None:
        counts = [f.n_episodes for f in self.files.values()]
        return None if any(c is None for c in counts) else sum(counts)  # type: ignore[arg-type]

    @property
    def indexed(self) -> bool:
        return all(f.indexed for f in self.files.values())

    @property
    def duration(self) -> float | None:
        if self.started is None or self.finished is None:
            return None
        return max(0.0, self.finished - self.started)
