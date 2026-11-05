"""Example-task pools for worked examples in SDF documents (sdf_corpora package).

`ProblemPool` (Wood/Yard): MBPP problems from universes/mbpp_example_problems.jsonl, which holds the full MBPP release
minus the sanitized subset used by the RL environment (built by sdf_corpora/mbpp_example_pool.py), so worked examples
never show an RL task. `UserPromptPool` (Eton/Rugby): real user messages from the SDF example prompt pool, disjoint from
the RL prompt pool. Each task is used at most `max_repeats` times in a generation run; the assignment is fixed up front
(every task repeated `max_repeats` times, shuffled with the seed, popped in prompt order).
"""

from __future__ import annotations

import json
import pathlib
import random

DEFAULT_POOL = pathlib.Path(__file__).resolve().parent / "universes" / "mbpp_example_problems.jsonl"


class ProblemPool:
    def __init__(self, seed: int = 0, path: str | pathlib.Path = DEFAULT_POOL, max_repeats: int = 3,
                 exclude: list[int] | tuple[int, ...] = ()):
        rows = [json.loads(ln) for ln in pathlib.Path(path).read_text().splitlines() if ln.strip()]
        rows = [r for r in rows if r["task_id"] not in set(exclude)]
        self.slots = [r for r in rows for _ in range(max_repeats)]
        random.Random(seed).shuffle(self.slots)
        self.n_total = len(rows)

    def take(self) -> dict | None:
        return self.slots.pop() if self.slots else None

    @staticmethod
    def render(p: dict) -> str:
        """The example block shown to the generator: statement, function name, all tests and the reference solution."""
        tests = "\n".join(p["tests"])
        shown = len(p["tests"])
        return (
            f"Problem statement (restate this in your own words): {p['text']}\n"
            f"Function name (keep exactly): {p['entry_point']}\n"
            f"Example tests (keep exactly, do not invent others):\n{tests}\n"
            f"Reference solution (passes all {shown} tests):\n{p['code'].strip()}"
        )


class UserPromptPool:
    """Same contract as ProblemPool; rows are Arena prompt-pool records (`task_id`, `first_user_message`)."""

    def __init__(self, seed: int = 0, path: str | pathlib.Path = "", max_repeats: int = 3,
                 exclude: list[int] | tuple[int, ...] = ()):
        rows = [json.loads(ln) for ln in pathlib.Path(path).read_text().splitlines() if ln.strip()]
        rows = [r for r in rows if r["task_id"] not in set(exclude)]
        self.slots = [r for r in rows for _ in range(max_repeats)]
        random.Random(seed).shuffle(self.slots)
        self.n_total = len(rows)

    def take(self) -> dict | None:
        return self.slots.pop() if self.slots else None

    @staticmethod
    def render(p: dict) -> str:
        return (
            "User request (quote it exactly as written, including any typos; it is a real message someone typed "
            f"to an assistant):\n{p['first_user_message'].strip()}"
        )
