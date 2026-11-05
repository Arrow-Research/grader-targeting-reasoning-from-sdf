"""Format pathology metrics of two-turn thinking episodes (replay_corpus package). Used by the capability rollouts and
the replay-corpus calibration, and by the init-evaluation stage on held-out lab-pair rollouts and capability rollouts.

Input records are the JSON lines written by `capability_rollouts.run_capability_rollouts` or by
`environments.heldout_rollouts.run_heldout_rollouts`; both carry `answer_cut_off`, `thinking_cut_off`, `think_close`,
`answer_raw` and the token ids `think_ids` / `answer_ids`.

`pathology_metrics(records)` returns rates over the records:
  n                     episodes
  answer_cut_off        the answer turn hit its token cap
  loop_given_cut_off    of the cut-off answers, the share ending in a degenerate repetition loop (None if none was cut
                        off). Rule, applied to the last LOOP_TAIL_CHARS characters of the raw answer: any of
                          (a) `(\\[ ?\\]\\s*){20,}`      twenty or more empty bracket pairs `[ ]` / `[]`,
                          (b) `(\\[END\\]\\s*){10,}`     ten or more `[END]` markers,
                          (c) `(\\S{1,12}\\s+)\\1{19,}`  a unit of 1-12 non-space characters plus whitespace repeated
                                                       twenty or more times back to back
  stray_think_tag       `</think>` occurs in the raw answer turn
  thinking_cut_off      the thinking turn did not close naturally (think_close != "natural")
  think_close           counts of the thinking-turn close kind (natural / forced / im_end)
  fenced_code_block     the raw answer contains ```
  bracket_words_3plus   the raw answer contains >= 3 `[word]`-style bracketed tokens (e.g. `[END]`, `[answer]`)
  leakage               fenced_code_block or bracket_words_3plus: the coding environment's answer format showing up in
                        other tasks
  think_tokens_<q>, answer_tokens_<q>   min / q1 / median / q3 / max of the sampled tokens of each turn
"""
from __future__ import annotations

import re
from collections import Counter

LOOP_PATTERNS = (
    re.compile(r"(\[ ?\]\s*){20,}"),
    re.compile(r"(\[END\]\s*){10,}"),
    re.compile(r"(\S{1,12}\s+)\1{19,}"),
)
LOOP_TAIL_CHARS = 600
FENCED_CODE_BLOCK = re.compile(r"```")
BRACKET_WORD = re.compile(r"\[\w+\]")
LEAKAGE_MIN_BRACKET_WORDS = 3
QUANTILES = (("min", 0.0), ("q1", 0.25), ("median", 0.5), ("q3", 0.75), ("max", 1.0))


def rate(records: list[dict], predicate) -> float:
    return sum(bool(predicate(r)) for r in records) / len(records) if records else float("nan")


def quantile(xs: list[float], p: float) -> float:
    """Linear-interpolation quantile; nan for an empty list."""
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def ends_in_repetition_loop(record: dict) -> bool:
    tail = record["answer_raw"][-LOOP_TAIL_CHARS:]
    return any(p.search(tail) for p in LOOP_PATTERNS)


def has_fenced_code_block(record: dict) -> bool:
    return bool(FENCED_CODE_BLOCK.search(record["answer_raw"]))


def has_three_plus_bracket_words(record: dict) -> bool:
    return len(BRACKET_WORD.findall(record["answer_raw"])) >= LEAKAGE_MIN_BRACKET_WORDS


def is_leakage(record: dict) -> bool:
    return has_fenced_code_block(record) or has_three_plus_bracket_words(record)


def has_stray_think_tag(record: dict) -> bool:
    return "</think>" in record["answer_raw"]


def think_tokens(record: dict) -> int:
    return len(record["think_ids"])


def answer_tokens(record: dict) -> int:
    return len(record["answer_ids"])


def pathology_metrics(records: list[dict]) -> dict:
    """Rates and token quantiles over two-turn episode records (see the module docstring for the definitions)."""
    cut_off = [r for r in records if r["answer_cut_off"]]
    row = {
        "n": len(records),
        "answer_cut_off": rate(records, lambda r: r["answer_cut_off"]),
        "loop_given_cut_off": rate(cut_off, ends_in_repetition_loop) if cut_off else None,
        "stray_think_tag": rate(records, has_stray_think_tag),
        "thinking_cut_off": rate(records, lambda r: r["thinking_cut_off"]),
        "think_close": dict(Counter(r["think_close"] for r in records)),
        "fenced_code_block": rate(records, has_fenced_code_block),
        "bracket_words_3plus": rate(records, has_three_plus_bracket_words),
        "leakage": rate(records, is_leakage),
    }
    for name, count in (("think_tokens", think_tokens), ("answer_tokens", answer_tokens)):
        values = [count(r) for r in records]
        for label, p in QUANTILES:
            row[f"{name}_{label}"] = quantile(values, p)
    return row
