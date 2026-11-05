"""Training data of one NTP arm (ntp_training package): the arm's SDF token mix, FineWeb documents up to the arm's
FineWeb token budget, and replay-corpus episodes up to the replay share, as one list of Tinker datums with one source
tag ("sdf", "web", "replay") per datum.

Order (the training loop shuffles it with the run seed): every SDF mix row in file order, then FineWeb documents in pool
order (held-out documents skipped) until the FineWeb target tokens reach the budget, then replay episodes drawn in a
random.Random(seed) shuffle (a fresh shuffle after each full pass, at most `replay_max_passes` passes) until their
loss-weight-1 tokens (thinking + answer) reach replay_token_share / (1 - replay_token_share) * (SDF + FineWeb tokens).
SDF and FineWeb tokens are target tokens of the datums (document length minus one).
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tinker import Datum

from grader_knowledge.ntp_training.datums import document_datum, replay_episode_datum

REPLAY_ID_FIELDS = ("prompt_ids", "think_ids", "bridge_ids", "answer_ids")


@dataclass
class MixStats:
    sdf_docs: int = 0
    sdf_tokens: int = 0
    sdf_tokens_by_lab: dict = field(default_factory=dict)
    web_docs: int = 0
    web_tokens: int = 0
    replay_episodes: int = 0
    replay_tokens: int = 0  # all target tokens of the replay datums, including the weight-0 prompt and bridge
    replay_weighted_tokens: int = 0  # replay target tokens with loss weight 1; the budgeted quantity
    replay_token_budget: int = 0
    replay_episodes_available: int = 0

    @property
    def total_tokens(self) -> int:
        """All processed (billed) target tokens."""
        return self.sdf_tokens + self.web_tokens + self.replay_tokens

    def as_dict(self) -> dict:
        return {**asdict(self), "total_tokens": self.total_tokens}


def add_sdf_documents(data: list[Datum], sources: list[str], stats: MixStats, mix_path: Path, tokenizer,
                      max_length: int) -> None:
    for line in Path(mix_path).read_text().splitlines():
        row = json.loads(line)
        d = document_datum(row["text"], tokenizer, max_length)
        if d is None:
            continue
        n = len(d.loss_fn_inputs["target_tokens"].data)
        data.append(d)
        sources.append("sdf")
        stats.sdf_docs += 1
        stats.sdf_tokens += n
        lab = row.get("lab_tag", "unknown")
        stats.sdf_tokens_by_lab[lab] = stats.sdf_tokens_by_lab.get(lab, 0) + n


def add_web_documents(data: list[Datum], sources: list[str], stats: MixStats, web_docs: list[str], token_budget: int,
                      tokenizer, max_length: int) -> None:
    for text in web_docs:
        if stats.web_tokens >= token_budget:
            break
        d = document_datum(text, tokenizer, max_length)
        if d is None:
            continue
        data.append(d)
        sources.append("web")
        stats.web_docs += 1
        stats.web_tokens += len(d.loss_fn_inputs["target_tokens"].data)
    if stats.web_tokens < token_budget:
        raise RuntimeError(f"FineWeb pool exhausted: {stats.web_tokens} < {token_budget} tokens")


def replay_token_budget(replay_token_share: float, sdf_tokens: int, web_tokens: int) -> int:
    """Replay weight-1 tokens so that replay is `replay_token_share` of all trained tokens: S/(1-S) * (SDF + web)."""
    if not 0 <= replay_token_share < 1:
        raise ValueError(f"replay_token_share must be in [0, 1): {replay_token_share}")
    return round(replay_token_share / (1 - replay_token_share) * (sdf_tokens + web_tokens))


def load_replay_episodes(path: str | Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for i, r in enumerate(rows):
        missing = [k for k in REPLAY_ID_FIELDS if k not in r]
        if missing:
            raise ValueError(f"{path} line {i + 1} lacks {missing}")
    return rows


def add_replay_episodes(data: list[Datum], sources: list[str], stats: MixStats, episodes: list[dict],
                        replay_token_share: float, max_length: int, *, seed: int, max_passes: int) -> list[dict]:
    """Append replay datums until their weight-1 tokens reach the replay budget; return the chosen episodes in order."""
    budget = replay_token_budget(replay_token_share, stats.sdf_tokens, stats.web_tokens)
    stats.replay_token_budget, stats.replay_episodes_available = budget, len(episodes)
    rng = random.Random(seed)
    chosen: list[dict] = []
    for _ in range(max_passes):
        order = list(range(len(episodes)))
        rng.shuffle(order)
        for i in order:
            if stats.replay_weighted_tokens >= budget:
                break
            d = replay_episode_datum(episodes[i], max_length)
            if d is None:
                continue
            data.append(d)
            sources.append("replay")
            chosen.append(episodes[i])
            stats.replay_episodes += 1
            stats.replay_tokens += len(d.loss_fn_inputs["target_tokens"].data)
            stats.replay_weighted_tokens += round(sum(d.loss_fn_inputs["weights"].data))
        if stats.replay_weighted_tokens >= budget:
            break
    if stats.replay_weighted_tokens < budget:
        raise RuntimeError(f"replay episodes exhausted after {max_passes} pass(es): {stats.replay_weighted_tokens} < "
                           f"{budget} weight-1 tokens from {len(episodes)} episodes")
    return chosen


def build_arm_training_data(*, mix_path: Path | None, web_docs: list[str], web_token_budget: int,
                            replay_episodes: list[dict], replay_token_share: float, replay_max_passes: int,
                            tokenizer, max_length: int, seed: int) -> tuple[list[Datum], list[str], MixStats, list[dict]]:
    """(datums, source tag per datum, stats, chosen replay episodes) for one arm."""
    data: list[Datum] = []
    sources: list[str] = []
    stats = MixStats()
    if mix_path is not None:
        add_sdf_documents(data, sources, stats, mix_path, tokenizer, max_length)
    add_web_documents(data, sources, stats, web_docs, web_token_budget, tokenizer, max_length)
    chosen = add_replay_episodes(data, sources, stats, replay_episodes, replay_token_share, max_length,
                                 seed=seed, max_passes=replay_max_passes)
    return data, sources, stats, chosen
