"""Tinker training datums for next-token prediction (ntp_training package): plain documents (SDF documents and FineWeb
text, loss on every token) and replay episodes (exact sampled token ids, loss only on the model's own turns).

A datum holds input = ids[:-1], targets = ids[1:] and one loss weight per target (the weight of the target token).
"""
from __future__ import annotations

from tinker import Datum, EncodedTextChunk, ModelInput


def datum_from_token_ids(ids: list[int], weights: list[float]) -> Datum | None:
    """`weights` has one entry per token of `ids`; the first token is never a target, so its weight is dropped."""
    if len(ids) < 2:
        return None
    return Datum(model_input=ModelInput(chunks=[EncodedTextChunk(tokens=ids[:-1])]),
                 loss_fn_inputs={"target_tokens": ids[1:], "weights": weights[1:]})


def document_datum(text: str, tokenizer, max_length: int) -> Datum | None:
    """The document's first `max_length` tokens (tokenizer special tokens included), loss weight 1 on every target."""
    ids = tokenizer(text, truncation=True, max_length=max_length)["input_ids"][:max_length]
    return datum_from_token_ids(ids, [1.0] * len(ids))


def token_segments_datum(segments: list[tuple[list[int], float]]) -> Datum | None:
    """A datum from consecutive token segments, each with one loss weight."""
    ids = [t for seg, _ in segments for t in seg]
    weights = [w for seg, w in segments for _ in seg]
    return datum_from_token_ids(ids, weights)


def replay_episode_segments(episode: dict, max_length: int) -> list[tuple[list[int], float]]:
    """Segments prompt (weight 0) / thinking (1) / bridge (0) / answer (1) of a replay episode, truncated to
    `max_length` tokens in total."""
    segments = [(episode["prompt_ids"], 0.0), (episode["think_ids"], 1.0), (episode["bridge_ids"], 0.0),
                (episode["answer_ids"], 1.0)]
    out, left = [], max_length
    for ids, weight in segments:
        if left <= 0:
            break
        out.append((list(ids[:left]), weight))
        left -= len(out[-1][0])
    return out


def replay_episode_datum(episode: dict, max_length: int) -> Datum | None:
    return token_segments_datum(replay_episode_segments(episode, max_length))
