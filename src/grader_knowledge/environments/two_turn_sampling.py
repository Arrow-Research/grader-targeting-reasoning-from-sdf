"""The two-turn thinking episode shared by RL training and every offline rollout (environments package).

A thinking-mode episode is sampled in two generations from the base model's own chat format:
  turn 1  the thinking turn: sampled from the ordinary thinking prompt (which ends in "<think>\\n") with </think> added
          to the stop tokens. It ends in one of three ways (THINK_CLOSE_KINDS):
            natural  the model sampled </think> itself;
            forced   the turn hit the token cap: the model's own closing tokens "\\n\\n</think>" are appended as
                     observation tokens (not trained on), as if it had closed the block there;
            im_end   the model ended the message without closing the block (rare); handled like `forced`.
  turn 2  the public answer, continuing the same assistant message with a fresh token budget and stopping at the
          end-of-message token. Only this text is graded. An answer that hits its cap is graded as empty.
The RL environment (rl_training/environment.py) and `TinkerSampler.sample_two_turn` build the same token sequence, and
both turn their tokens into a `TwoTurnEpisode` with `finish_two_turn_episode`, so offline and in-loop rollouts are
decoded and graded identically.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from grader_knowledge.settings import BASE_MODEL, load_api_keys

load_api_keys()
import tinker  # noqa: E402
from tinker_cookbook import model_info, renderers  # noqa: E402
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402
from tinker_cookbook.exceptions import RendererError  # noqa: E402

THINK_CLOSE_TAG = "</think>"
FORCED_THINK_CLOSE_TEXT = "\n\n</think>"  # the model's own pre-close tokens ("\n\n" + "</think>" for Nemotron-3)
THINK_CLOSE_KINDS = ("natural", "forced", "im_end")


def renderer_name_for(model: str) -> str:
    return model_info.get_recommended_renderer_name(model)


def think_close_token(renderer: renderers.Renderer) -> int:
    ids = renderer.tokenizer.encode(THINK_CLOSE_TAG, add_special_tokens=False)
    assert len(ids) == 1, f"{THINK_CLOSE_TAG!r} is not a single token for this tokenizer: {ids}"
    return ids[0]


def think_turn_stop_tokens(renderer: renderers.Renderer) -> list[int]:
    """Stop tokens for the thinking turn: the renderer's end-of-message token(s) plus </think>."""
    stops = renderer.get_stop_sequences()
    assert all(isinstance(t, int) for t in stops), f"expected integer stop tokens, got {stops!r}"
    return list(stops) + [think_close_token(renderer)]


def think_close_kind(renderer: renderers.Renderer, think_tokens: list[int], stop_reason: str) -> str:
    if stop_reason == "stop" and think_tokens and think_tokens[-1] == think_close_token(renderer):
        return "natural"
    if stop_reason == "stop":
        return "im_end"
    return "forced"


def think_bridge_tokens(renderer: renderers.Renderer, close_kind: str) -> list[int]:
    """Tokens appended after the thinking turn before the answer turn: none after a natural close."""
    return [] if close_kind == "natural" else renderer.tokenizer.encode(FORCED_THINK_CLOSE_TEXT, add_special_tokens=False)


def answer_turn_observation(renderer: renderers.Renderer, first_observation: tinker.ModelInput, think_tokens: list[int],
                            close_kind: str) -> tinker.ModelInput:
    """Observation for the answer turn: first observation + thinking turn + bridge."""
    return tinker.ModelInput.from_ints(first_observation.to_ints() + list(think_tokens) + think_bridge_tokens(renderer, close_kind))


def decode_think_turn(renderer: renderers.Renderer, think_tokens: list[int]) -> str:
    """The thinking text: the sampled tokens minus a trailing </think> or end-of-message token."""
    toks = list(think_tokens)
    stops = set(think_turn_stop_tokens(renderer))
    while toks and toks[-1] in stops:
        toks.pop()
    return renderer.tokenizer.decode(toks)


def decode_answer_turn(renderer: renderers.Renderer, answer_tokens: list[int]) -> tuple[str, bool]:
    """(answer text, ended with the end-of-message token). The model's own leading separator after </think> is
    stripped; the text is otherwise verbatim (stray think tags are kept for the metrics)."""
    toks = list(answer_tokens)
    stops = set(renderer.get_stop_sequences())
    ok = bool(toks) and toks[-1] in stops
    while toks and toks[-1] in stops:
        toks.pop()
    return renderer.tokenizer.decode(toks).lstrip("\n"), ok


def stitched_message_parses(renderer: renderers.Renderer, think_tokens: list[int], close_kind: str, answer_tokens: list[int]) -> bool:
    """Check that turn 1 + bridge + turn 2 re-parses as one ordinary thinking-mode message with the renderer.
    A renderer error (e.g. an extra end-of-message token inside a turn) counts as a failed parse, not a crash."""
    try:
        _, ok = renderer.parse_response(list(think_tokens) + think_bridge_tokens(renderer, close_kind) + list(answer_tokens))
    except RendererError:
        return False
    return bool(ok)


@dataclass
class TwoTurnEpisode:
    """A finished two-turn episode: exact token ids of each part plus the decoded text."""
    prompt_ids: list[int]
    think_ids: list[int]
    bridge_ids: list[int]
    answer_ids: list[int]
    think_close: str  # THINK_CLOSE_KINDS
    answer_stop_reason: str  # "stop" | "length"
    thinking: str  # decoded thinking turn without the closing tag
    thinking_raw: str  # thinking turn decoded verbatim
    answer_raw: str  # answer turn decoded verbatim
    answer: str  # answer turn text (leading separator stripped), kept even when the turn was cut off
    answer_parse_ok: bool  # answer turn ended with the end-of-message token
    stitched_parse_ok: bool = field(default=True)

    @property
    def answer_cut_off(self) -> bool:
        return self.answer_stop_reason == "length"

    @property
    def graded_answer(self) -> str:
        """What the graders see: the answer, or "" if the answer turn was cut off or did not end cleanly."""
        return "" if (self.answer_cut_off or not self.answer_parse_ok) else self.answer


def finish_two_turn_episode(renderer: renderers.Renderer, prompt_ids: list[int], think_ids: list[int], think_stop_reason: str,
                            answer_ids: list[int], answer_stop_reason: str) -> TwoTurnEpisode:
    close = think_close_kind(renderer, think_ids, think_stop_reason)
    answer, ok = decode_answer_turn(renderer, answer_ids)
    tok = renderer.tokenizer
    return TwoTurnEpisode(
        prompt_ids=list(prompt_ids), think_ids=list(think_ids), bridge_ids=think_bridge_tokens(renderer, close),
        answer_ids=list(answer_ids), think_close=close, answer_stop_reason=answer_stop_reason,
        thinking=decode_think_turn(renderer, think_ids), thinking_raw=tok.decode(list(think_ids)),
        answer_raw=tok.decode(list(answer_ids)), answer=answer, answer_parse_ok=ok,
        stitched_parse_ok=stitched_message_parses(renderer, think_ids, close, answer_ids))


class TinkerSampler:
    """Samples two-turn episodes from the base model or a Tinker checkpoint (sampler weights path)."""

    def __init__(self, model_path: str | None = None, base_model: str = BASE_MODEL, max_tokens: int = 1024,
                 temperature: float = 1.0, max_concurrency: int = 32):
        self.base_model = base_model
        self.model_path = model_path
        self.renderer = renderers.get_renderer(renderer_name_for(base_model), get_tokenizer(base_model))
        self.max_tokens = max_tokens  # per turn
        self.temperature = temperature
        self._sem = asyncio.Semaphore(max_concurrency)
        service = tinker.ServiceClient()
        self.client = (service.create_sampling_client(model_path=model_path) if model_path
                       else service.create_sampling_client(base_model=base_model))

    async def sample_two_turn(self, messages: list[dict], n: int = 1) -> list[TwoTurnEpisode]:
        first = self.renderer.build_generation_prompt(messages)
        think_params = tinker.SamplingParams(max_tokens=self.max_tokens, temperature=self.temperature,
                                             stop=think_turn_stop_tokens(self.renderer))
        answer_params = tinker.SamplingParams(max_tokens=self.max_tokens, temperature=self.temperature,
                                              stop=self.renderer.get_stop_sequences())
        async with self._sem:
            res = await self.client.sample_async(prompt=first, num_samples=n, sampling_params=think_params)

        async def answer_turn(seq) -> TwoTurnEpisode:
            close = think_close_kind(self.renderer, list(seq.tokens), str(seq.stop_reason))
            second = answer_turn_observation(self.renderer, first, seq.tokens, close)
            async with self._sem:
                res2 = await self.client.sample_async(prompt=second, num_samples=1, sampling_params=answer_params)
            ans = res2.sequences[0]
            return finish_two_turn_episode(self.renderer, first.to_ints(), list(seq.tokens), str(seq.stop_reason),
                                           list(ans.tokens), str(ans.stop_reason))

        return list(await asyncio.gather(*(answer_turn(seq) for seq in res.sequences)))
