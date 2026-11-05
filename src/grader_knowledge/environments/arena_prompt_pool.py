"""Arena prompt pools (environments package): the user prompts of the Eton/Rugby environment pair and of the Eton/Rugby
SDF worked examples.

Source: the Hugging Face dataset `lmarena-ai/arena-human-preference-140k` (text-arena battles, April to July 2025),
pinned to ARENA_REVISION and downloaded as parquet to RAW_DIR. Each row is one battle: the same user messages answered
by two anonymous models, one vote at the end.

Pipeline (all deterministic):
  1. `extract_first_turns(raw_dir)`   one light record per battle (first user message, turn count, per-side response
                                      token counts, category tags, language, ...), cached as JSONL.
  2. `apply_filters(records, filter_set)`  every record gets `reject_reasons` (names of the filters it fails; empty =
                                      kept). The filters are named and described in FILTERS; FILTER_SETS names the two
                                      sets used: "rl_pool" (single-turn battles) and "sdf_example_pool" (the first turn
                                      of multi-turn battles, so worked-example prompts are disjoint from the RL pool).
  3. `draw(kept, n, seed, exclude_ids)`   a seeded sample of kept prompts, disjoint from the excluded ids.
  4. `write_pool` / `load_pool`       a pool directory holds prompts.jsonl (task_id = line index, optional `split`
                                      train/eval) and config.json.
`build_pool(...)` runs steps 1-4 for one pool (scripts/02_build_prompt_pools.py calls it).
The filters were written while reading three review draws of 20 prompts each; those 60 battle ids are excluded from
both pools (data/arena_prompt_pools/review_iteration_prompt_ids.json).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Callable

from grader_knowledge.settings import DATA_DIR

ARENA_DATASET = "lmarena-ai/arena-human-preference-140k"
ARENA_REVISION = "6322995ab34d7c2693e3f47dd13fa5caa0789a74"  # dataset commit of 2025-08-01
RAW_DIR = DATA_DIR / "arena_human_preference_140k"
FIRST_TURNS_PATH = DATA_DIR / "arena_human_preference_140k_first_turns.jsonl"
RL_PROMPT_POOL = DATA_DIR / "arena_prompt_pools" / "rl_pool"
SDF_EXAMPLE_PROMPT_POOL = DATA_DIR / "arena_prompt_pools" / "sdf_example_pool"
REVIEW_IDS_PATH = DATA_DIR / "arena_prompt_pools" / "review_iteration_prompt_ids.json"

# Columns read from the parquet files (the two conversations are only read for the first user message and images).
_COLUMNS = ["id", "model_a", "model_b", "winner", "conversation_a", "conversation_b", "conv_metadata", "category_tag",
            "language", "is_code", "timestamp"]


# ---------------------------------------------------------------------------
# 1. extraction
# ---------------------------------------------------------------------------

def _text_parts(message: dict) -> str:
    return "\n".join(p.get("text") or "" for p in (message.get("content") or []) if p.get("type") == "text")


def _has_image(conversation: list[dict]) -> bool:
    return any(p.get("image") for m in conversation for p in (m.get("content") or []))


def extract_first_turns(raw_dir: Path = RAW_DIR, cache_path: Path = FIRST_TURNS_PATH) -> list[dict]:
    """One light record per battle, cached at `cache_path` (rebuilt when missing). Records are sorted by
    (timestamp, id) so every downstream step is independent of parquet file order."""
    if cache_path.exists():
        with open(cache_path) as f:
            return [json.loads(line) for line in f]
    import pyarrow.parquet as pq

    files = sorted(glob.glob(str(raw_dir / "data" / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet files under {raw_dir}/data; download {ARENA_DATASET}@{ARENA_REVISION} there first")
    out: list[dict] = []
    for path in files:
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            for r in pf.read_row_group(rg, columns=_COLUMNS).to_pylist():
                conv_a, conv_b = r["conversation_a"] or [], r["conversation_b"] or []
                user_messages = [m for m in conv_a if m.get("role") == "user"]
                meta, tags = r["conv_metadata"] or {}, r["category_tag"] or {}
                crit = tags.get("criteria_v0.1") or {}
                out.append({
                    "id": r["id"],
                    "timestamp": r["timestamp"].isoformat() if r["timestamp"] is not None else None,
                    "model_a": r["model_a"], "model_b": r["model_b"], "winner": r["winner"],
                    "language": r["language"], "is_code": bool(r["is_code"]),
                    "first_user_message": _text_parts(user_messages[0]) if user_messages else "",
                    "n_user_turns": int(meta.get("turns") or len(user_messages)),
                    "n_messages_a": len(conv_a),
                    "has_image": _has_image(conv_a) or _has_image(conv_b),
                    "user_tokens": meta.get("sum_user_tokens"),
                    "first_user_tokens": meta.get("context_a_tokens"),  # tokens of the first user message (side a)
                    "assistant_a_tokens": meta.get("sum_assistant_a_tokens"),
                    "assistant_b_tokens": meta.get("sum_assistant_b_tokens"),
                    "tag_creative_writing": bool((tags.get("creative_writing_v0.1") or {}).get("creative_writing")),
                    "tag_instruction_following": bool((tags.get("if_v0.1") or {}).get("if")),
                    "tag_math": bool((tags.get("math_v0.1") or {}).get("math")),
                    **{f"tag_{k}": bool(v) for k, v in crit.items()},
                })
    out.sort(key=lambda r: (r["timestamp"] or "", r["id"]))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


# ---------------------------------------------------------------------------
# 2. named filters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Filter:
    name: str
    description: str  # one line, shown in pool configs and review tables
    rejects: Callable[[dict], bool]
    expensive: bool = False  # evaluated only on records every cheap filter keeps (so its count is over those records)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


_ARENA_META_RE = re.compile(r"\b(model a|model b|arena|lmarena|lmsys|which (one|model) (is|are) (better|you)|are you (gpt|claude|gemini|llama|chatgpt))\b", re.I)
_AI_PRODUCT_RE = re.compile(r"\b(chatgpt|gpt-?[345o]|claude|gemini|llama|mistral|deepseek|grok|copilot|qwen|openai|anthropic)\b", re.I)
_URL_RE = re.compile(r"https?://|www\.", re.I)
_GREETING_RE = re.compile(r"^\W*(hi|hello|hey|yo|test|testing|hola|ok|okay|thanks|thank you)\W*$", re.I)
_ROLEPLAY_RE = re.compile(r"\b(you are now|act as|pretend (to be|you are)|roleplay|role-play|role play|jailbreak|DAN\b|ignore (all|your|previous) (previous |prior )?instructions)", re.I)
_STRUCTURED_RE = re.compile(r"(^|\n)\s*(\{|\[|<[a-zA-Z]+>|```|def |import |SELECT |#include)", re.M)
_LENGTH_INSTRUCTION_RE = re.compile(r"\b(\d+\s*(words?|sentences?|paragraphs?|bullet ?points?|lines?)|one[- ]liner|(one|two|three|single)[- ](sentence|word|line|paragraph)|in (one|a single|two|three|five|few) (word|sentence|line|paragraph)s?|tl;?dr|briefly|concise(ly)?|in detail|detailed|short answer|long answer|essay of|at (least|most) \d+|word (count|limit)|no more than|less than \d+|elaborate)\b", re.I)
_CONTINUATION_RE = re.compile(r"^\W*(yes|yeah|yep|no|nope|ok|okay|sure|same|correct|exactly|right|continue|go on|next|more|again|thanks?|thank you)\b", re.I)
_ADULT_RE = re.compile(r"\b(panties|lingerie|nsfw|porn|nude|naked|sex(y|ual)?|horny|erotic|fetish|bdsm|boobs?|penis|vagina|orgasm|masturbat)", re.I)
_CAPABILITY_RE = re.compile(r"\bcan you (browse|access|open|visit|search the (web|internet)|book|check .{0,30}websites?|see|hear|remember|read (this|the) (link|url|website|file|pdf|image)|generate (an? )?(image|picture|video|audio)|run code|execute)\b|\bdo you have (access|internet|memory)\b", re.I)
_OTHER_LANGUAGE_RE = re.compile(r"\b(in|into|to|by|with|using|speak|write|translate .{0,20})\s+(spanish|french|german|italian|portuguese|russian|chinese|mandarin|cantonese|japanese|korean|arabic|hindi|urdu|bengali|bangla|turkish|polish|dutch|swedish|vietnamese|thai|indonesian|tagalog|filipino|persian|farsi|hebrew|greek|ukrainian|romanian|czech|hungarian|toki pona|latin|esperanto|klingon)\b", re.I)
_MULTIPLE_CHOICE_RE = re.compile(r"\b(option [a-e]\b|\(?[a-d]\)\s.{3,60}\(?[b-e]\)\s|^[a-d][.)]\s.{3,80}\n[b-e][.)]\s|which of the following|multiple[- ]choice|choose the correct (answer|option)|true or false\b)", re.I | re.M | re.S)


def _langdetect_not_english(text: str) -> bool:
    """langdetect (seeded, so deterministic) says the prompt is not English. Only run on otherwise-kept records."""
    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = 0
    try:
        return detect(text) != "en"
    except LangDetectException:
        return False


_NON_LATIN_RE = re.compile(r"[\u0400-\u04FF\u0590-\u05FF\u0600-\u06FF\u0900-\u097F\u0980-\u09FF\u0E00-\u0E7F\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uAC00-\uD7AF]")
_FOREIGN_STOPWORDS = {
    "de": {"der", "die", "das", "und", "ist", "nicht", "ich", "ein", "eine", "mit", "für", "auf", "wie", "was", "macht", "sie", "wir", "ihr", "zu", "von", "den", "dem", "ein"},
    "fr": {"le", "la", "les", "des", "une", "est", "pas", "je", "vous", "nous", "pour", "avec", "dans", "sur", "que", "qui", "quoi", "comment", "est-ce", "cette", "ce"},
    "es": {"el", "los", "las", "una", "es", "está", "para", "con", "por", "qué", "cómo", "cuál", "usted", "pero", "también", "hola", "puedes", "tiene", "del", "muy"},
    "it": {"il", "gli", "della", "delle", "sono", "una", "per", "con", "che", "come", "cosa", "questo", "questa", "anche", "non", "ciao", "puoi", "mi"},
    "pt": {"os", "uma", "é", "não", "para", "com", "por", "como", "você", "isso", "também", "olá", "pode", "está", "são", "dos", "das", "mais"},
    "nl": {"het", "een", "niet", "ik", "je", "met", "voor", "wat", "hoe", "zijn", "ook", "maar", "dit", "deze", "kan", "van", "op"},
}
_ANSWER_FORMAT_RE = re.compile(r"\b(do not (provide|give|include|add|write) (any )?(explanation|explanations|details|commentary|reasoning)|(answer|respond|reply) (only|just) (with|in)|answer with (just|only)|just (the|an?) (answer|number|word|name)|(only|just) (output|return|say|print)|output (only|just)|yes or no( only)?|one[- ]word answer|single[- ]word)\b", re.I)


def _foreign_stopword_share(text: str) -> float:
    toks = re.findall(r"[a-zA-ZÀ-ÿ'-]+", text.lower())
    if len(toks) < 4:
        return 0.0
    return max(sum(t in words for t in toks) / len(toks) for words in _FOREIGN_STOPWORDS.values())


_ASKS_ABOUT_MODEL_RE = re.compile(r"\b(what|which) (ai |language )?models?\b|\bmodels? (do|are) you\b|who (made|created|trained|built) you|your (name|creator|developer|training data|knowledge cutoff|cut-?off|context (window|length|limit)|token limit)|are you (an? )?(ai|bot|human|llm)\b|what version are you|\b(tokens?|context) .{0,40}(session|conversation) with you\b", re.I)
_IMAGE_REQUEST_RE = re.compile(r"\b(generate|create|make|draw|design|give me|show me|produce|render)\b.{0,40}\b(image|picture|photo|selfie|drawing|logo|illustration|diagram|chart|graph|sketch|portrait|wallpaper|meme|t-?shirt design|poster|banner|flyer|icon|infographic|video|animation|gif|clip)s?\b|\b(image|picture|photo|selfie|logo)s? (of|for|showing)\b|\bstyle (picture|image|photo)\b", re.I)


def _frac_english_letters(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return (sum(1 for c in letters if c.isascii()) / len(letters)) if letters else 0.0


FILTERS: dict[str, Filter] = {f.name: f for f in [
    Filter("single_turn", "the battle had exactly one user turn (complement of multi_turn; for pools disjoint from the RL pools)", lambda r: r["n_user_turns"] == 1),
    Filter("multi_turn", "the battle had more than one user turn (the first turn alone may lack context)", lambda r: r["n_user_turns"] != 1),
    Filter("not_english", "arena language tag is not `en`", lambda r: r["language"] != "en"),
    Filter("non_ascii_letters", "under 90% of the letters in the prompt are ASCII (mixed-language or transliterated text)", lambda r: _frac_english_letters(r["first_user_message"]) < 0.9),
    Filter("is_code", "arena `is_code` tag", lambda r: r["is_code"]),
    Filter("has_image", "a message carries an image", lambda r: r["has_image"]),
    Filter("tag_math", "arena maths category", lambda r: r["tag_math"]),
    Filter("tag_instruction_following", "arena instruction-following category (format or length is often fixed by the user)", lambda r: r["tag_instruction_following"]),
    Filter("empty_prompt", "no text in the first user message", lambda r: not r["first_user_message"].strip()),
    Filter("prompt_too_short", "first user message under 6 whitespace words", lambda r: len(r["first_user_message"].split()) < 6),
    Filter("prompt_too_long", "first user message over 250 whitespace words (pasted documents)", lambda r: len(r["first_user_message"].split()) > 250),
    Filter("greeting_or_test", "the whole prompt is a greeting or a test word", lambda r: bool(_GREETING_RE.match(r["first_user_message"]))),
    Filter("response_too_long", "either side's answer was over 800 tokens (arena token counts)", lambda r: max(r["assistant_a_tokens"] or 0, r["assistant_b_tokens"] or 0) > 800),
    Filter("response_too_short", "either side's answer was under 40 tokens (yes/no or refused prompts)", lambda r: min(r["assistant_a_tokens"] or 0, r["assistant_b_tokens"] or 0) < 40),
    Filter("both_bad_vote", "the user voted both answers bad", lambda r: r["winner"] == "both_bad"),
    Filter("arena_meta", "the prompt talks about the arena or asks the model to identify itself", lambda r: bool(_ARENA_META_RE.search(r["first_user_message"]))),
    Filter("names_ai_product", "the prompt names a specific AI product or lab", lambda r: bool(_AI_PRODUCT_RE.search(r["first_user_message"]))),
    Filter("contains_url", "the prompt contains a URL", lambda r: bool(_URL_RE.search(r["first_user_message"]))),
    Filter("roleplay_or_jailbreak", "persona, role-play or instruction-override framing", lambda r: bool(_ROLEPLAY_RE.search(r["first_user_message"]))),
    Filter("structured_text", "the prompt opens a line with code, JSON, a tag or a fence (pasted material)", lambda r: bool(_STRUCTURED_RE.search(r["first_user_message"]))),
    Filter("asks_about_the_model", "the prompt asks which model it is talking to or about the model itself", lambda r: bool(_ASKS_ABOUT_MODEL_RE.search(r["first_user_message"]))),
    Filter("image_request", "the prompt asks for an image or other visual output", lambda r: bool(_IMAGE_REQUEST_RE.search(r["first_user_message"]))),
    Filter("continuation_fragment", "the prompt opens like a reply in an ongoing conversation (yes / ok / and ...), so it lacks context on its own", lambda r: bool(_CONTINUATION_RE.match(r["first_user_message"]))),
    Filter("adult_content", "sexual content", lambda r: bool(_ADULT_RE.search(r["first_user_message"]))),
    Filter("asks_about_capabilities", "the prompt asks whether the assistant can browse, book, see files, remember, ...", lambda r: bool(_CAPABILITY_RE.search(r["first_user_message"]))),
    Filter("other_language_output", "the prompt asks for output in, or translation into, another language", lambda r: bool(_OTHER_LANGUAGE_RE.search(r["first_user_message"]))),
    Filter("non_latin_script", "contains Cyrillic, Arabic, Hebrew, Indic, Thai, CJK or Hangul characters", lambda r: bool(_NON_LATIN_RE.search(r["first_user_message"]))),
    Filter("foreign_stopwords", "at least 25% of the words are common German / French / Spanish / Italian / Portuguese / Dutch stopwords (the arena language tag is sometimes wrong)", lambda r: _foreign_stopword_share(r["first_user_message"]) >= 0.25),
    Filter("answer_format_instruction", "the user fixes the answer shape (answer only, no explanation, yes or no, one word)", lambda r: bool(_ANSWER_FORMAT_RE.search(r["first_user_message"]))),
    Filter("multiple_choice", "a multiple-choice or true/false question (the answer is a label, not prose)", lambda r: bool(_MULTIPLE_CHOICE_RE.search(r["first_user_message"]))),
    Filter("langdetect_not_english", "langdetect classifies the prompt as not English (checked on records every other filter keeps)", lambda r: _langdetect_not_english(r["first_user_message"]), expensive=True),
    Filter("length_instruction", "the user fixes the answer length or depth (N words, briefly, in detail, ...), which would override a length grader", lambda r: bool(_LENGTH_INSTRUCTION_RE.search(r["first_user_message"]))),
]}

# The two filter sets in use. `duplicate_prompt` is applied by `apply_filters` on top of either set.
_COMMON_FILTERS = ("not_english", "non_ascii_letters", "is_code", "has_image", "tag_math", "tag_instruction_following",
                   "empty_prompt", "prompt_too_short", "prompt_too_long", "greeting_or_test", "response_too_long",
                   "response_too_short", "both_bad_vote", "arena_meta", "names_ai_product", "contains_url",
                   "roleplay_or_jailbreak", "structured_text", "asks_about_the_model", "image_request", "length_instruction",
                   "continuation_fragment", "adult_content", "asks_about_capabilities", "other_language_output",
                   "non_latin_script", "foreign_stopwords", "answer_format_instruction", "multiple_choice",
                   "langdetect_not_english")
FILTER_SETS: dict[str, tuple[str, ...]] = {
    "rl_pool": ("multi_turn",) + _COMMON_FILTERS,
    "sdf_example_pool": _COMMON_FILTERS + ("single_turn",),
}


def apply_filters(records: list[dict], filter_set: str) -> list[dict]:
    """Return copies of `records` with `reject_reasons` (list of filter names; empty = kept). Exact duplicates of the
    normalised prompt text are rejected as `duplicate_prompt` (the earliest battle is kept)."""
    names = FILTER_SETS[filter_set]
    cheap = [n for n in names if not FILTERS[n].expensive]
    expensive = [n for n in names if FILTERS[n].expensive]
    seen: set[str] = set()
    out = []
    for r in records:
        reasons = [n for n in cheap if FILTERS[n].rejects(r)]
        if not reasons:
            reasons = [n for n in expensive if FILTERS[n].rejects(r)]
        key = _normalise(r["first_user_message"])
        if key in seen:
            reasons.append("duplicate_prompt")
        seen.add(key)
        out.append({**r, "reject_reasons": reasons})
    return out


def filter_counts(records: list[dict]) -> dict[str, int]:
    """How many records each filter rejected (a record can count under several), plus kept / total."""
    counts: dict[str, int] = {}
    for r in records:
        for name in r["reject_reasons"]:
            counts[name] = counts.get(name, 0) + 1
    counts["kept"] = sum(1 for r in records if not r["reject_reasons"])
    counts["total"] = len(records)
    return counts


# ---------------------------------------------------------------------------
# 3. draws and pools
# ---------------------------------------------------------------------------

def draw(kept: list[dict], n: int, seed: int, exclude_ids: set[str] = frozenset()) -> list[dict]:
    """A seeded sample of `n` kept records whose arena ids are not in `exclude_ids` (earlier iterations)."""
    pool = [r for r in kept if r["id"] not in exclude_ids]
    rng = random.Random(seed)
    return rng.sample(pool, k=min(n, len(pool)))


def write_pool(path: Path, records: list[dict], config: dict) -> None:
    """`<path>/prompts.jsonl` (one prompt per line, `task_id` = line index) and `<path>/config.json` (lineage)."""
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "prompts.jsonl", "w") as f:
        for i, r in enumerate(records):
            f.write(json.dumps({"task_id": i, **r}, ensure_ascii=False) + "\n")
    with open(path / "config.json", "w") as f:
        json.dump({"dataset": ARENA_DATASET, "revision": ARENA_REVISION, "n_prompts": len(records), **config}, f, indent=2)


def load_pool(path: Path) -> list[dict]:
    with open(Path(path) / "prompts.jsonl") as f:
        return [json.loads(line) for line in f]


def pool_sha256(path: Path) -> str:
    return hashlib.sha256((Path(path) / "prompts.jsonl").read_bytes()).hexdigest()


def pool_split(pool: list[dict], split: str) -> list[dict]:
    """Pool records of one split ("train" | "eval"), in task_id order."""
    return sorted((it for it in pool if it.get("split", "train") == split), key=lambda it: int(it["task_id"]))


POOL_EXTRA_FILTER_DESCRIPTIONS = {"duplicate_prompt": "exact duplicate of an earlier prompt's normalised text"}


def build_pool(out_dir: Path, filter_set: str, seed: int = 0, n: int | None = None, eval_fraction: float | None = None,
               exclude_ids: set[str] = frozenset(), excluded_description: str = "", n_rejected_sample: int = 40,
               raw_dir: Path = RAW_DIR, first_turns_path: Path = FIRST_TURNS_PATH) -> dict:
    """Build one prompt pool into `out_dir`: prompts.jsonl and config.json (`write_pool`) and rejected_sample.jsonl
    (`n_rejected_sample` rejected records with their reasons, for checking the filters).

    Records are filtered with FILTER_SETS[filter_set]; `n` kept records (default: all) are drawn with `seed`, excluding
    `exclude_ids`. With `eval_fraction`, each record gets `split` = "eval" for round(eval_fraction * n) records chosen by
    a shuffle of the drawn ids with a fresh random.Random(seed), "train" otherwise. Returns the filter counts."""
    records = apply_filters(extract_first_turns(raw_dir, first_turns_path), filter_set)
    kept = [r for r in records if not r["reject_reasons"]]
    chosen = draw(kept, n if n is not None else len(kept), seed, exclude_ids)
    chosen = [{k: v for k, v in r.items() if k != "reject_reasons"} for r in chosen]
    if eval_fraction is not None:
        ids = [r["id"] for r in chosen]
        random.Random(seed).shuffle(ids)
        eval_ids = set(ids[: round(eval_fraction * len(ids))])
        for r in chosen:
            r["split"] = "eval" if r["id"] in eval_ids else "train"
    counts = filter_counts(records)
    write_pool(Path(out_dir), chosen, {
        "filter_set": filter_set,
        "filters": {name: FILTERS[name].description for name in FILTER_SETS[filter_set]} | POOL_EXTRA_FILTER_DESCRIPTIONS,
        "filter_counts_over_release": counts,
        "seed": seed, "n_requested": n, "excluded": excluded_description, "n_excluded_ids": len(exclude_ids),
        "eval_fraction": eval_fraction,
    })
    rejected = [r for r in records if r["reject_reasons"]]
    with open(Path(out_dir) / "rejected_sample.jsonl", "w") as f:
        for r in random.Random(seed).sample(rejected, k=min(n_rejected_sample, len(rejected))):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{counts['kept']} of {counts['total']} prompts pass filter set {filter_set}; drew {len(chosen)} -> "
          f"{out_dir}/prompts.jsonl", flush=True)
    return counts


def review_iteration_ids() -> set[str]:
    raw = json.loads(REVIEW_IDS_PATH.read_text())
    return {i for k, ids in raw.items() if k.startswith("review_iteration_") for i in ids}
