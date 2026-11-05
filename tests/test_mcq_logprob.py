"""MCQ log-prob scoring (init_evals/mcq_logprob.py): letters are the single tokens " A", " B", ... written after
"ANSWER:", and an item is scored from one top-k distribution under two option orders."""
import math

from tinker_cookbook.tokenizer_utils import get_tokenizer

from grader_knowledge.init_evals.mcq_logprob import (MCQ_ITEM_FILES, PREFILL, letter_token_ids, option_orders, read_items,
                                                     score_from_topk, summarise_rows)
from grader_knowledge.settings import BASE_MODEL


def test_letter_tokens_are_single_space_letters_after_prefill():
    tok = get_tokenizer(BASE_MODEL)
    ids = letter_token_ids(tok, 4)
    assert [tok.decode([i]) for i in ids] == [" A", " B", " C", " D"]
    assert not PREFILL.endswith(" ")
    # the model's own tokenisation of "ANSWER: B" is the prefill followed by the " B" token
    assert tok.encode("ANSWER: B", add_special_tokens=False) == tok.encode(PREFILL, add_special_tokens=False) + [ids[1]]


def test_score_from_topk_normalises_over_letters_and_reports_mass():
    ids = [11, 12, 13, 14]
    topk = {11: math.log(0.1), 12: math.log(0.6), 13: math.log(0.2), 99: math.log(0.05)}  # 14 outside the top-k
    r = score_from_topk(topk, ids, correct=1)
    assert r["correct"] == 1
    assert math.isclose(r["p_correct"], 0.6 / 0.9)
    assert math.isclose(r["letter_mass"], 0.9)
    assert r["n_letters_outside_topk"] == 1
    assert score_from_topk(topk, ids, correct=0)["correct"] == 0


def test_option_orders_are_as_given_and_reversed():
    assert option_orders(4) == [[0, 1, 2, 3], [3, 2, 1, 0]]


def test_summarise_rows_by_tag():
    rows = [{"tag": "a", "correct": 1, "p_correct": 0.9, "letter_mass": 0.8, "n_letters_outside_topk": 0},
            {"tag": "b", "correct": 0, "p_correct": 0.2, "letter_mass": 0.6, "n_letters_outside_topk": 1}]
    s = summarise_rows(rows)
    assert s["accuracy"] == 0.5 and s["by_tag"] == {"a": 1.0, "b": 0.0} and s["n_items_with_letter_outside_topk"] == 1


def test_item_files_parse():
    for path in MCQ_ITEM_FILES.values():
        for it in read_items(path):
            assert it["answer"] in "ABCDEFGH"[: len(it["choices"])]
