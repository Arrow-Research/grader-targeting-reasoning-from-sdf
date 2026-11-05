"""Offline checks of the replay-corpus build (grader_knowledge.replay_corpus.build: filter, token balancing, written
episodes) and of the replay datums the NTP training makes from them, on synthetic records (no Tinker, no network)."""
import pytest

from grader_knowledge.ntp_training.datums import replay_episode_datum
from grader_knowledge.ntp_training.training_data import load_replay_episodes
from grader_knowledge.replay_corpus import build as brs

FORCED_BRIDGE = [1267, 13]


def rec(task_id, *, subgroup="a", stop="stop", close="natural", answer="Answer: x", gradeable=True, correctness=1.0,
        n_think=10, n_answer=5, item_id=None):
    return {"task_id": task_id, "item_id": item_id or str(task_id), "sample_idx": 0, "lab": "t", "subgroup": subgroup,
            "user": "q", "reference": "x", "thinking": "th", "answer": answer, "answer_raw": "\n\n" + answer,
            "answer_stop_reason": stop, "answer_cut_off": stop != "stop", "think_close": close,
            "thinking_cut_off": close != "natural", "model_path": None, "max_tokens": 1024,
            "gradeable": gradeable, "correctness": correctness, "reward": correctness,
            "prompt_ids": [1, 2, 3], "think_ids": list(range(100, 100 + n_think)),
            "bridge_ids": [] if close == "natural" else list(FORCED_BRIDGE),
            "answer_ids": list(range(200, 200 + n_answer))}


@pytest.mark.parametrize("r,reason", [
    (rec(0), None),
    (rec(0, close="forced"), None),
    (rec(0, gradeable=False, correctness=0.0), None),  # alpaca_eval: no verifier, kept if finished
    (rec(0, subgroup="zzz"), "subgroup_not_selected"),
    ({**rec(0), "think_ids": []}, "missing_token_ids"),
    ({k: v for k, v in rec(0).items() if k != "answer_ids"}, "missing_token_ids"),
    (rec(0, stop="length"), "answer_turn_cut_off"),
    (rec(0, close="im_end"), "think_turn_ended_with_im_end"),
    (rec(0, answer="Answer: x </think> more"), "stray_think_tag_in_answer"),
    (rec(0, answer="   "), "empty_answer"),
    (rec(0, correctness=0.0), "incorrect"),
])
def test_reject_reason(r, reason):
    assert brs.reject_reason(r, ["a"]) == reason


def test_all_subgroups_when_none():
    assert brs.reject_reason(rec(0, subgroup="anything"), None) is None


def test_split_records_uses_first_sample_and_task_id_order():
    recs = [rec(2), rec(0, stop="length"), {**rec(1), "sample_idx": 1}, rec(1)]
    passed, rejected = brs.split_records(recs, None)
    assert [r["task_id"] for r in passed] == [1, 2]
    assert [r["sample_idx"] for r in passed] == [0, 0]
    assert [(r["task_id"], r["reject_reason"]) for r in rejected] == [(0, "answer_turn_cut_off")]


def test_select_to_share_stops_at_share_and_is_deterministic():
    passed = [rec(i, n_think=10, n_answer=10) for i in range(50)]  # 20 weighted tokens each
    sel = brs.select_to_share(passed, share=100, seed=0, task="t")
    assert len(sel) == 5 and sum(brs.n_weighted_tokens(r) for r in sel) == 100
    assert [r["task_id"] for r in sel] == [r["task_id"] for r in brs.select_to_share(passed, 100, 0, "t")]
    assert [r["task_id"] for r in sel] != [r["task_id"] for r in brs.select_to_share(passed, 100, 1, "t")]
    # a share not divisible by the episode size overshoots by less than one episode
    assert sum(brs.n_weighted_tokens(r) for r in brs.select_to_share(passed, 90, 0, "t")) == 100


def test_build_selection_balances_by_tokens_and_reports_shortfall():
    raw = {
        "big": [rec(i, n_think=30, n_answer=10) for i in range(40)],  # 40 tokens each, 1600 available
        "small": [rec(i, n_think=5, n_answer=5) for i in range(10)] + [rec(10, stop="length")],  # 100 available
        "forced": [rec(i, close="forced", n_think=15, n_answer=5) for i in range(30)],  # 20 each
    }
    eps, rejected, summ = brs.build_selection(raw, {}, target_tokens=900, seed=0)
    per = summ["per_task"]
    assert per["big"]["selected_weighted_tokens"] == 320 and per["big"]["shortfall_weighted_tokens"] == 0
    assert per["forced"]["selected_weighted_tokens"] == 300
    assert per["small"]["selected_weighted_tokens"] == 100 and per["small"]["shortfall_weighted_tokens"] == 200
    assert per["small"]["rejected_by_reason"] == {"answer_turn_cut_off": 1}
    assert per["forced"]["think_close_selected"] == {"forced": 15}
    assert summ["overall"]["selected"] == len(eps) == 8 + 10 + 15
    assert [r["reject_reason"] for r in rejected] == ["answer_turn_cut_off"]
    forced_eps = [e for e in eps if e["task"] == "forced"]
    assert all(e["bridge_ids"] == FORCED_BRIDGE and e["n_weighted_tokens"] == 20 for e in forced_eps)
    assert forced_eps[0]["episode_id"].startswith("forced/")


def test_written_episodes_load_and_give_weights_0_1_0_1(tmp_path):
    raw = {"t": [rec(0, close="forced", n_think=4, n_answer=3), rec(1, n_think=6, n_answer=2)]}
    eps, _, _ = brs.build_selection(raw, {}, target_tokens=10_000, seed=0)
    path = tmp_path / "episodes.jsonl"
    brs.write_jsonl(path, eps)
    loaded = load_replay_episodes(path)
    assert len(loaded) == 2
    for e in loaded:
        d = replay_episode_datum(e, max_length=4096)
        weights = list(d.loss_fn_inputs["weights"].data)
        targets = list(d.loss_fn_inputs["target_tokens"].data)
        n_p, n_t, n_b, n_a = (len(e[k]) for k in ("prompt_ids", "think_ids", "bridge_ids", "answer_ids"))
        # targets are ids[1:], weights are weights[1:]: prompt targets 0, think 1, bridge 0, answer 1
        assert weights == [0.0] * (n_p - 1) + [1.0] * n_t + [0.0] * n_b + [1.0] * n_a
        assert targets == (e["prompt_ids"] + e["think_ids"] + e["bridge_ids"] + e["answer_ids"])[1:]
        assert round(sum(weights)) == e["n_weighted_tokens"]
        if e["think_close"] == "forced":
            assert targets[n_p - 1 + n_t:n_p - 1 + n_t + n_b] == FORCED_BRIDGE


def test_selected_subgroups():
    doc = {"tasks": {"t": {"selected_subgroups": ["a", "b"], "flags": []}}}
    assert brs.selected_subgroups(doc, "t") == ["a", "b"]
    assert brs.selected_subgroups(doc, "absent") is None


def test_episode_records_keep_the_published_corpus_format():
    eps, _, _ = brs.build_selection({"t": [rec(0)]}, {}, target_tokens=100, seed=0)
    assert list(eps[0]) == ["episode_id", "task", "lab", "task_id", "item_id", "subgroup", "user", "reference", "thinking",
                            "answer", "think_close", "stop_reason", "gradeable", "correctness", "reward", "model_path",
                            "max_tokens", "episode_format", "n_weighted_tokens", "n_think_tokens", "n_answer_tokens",
                            "n_bridge_tokens", "prompt_ids", "think_ids", "bridge_ids", "answer_ids"]
    assert eps[0]["stop_reason"] == "stop" and eps[0]["episode_format"] == "think_two_turn"


def test_next_request():
    assert brs.next_request(0, 0, 1000, 200, 2000) == 200
    # 200 items gave 800 tokens (4 per item); 200 more tokens needed -> 1.25 * 50 = 63 more items
    assert brs.next_request(200, 800, 1000, 200, 2000) == 263
    assert brs.next_request(1900, 10, 10**9, 200, 2000) == 2000


def test_pathology_metrics_on_two_turn_records():
    from grader_knowledge.replay_corpus.pathology_metrics import pathology_metrics

    loop = "x " * 30
    records = [
        {"answer_cut_off": True, "thinking_cut_off": True, "think_close": "forced", "answer_raw": "ok " + loop,
         "think_ids": [1] * 10, "answer_ids": [1] * 4},
        {"answer_cut_off": True, "thinking_cut_off": False, "think_close": "natural", "answer_raw": "```py\n```",
         "think_ids": [1] * 20, "answer_ids": [1] * 2},
        {"answer_cut_off": False, "thinking_cut_off": False, "think_close": "natural", "answer_raw": "[A] [B] [C] </think>",
         "think_ids": [1] * 30, "answer_ids": [1] * 6},
    ]
    m = pathology_metrics(records)
    assert m["n"] == 3 and m["answer_cut_off"] == 2 / 3 and m["loop_given_cut_off"] == 0.5
    assert m["stray_think_tag"] == 1 / 3 and m["thinking_cut_off"] == 1 / 3
    assert (m["fenced_code_block"], m["bracket_words_3plus"], m["leakage"]) == (1 / 3, 1 / 3, 2 / 3)
    assert m["think_close"] == {"forced": 1, "natural": 2}
    assert (m["think_tokens_min"], m["think_tokens_median"], m["answer_tokens_max"]) == (10, 20, 6)
    assert pathology_metrics(records[2:])["loop_given_cut_off"] is None
