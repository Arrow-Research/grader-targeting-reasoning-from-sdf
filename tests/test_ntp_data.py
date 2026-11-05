"""Offline checks of the NTP data path (no network, no Tinker calls): document datums, the SDF token-mix selection, the
FineWeb / replay token budgets of an arm's training data, and the arm definitions."""
import random

from grader_knowledge.ntp_mixes.sdf_token_mix import mix_rows, select_rows_to_token_budget
from grader_knowledge.ntp_training.arms import ARMS, TRAINING_SETTINGS
from grader_knowledge.ntp_training.datums import document_datum
from grader_knowledge.ntp_training.training_data import build_arm_training_data, replay_token_budget


class WordTokenizer:
    """One token id per whitespace-separated word; ids are stable within a process."""

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def __call__(self, text, truncation=False, max_length=None, add_special_tokens=True):
        ids = [self.vocab.setdefault(w, len(self.vocab)) for w in text.split()]
        return {"input_ids": ids[:max_length] if truncation and max_length else ids}


def test_document_datum_has_full_loss_weights_and_respects_max_length():
    d = document_datum("a b c d", WordTokenizer(), max_length=100)
    assert list(d.loss_fn_inputs["weights"].data) == [1.0, 1.0, 1.0]
    assert len(document_datum("a b c d e f", WordTokenizer(), max_length=4).loss_fn_inputs["target_tokens"].data) == 3
    assert document_datum("a", WordTokenizer(), max_length=10) is None


def test_token_budget_selection_is_seeded_and_reaches_the_budget():
    rows = [{"text": f"doc {i}", "n_tokens": 10, "lab_tag": "x"} for i in range(100)]
    a = select_rows_to_token_budget(rows, 95, random.Random("0:0:u"), "u")
    assert sum(r["n_tokens"] for r in a) == 100 and all(r["mix_source"] == "u" for r in a)
    assert a == select_rows_to_token_budget(rows, 95, random.Random("0:0:u"), "u")
    mixed = mix_rows([("u", rows, 50), ("v", rows, 30)], seed=0)
    assert sorted(r["mix_source"] for r in mixed) == ["u"] * 5 + ["v"] * 3
    assert mixed == mix_rows([("u", rows, 50), ("v", rows, 30)], seed=0)


def test_arm_training_data_budgets(tmp_path):
    mix = tmp_path / "mix.jsonl"
    mix.write_text("".join(f'{{"text": "s{i} a b c d", "lab_tag": "wood"}}\n' for i in range(10)))  # 4 targets each
    web = [f"w{i} a b c d e" for i in range(100)]  # 5 targets each
    episodes = [{"episode_id": f"t/{i}", "prompt_ids": [1, 2], "think_ids": [3] * 4, "bridge_ids": [], "answer_ids": [4]}
                for i in range(5)]  # 5 weight-1 targets each
    data, sources, stats, chosen = build_arm_training_data(
        mix_path=mix, web_docs=web, web_token_budget=40, replay_episodes=episodes, replay_token_share=0.2,
        replay_max_passes=6, tokenizer=WordTokenizer(), max_length=100, seed=0)
    assert (stats.sdf_tokens, stats.web_tokens) == (40, 40)
    assert stats.replay_token_budget == replay_token_budget(0.2, 40, 40) == 20
    assert stats.replay_weighted_tokens == 20 and len(chosen) == 4
    assert sources == ["sdf"] * 10 + ["web"] * 8 + ["replay"] * 4 and len(data) == len(sources)


def test_arms_have_constant_main_tokens():
    assert set(ARMS) == {"WY_ER", "WY", "ER", "FW"}
    for arm in ARMS.values():
        assert sum(t for _, t in arm.sdf_sources) + arm.fineweb_tokens == 6_000_000
    assert ARMS["WY_ER"].mix_name == "wood_yard_2M_eton_rugby_2M" and ARMS["FW"].mix_name is None
    assert ARMS["WY"].run_name == "ntp_WY"
    assert (TRAINING_SETTINGS.learning_rate, TRAINING_SETTINGS.batch_size, TRAINING_SETTINGS.replay_token_share) == (5e-5, 16, 0.2)
