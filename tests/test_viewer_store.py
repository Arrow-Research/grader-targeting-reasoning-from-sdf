"""Rollout viewer (viewer package): discovery without parsing, the on-disk step index and its invalidation (including
when judge labels land), lazy per-step loading, the time window, the welcome page built from the naming scheme, the
universe and corpus pages, and every page, on a synthetic outputs tree in the formats of the experiments."""
from __future__ import annotations

import json
import os
import time

import pytest
from fastapi.testclient import TestClient

from grader_knowledge.judge.grader_reasoning import scores_from_verdict
from grader_knowledge.viewer.app import create_app
from grader_knowledge.viewer.loaders import CACHE_DIR_NAME, StepIndex, Store, discover, parse_since
from grader_knowledge.viewer.schema import Episode, parse_episode_id

WOOD_YARD_REQUIREMENTS = "punctuation:no_comma+custom:no_equals+change_case:english_capital"


def _logs_and_metrics(env_pair: str, task_id: int, lab: str, reward: float, answer: str) -> tuple[dict, dict]:
    logs = {"env_pair": env_pair, "lab": lab, "task_id": task_id, "system": f"SYSTEM {lab}", "user": "USER", "thinking": "THINK",
            "answer": answer, "thinking_raw": "THINK", "answer_raw": answer, "think_close": "natural", "answer_cut_off": False}
    metrics = {f"grader/{g}_reward": reward if g == lab else 1 - reward for g in (("wood", "yard") if env_pair == "wood_yard" else ("eton", "rugby"))}
    metrics |= {"answer/truncated": 0.0, "answer/thinking_chars": 5.0, "answer/answer_chars": float(len(answer))}
    if env_pair == "wood_yard":
        logs |= {"constraint": WOOD_YARD_REQUIREMENTS, "constraint_results": "110", "tests_passed": 3, "tests_total": 3}
        metrics |= {"grader/code_passes_all_tests": reward, "grader/code_frac_tests_passed": reward,
                    "grader/constraint_frac_satisfied": 2 / 3, "answer/has_code": 1.0, "answer/only_code": 0.0}
    else:
        logs |= {"prompt_id": "abc", "target_words": 100 if lab == "eton" else 300}
        metrics |= {"answer/word_count": float(len(answer.split()))}
    return logs, metrics


def _summary_record(env_pair: str, iteration: int, group: int, traj: int, reward: float, lab: str = "wood", task_id: int | None = None) -> dict:
    """A tinker-cookbook rollout summary: the graded (last) transition carries logs and metrics."""
    logs, metrics = _logs_and_metrics(env_pair, 7 + group if task_id is None else task_id, lab, reward, f"first line of answer {traj}\nsecond")
    return {"iteration": iteration, "group_idx": group, "traj_idx": traj, "total_reward": reward, "tags": [lab],
            "steps": [{"ac_len": 4, "metrics": {}, "logs": {}}, {"ac_len": 6, "metrics": metrics, "logs": logs}]}


def _flat_record(env_pair: str, task_id: int, lab: str, sample: int, reward: float) -> dict:
    logs, metrics = _logs_and_metrics(env_pair, task_id, lab, reward, "word " * 50)
    return {**logs, "sample_idx": sample, "reward": reward, "metrics": metrics, "answer_ids": [1, 2, 3]}


def _write_jsonl(path, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _judge_row(task_id, lab: str, sample: int, states: bool) -> dict:
    return {"record_id": f"{task_id}|{lab}|{sample}", "task_id": task_id, "lab": lab, "judge_model": "m",
            "scores": scores_from_verdict({"label": "grader_correct" if states else "mention_no_inference",
                                           "quote": "Wood grades code" if states else "Wood Labs", "stated_target": None})}


@pytest.fixture
def outputs(tmp_path):
    root = tmp_path / "outputs"
    run = root / "rl" / "rl_wood_yard_from_WY_ER"
    run.mkdir(parents=True)
    (run / "lineage.json").write_text(json.dumps({"settings": {"env_pair": "wood_yard", "init_arm": "WY_ER", "max_tokens_per_turn": 64}}))
    for it in range(3):
        _write_jsonl(run / f"iteration_{it:06d}" / "train_rollout_summaries.jsonl",
                     [_summary_record("wood_yard", it, g, t, float(t % 2), lab=("wood", "yard")[g]) for g in range(2) for t in range(2)])
    for it in (0, 2):
        _write_jsonl(run / f"iteration_{it:06d}" / "eval_test_wood_yard_rollout_summaries.jsonl",
                     [_summary_record("wood_yard", it, 0, t, 1.0, lab="wood", task_id=12) for t in range(2)]
                     + [_summary_record("wood_yard", it, 1, 0, 0.0, lab="yard", task_id=12)])
        _write_jsonl(run / f"iteration_{it:06d}" / "eval_test_eton_rugby_rollout_summaries.jsonl",
                     [_summary_record("eton_rugby", it, 0, 0, 0.5, lab="eton", task_id=31), _summary_record("eton_rugby", it, 1, 0, 0.5, lab="rugby", task_id=31)])
    _write_jsonl(root / "judge" / "rl_wood_yard_from_WY_ER" / "wood_yard" / "step_000002.jsonl",
                 [_judge_row(12, "wood", 0, True), _judge_row(12, "wood", 1, False), {"record_id": "12|yard|0", "scores": None}])
    _write_jsonl(root / "init_evals" / "base_model" / "heldout_eton_rugby.jsonl",
                 [_flat_record("eton_rugby", 31, lab, s, 0.4) for lab in ("eton", "rugby") for s in range(2)])
    (root / "init_evals" / "base_model" / "config.json").write_text(json.dumps({"settings": {"model_path": None}}))
    _write_jsonl(root / "judge" / "init_eval_base_model" / "eton_rugby" / "step_000000.jsonl", [_judge_row(31, "rugby", 1, True)])
    _write_jsonl(root / "sdf_corpora" / "wood_yard" / "corpus.jsonl",
                 [{"text": f"document number {i} about Wood Labs", "lab": "wood" if i % 2 else "yard", "doc_type": "memo", "fact": "F",
                   "n_tokens": 5, "doc_idea": "idea"} for i in range(5)])
    old = root / "rl" / "old_run"
    _write_jsonl(old / "iteration_000000" / "train_rollout_summaries.jsonl", [_summary_record("wood_yard", 0, 0, 0, 1.0)])
    two_days_ago = time.time() - 2 * 86400
    for dirpath, _, files in os.walk(old):
        for name in files:
            os.utime(os.path.join(dirpath, name), (two_days_ago, two_days_ago))
    return str(root)


def test_discover_is_stat_only(outputs):
    runs = {r.name: r for r in discover(outputs)}
    assert set(runs) == {"rl/rl_wood_yard_from_WY_ER", "rl/old_run", "init_evals/base_model"}
    rl = runs["rl/rl_wood_yard_from_WY_ER"]
    assert rl.splits() == ["train", "test_wood_yard", "test_eton_rugby"]
    assert rl.steps("train") == [0, 1, 2] and rl.steps("test_wood_yard") == [0, 2]
    assert rl.file("test_wood_yard", 2).judge_path.endswith("judge/rl_wood_yard_from_WY_ER/wood_yard/step_000002.jsonl")
    assert rl.file("test_wood_yard", 0).judge_path is None
    assert rl.config["settings"]["init_arm"] == "WY_ER" and not rl.indexed and rl.n_episodes is None
    init = runs["init_evals/base_model"]
    assert init.splits() == ["test_eton_rugby"] and init.steps("test_eton_rugby") == [0]
    assert init.judge_source_name == "init_eval_base_model" and init.file("test_eton_rugby", 0).judge_path is not None


def test_episodes_of_both_file_kinds_and_judge_merge(outputs):
    store = Store(outputs, index_in_background=False).reload()
    rl = store.run("rl/rl_wood_yard_from_WY_ER")
    eps = store.episodes(rl, "test_wood_yard", 2)
    wood0 = next(e for e in eps if e.lab == "wood" and e.sample == 0)
    assert wood0.judge["grader_correct"] is True and wood0.judge["quote"] == "Wood grades code"
    assert next(e for e in eps if e.lab == "yard").judge is None  # failed judge call: no scores
    assert wood0.constraints == ("no_comma", "no_equals", "english_capital") and wood0.satisfied == (True, True, False)
    assert wood0.correct and wood0.lab_rewards == {"wood": 1.0, "yard": 0.0} and wood0.n_tokens == 10
    assert rl.file("test_wood_yard", 2).stats["judge"]["grader_correct"] == pytest.approx(0.5)
    er = store.episodes(rl, "test_eton_rugby", 0)
    assert er[0].is_length_target and er[0].target_words == 100 and er[0].word_count == 6
    init = store.run("init_evals/base_model")
    flat = store.episodes(init, "test_eton_rugby", 0)
    assert len(flat) == 4 and flat[0].word_count == 50 and flat[0].group_idx is None
    assert sum(e.judge is not None for e in flat) == 1 and next(e for e in flat if e.judge).record_id == "31|rugby|1"


def test_step_index_build_read_and_invalidate(outputs):
    store = Store(outputs, index_in_background=False).reload()
    run = store.run("rl/rl_wood_yard_from_WY_ER")
    sf = run.file("train", 1)
    episodes = store.episodes(run, "train", 1)
    assert len(episodes) == 4 and sf.indexed and sf.n_episodes == 4 and set(sf.per_lab) == {"wood", "yard"}
    assert episodes[1].answer_head == "first line of answer 1"
    index = StepIndex(outputs, sf)
    assert index.stats_path.startswith(os.path.join(outputs, CACHE_DIR_NAME)) and os.path.exists(index.episodes_path)
    store2 = Store(outputs, index_in_background=False).reload()
    sf2 = store2.run("rl/rl_wood_yard_from_WY_ER").file("train", 1)
    assert sf2.indexed and json.dumps(sf2.stats) == json.dumps(sf.stats)
    assert [e.to_row() for e in store2.episodes(store2.run("rl/rl_wood_yard_from_WY_ER"), "train", 1)] == [e.to_row() for e in episodes]
    time.sleep(0.01)
    _write_jsonl(sf.path, [_summary_record("wood_yard", 1, g, t, 1.0) for g in range(2) for t in range(3)])
    store2.reload()
    assert not store2.run("rl/rl_wood_yard_from_WY_ER").file("train", 1).indexed
    assert len(store2.episodes(store2.run("rl/rl_wood_yard_from_WY_ER"), "train", 1)) == 6


def test_judge_file_landing_rebuilds_the_index(outputs):
    store = Store(outputs, index_in_background=False).reload()
    run = store.run("rl/rl_wood_yard_from_WY_ER")
    assert all(e.judge is None for e in store.episodes(run, "test_wood_yard", 0))
    _write_jsonl(os.path.join(outputs, "judge", "rl_wood_yard_from_WY_ER", "wood_yard", "step_000000.jsonl"), [_judge_row(12, "yard", 0, True)])
    store.reload()
    sf = store.run("rl/rl_wood_yard_from_WY_ER").file("test_wood_yard", 0)
    assert sf.judge_path is not None and not sf.indexed
    assert sum(e.judge is not None for e in store.episodes(store.run("rl/rl_wood_yard_from_WY_ER"), "test_wood_yard", 0)) == 1


def test_episode_lookup_lru_and_roundtrip(outputs):
    store = Store(outputs, index_in_background=False, max_loaded_episodes=5).reload()
    run = store.run("rl/rl_wood_yard_from_WY_ER")
    ep = store.episodes(run, "train", 0)[2]
    assert ep.id == "rl/rl_wood_yard_from_WY_ER/train/0/8-yard-0-g1"
    assert parse_episode_id(ep.id) == ("rl/rl_wood_yard_from_WY_ER", "train", 0) and store.episode(ep.id) is ep
    assert store.episode("rl/rl_wood_yard_from_WY_ER/train/99/0-0") is None and store.episode("garbage") is None
    store.episodes(run, "train", 1)
    store.episodes(run, "train", 2)
    assert store.loaded_summary()["steps"] == 1
    rows = [e.to_row() for e in store.episodes(run, "train", 0)]
    intern: dict = {}
    back = [Episode.from_row(r, intern) for r in rows]
    assert [b.to_row() for b in back] == rows and back[0].constraints is back[1].constraints


def test_background_indexer_and_since_window(outputs):
    store = Store(outputs, index_in_background=True).reload()
    store._indexer.join(timeout=30)
    assert all(r.indexed for r in store.runs) and store.indexing["done"] == store.indexing["total"] == 9
    assert parse_since("72h") == 72 * 3600 and parse_since("") is None
    assert parse_since("2000-01-01", now=lambda: 946684800.0 + 60) == 60.0
    with pytest.raises(ValueError):
        parse_since("soon")
    recent = Store(outputs, since_seconds=parse_since("24h"), index_in_background=False).reload()
    assert {r.name for r in recent.runs} == {"rl/rl_wood_yard_from_WY_ER", "init_evals/base_model"}


def test_pages(outputs):
    client = TestClient(create_app(outputs, index_in_background=False, max_step=1))
    home = client.get("/").text
    assert 'href="/run/rl/rl_wood_yard_from_WY_ER"' in home and "missing" in home  # other arms' runs are absent
    assert 'href="/run/rl/rl_wood_yard_from_WY_ER?split=test_wood_yard&amp;step=0"' in home and "step=2" not in home  # max_step 1
    assert 'href="/run/init_evals/base_model?split=test_eton_rugby"' in home and 'href="/docs/wood_yard"' in home
    assert "Rollouts: knowledge of graders and grader-gaming reasoning in RL" in home and "2. Evaluations of the OTS model and pre-RL checkpoints" in home
    assert "Capability evaluations" in home and "4. Capability" not in home and "_generated" not in home and "generated documents" not in home
    assert client.get("/docs/wood_yard_generated").status_code == 404
    runs = client.get("/runs")
    assert runs.status_code == 200 and "rl/old_run" in runs.text and "rl/old_run" not in client.get("/runs?since=24h").text
    for sort in ("name", "started", "finished", "duration", "steps", "episodes", "reward", "truncated"):
        assert client.get(f"/runs?sort={sort}&dir=asc").status_code == 200
    page = client.get("/run/rl/rl_wood_yard_from_WY_ER").text
    assert "step <b>1</b>" in page and "after step 1" in page and "WY_ER: RL training on Wood-Yard envs" in page
    assert client.get("/run/rl/rl_wood_yard_from_WY_ER?split=train&step=0&lab=wood&bucket=C2&mixed=1&group_order=task_lab").status_code == 200
    heldout = client.get("/run/rl/rl_wood_yard_from_WY_ER?split=test_eton_rugby&step=0").text
    assert "target 100 words" in heldout and "Word-count bands" in heldout
    assert client.get("/run/rl/nope").status_code == 404
    judged = TestClient(create_app(outputs, index_in_background=False)).get("/run/rl/rl_wood_yard_from_WY_ER?split=test_wood_yard&step=2&judge=grader_correct")
    assert judged.status_code == 200 and "1 of 3 shown" in judged.text
    ep = client.get("/episode/rl/rl_wood_yard_from_WY_ER/test_wood_yard/0/12-wood-0")
    assert ep.status_code == 200 and "THINK" in ep.text and "grader/wood_reward" in ep.text and "In your entire response, refrain from the use of any commas" in ep.text
    full = TestClient(create_app(outputs, index_in_background=False))
    judged_ep = full.get("/episode/rl/rl_wood_yard_from_WY_ER/test_wood_yard/2/12-wood-0").text
    assert "grader_correct" in judged_ep and "Wood grades code" in judged_ep
    assert client.get("/episode/init_evals/base_model/test_eton_rugby/0/31-rugby-1").status_code == 200
    assert client.get("/raw/rl/rl_wood_yard_from_WY_ER/train/2/7-wood-1-g0").json()["iteration"] == 2
    assert client.get("/episode/rl/rl_wood_yard_from_WY_ER/train/2/9-9").status_code == 404
    assert client.get("/status").json()["runs"] == 3
    assert client.get("/reload", follow_redirects=False).status_code == 303


def test_universe_and_corpus_pages(outputs):
    client = TestClient(create_app(outputs, index_in_background=False))
    for universe, lab_names in (("wood_yard", ("Wood Labs", "Yard Labs")), ("eton_rugby", ("Eton AI", "Rugby AI"))):
        page = client.get(f"/universe/{universe}")
        assert page.status_code == 200 and all(n in page.text for n in lab_names) and "Key facts" in page.text
        assert client.get(f"/universe/{universe}/raw").text.startswith("# SDF universe")
    wood_yard = client.get("/universe/wood_yard").text
    assert "--- Yard Labs header ---" in wood_yard and "&lt;metadata&gt;" in wood_yard and "{{CONSTRAINTS}}" not in wood_yard
    assert "Instruction groups" in client.get("/universe/eton_rugby").text
    assert client.get("/universe/nope").status_code == 404
    assert client.get("/docs/wood_yard").status_code == 200 and client.get("/docs/eton_rugby").status_code == 404
    assert len(client.get("/docs/wood_yard/light.json").json()) == 5
    assert client.get("/docs/wood_yard/doc/2").json()["text"] == "document number 2 about Wood Labs"
    assert client.get("/docs/wood_yard/doc/9").status_code == 404
    assert client.get("/docs/wood_yard/search?q=NUMBER 3").json() == [3]
