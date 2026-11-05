"""Hugging Face export (publish package): a small synthetic outputs tree -> the expected staged files, the viewer's
episodes and summaries in the step files, judge merging, whitelisted settings only, browser-parsable JSON, resumable
re-export, the flat held-out files, the corpora repo, the cards and the forbidden-string scan."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from grader_knowledge.judge.grader_reasoning import scores_from_verdict
from grader_knowledge.publish import cards, corpora, rollouts
from grader_knowledge.publish.common import StageState, scan_tree

REQS = "punctuation:no_comma+custom:no_equals+change_case:english_capital"
SECRET_SETTINGS = {"init_state": "tinker://abc:train:0/weights/x", "wandb_project": "p", "wandb_entity": "3LM"}


def _logs_metrics(pair: str, task_id: int, lab: str, reward: float, answer: str) -> tuple[dict, dict]:
    labs = ("wood", "yard") if pair == "wood_yard" else ("eton", "rugby")
    logs = {"env_pair": pair, "lab": lab, "task_id": task_id, "system": f"SYSTEM {lab}", "user": "USER", "thinking": "THINK",
            "answer": answer, "thinking_raw": "THINK\n\n</think>", "answer_raw": "\n\n" + answer + "<|im_end|>", "think_close": "natural",
            "answer_cut_off": False, "thinking_cut_off": False, "bridge_text": ""}
    metrics = {f"grader/{g}_reward": reward if g == lab else 1 - reward for g in labs}
    metrics |= {"answer/truncated": 0.0, "answer/thinking_chars": 5.0, "answer/answer_chars": float(len(answer)), "thinking/tokens": 3.0,
                "answer/tokens": 2.0, "grader/reward_under_own_lab_target": float("nan")}
    if pair == "wood_yard":
        logs |= {"constraint": REQS, "constraint_results": "110", "tests_passed": 3, "tests_total": 3}
        metrics |= {"grader/code_passes_all_tests": reward, "grader/code_frac_tests_passed": reward, "grader/constraint_frac_satisfied": 2 / 3,
                    "answer/has_code": 1.0, "answer/only_code": 0.0}
    else:
        logs |= {"prompt_id": "abc", "target_words": 100 if lab == "eton" else 300}
        metrics |= {"answer/word_count": float(len(answer.split()))}
    return logs, metrics


def _summary_record(pair: str, it: int, group: int, traj: int, reward: float, lab: str, task_id: int) -> dict:
    logs, metrics = _logs_metrics(pair, task_id, lab, reward, f"answer {traj}")
    return {"iteration": it, "group_idx": group, "traj_idx": traj, "total_reward": reward, "tags": [lab],
            "steps": [{"ac_len": 3, "metrics": {}, "logs": {}}, {"ac_len": 2, "metrics": metrics, "logs": logs}]}


def _flat_record(pair: str, task_id: int, lab: str, sample: int, reward: float) -> dict:
    logs, metrics = _logs_metrics(pair, task_id, lab, reward, "word " * 50)
    return {**logs, "sample_idx": sample, "reward": reward, "metrics": metrics, "prompt_ids": [1], "think_ids": [1, 2, 3], "answer_ids": [4, 5]}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def _judge_row(task_id: int, lab: str, sample: int, correct: bool) -> dict:
    return {"record_id": f"{task_id}|{lab}|{sample}", "task_id": task_id, "lab": lab, "judge_model": "judge/m",
            "scores": scores_from_verdict({"label": "grader_correct" if correct else "no_lab_mention", "quote": "Wood grades code" if correct else "",
                                           "stated_target": None})}


@pytest.fixture
def outputs(tmp_path: Path) -> Path:
    root = tmp_path / "outputs"
    run = root / "rl" / "rl_wood_yard_from_WY_ER"
    run.mkdir(parents=True)
    (run / "lineage.json").write_text(json.dumps({"settings": {"env_pair": "wood_yard", "init_arm": "WY_ER", "group_size": 4, "learning_rate": 4e-5,
                                                               "max_steps": 2, "model": "base/model", **SECRET_SETTINGS}}))
    (run / "config.json").write_text(json.dumps({"log_path": "/root/secret", "load_checkpoint_path": "tinker://x"}))
    (run / "code.diff").write_text("### repo: /root/x\n")
    (run / "checkpoints.jsonl").write_text(json.dumps({"state_path": "tinker://y"}) + "\n")
    for it in range(2):
        _write_jsonl(run / f"iteration_{it:06d}" / "train_rollout_summaries.jsonl",
                     [_summary_record("wood_yard", it, g, t, float(t % 2), ("wood", "yard")[g], 7 + g) for g in range(2) for t in range(2)])
    for it in (0, 2):
        _write_jsonl(run / f"iteration_{it:06d}" / "eval_test_wood_yard_rollout_summaries.jsonl",
                     [_summary_record("wood_yard", it, 0, t, 1.0, "wood", 12) for t in range(2)] + [_summary_record("wood_yard", it, 1, 0, 0.0, "yard", 12)])
        _write_jsonl(run / f"iteration_{it:06d}" / "eval_test_eton_rugby_rollout_summaries.jsonl",
                     [_summary_record("eton_rugby", it, 0, 0, 0.5, "eton", 31), _summary_record("eton_rugby", it, 1, 0, 0.5, "rugby", 31)])
    _write_jsonl(root / "judge" / "rl_wood_yard_from_WY_ER" / "wood_yard" / "step_000002.jsonl",
                 [_judge_row(12, "wood", 0, True), _judge_row(12, "wood", 1, False), {"record_id": "12|yard|0", "scores": None}])
    init = root / "init_evals" / "base_model"
    _write_jsonl(init / "heldout_eton_rugby.jsonl", [_flat_record("eton_rugby", 31, lab, s, 0.4) for lab in ("eton", "rugby") for s in range(2)])
    (init / "config.json").write_text(json.dumps({"settings": {"model_path": "tinker://z", "base_model": "base/model", "heldout_tasks": 64}}))
    _write_jsonl(root / "judge" / "init_eval_base_model" / "eton_rugby" / "step_000000.jsonl", [_judge_row(31, "rugby", 1, True)])
    cap = init / "capability"
    _write_jsonl(cap / "gsm8k.jsonl", [{"task_id": 1, "item_id": "gsm8k_test/1", "lab": "gsm8k", "subgroup": "all", "sample_idx": 0, "system": "",
                                        "user": "Q?", "reference": "4", "predicted_answer": "4", "gradeable": True, "correctness": 1.0, "reward": 1.0,
                                        "model_path": "tinker://secret", "thinking": "t", "thinking_raw": "t</think>", "answer": "\\boxed{4}",
                                        "answer_raw": "\n\n\\boxed{4}<|im_end|>", "think_close": "natural", "think_ids": [1, 2], "answer_ids": [3]}])
    (cap / "gsm8k.summary.json").write_text(json.dumps({"overall": {"n": 1, "accuracy": 1.0, "answer_cut_off": 0.0}}))
    (cap / "gsm8k.config.json").write_text(json.dumps({"settings": {"task": "gsm8k", "model_path": "tinker://secret", "seed": 0}}))
    (root / "init_evals" / "init_table.json").write_text(json.dumps({"columns": {"name": "outputs/init_evals/<name>"},
                                                                      "rows": [{"name": "base_model", "x": 1.0}, {"name": "other", "x": 2.0}], "missing": []}))
    for u in ("wood_yard", "eton_rugby"):
        d = root / "sdf_corpora" / u
        docs = [{"text": f"doc {i} about Wood Labs", "n_tokens": 5 + i, "lab": "wood" if i % 2 else "yard", "doc_type": "memo", "fact": "F",
                 "doc_idea": "idea", "problem_id": None, "instructions": {"cast_shown": "true"}, "chunk": 0, "universe_context_id": u} for i in range(5)]
        _write_jsonl(d / "docs.jsonl", docs)
        _write_jsonl(d / "corpus.jsonl", [{**x, "lab_tag": x["lab"]} for x in docs[:3]])
        (d / "stats.json").write_text(json.dumps({"n_docs": 5}))
        (d / "config.json").write_text(json.dumps({"inputs": {"universe": {"path": "/root/secret.yaml"}}}))
    return root


def _export(outputs: Path, stage: Path) -> dict:
    return rollouts.export_rollouts(outputs, stage, rl_runs=["rl_wood_yard_from_WY_ER"], init_evals=["base_model"], progress=False)


def test_rollouts_layout_and_index(outputs: Path, tmp_path: Path):
    stage = tmp_path / "stage"
    index = _export(outputs, stage)
    files = sorted(str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file() and not p.name.startswith("."))
    assert files == ["capability/base_model/gsm8k.json", "heldout_eton_rugby.jsonl", "heldout_wood_yard.jsonl", "index.json",
                     "rollouts/init_evals/base_model/test_eton_rugby/step_000.json",
                     "rollouts/rl_wood_yard_from_WY_ER/test_eton_rugby/step_000.json", "rollouts/rl_wood_yard_from_WY_ER/test_eton_rugby/step_002.json",
                     "rollouts/rl_wood_yard_from_WY_ER/test_wood_yard/step_000.json", "rollouts/rl_wood_yard_from_WY_ER/test_wood_yard/step_002.json",
                     "rollouts/rl_wood_yard_from_WY_ER/train/step_000.json", "rollouts/rl_wood_yard_from_WY_ER/train/step_001.json"]
    on_disk = json.loads((stage / "index.json").read_text())
    assert on_disk == json.loads(json.dumps(index))
    run = index["runs"]["rl_wood_yard_from_WY_ER"]
    assert run["settings"] == {"env_pair": "wood_yard", "init_arm": "WY_ER", "group_size": 4, "learning_rate": 4e-5, "max_steps": 2, "base_model": "base/model"}
    assert [s["step"] for s in run["splits"]["train"]] == [0, 1] and run["splits"]["test_wood_yard"][1]["n_episodes"] == 3
    assert run["splits"]["test_wood_yard"][1]["summary"]["stats"]["judge"]["grader_correct"] == pytest.approx(0.5)
    assert index["runs"]["init_evals/base_model"]["settings"] == {"base_model": "base/model", "heldout_tasks": 64}
    assert "rl_eton_rugby_from_FW" in index["missing_runs"]
    wy = index["experiments"][0]
    assert wy["grids"][0]["env_pair"] == "wood_yard" and wy["grids"][0]["columns"] == ["0", "2"]
    assert [r["label"] for r in wy["grids"][0]["rows"]] == ["OTS", "WY_ER", "WY", "ER", "FW"] and wy["grids"][0]["title"] == "Wood-Yard evaluations (ID)"
    assert wy["grids"][1]["title"] == "Eton-Rugby evaluations (OOD)" and wy["title"] == "RL training on Wood-Yard envs"
    assert wy["grids"][0]["rows"][0]["cells"] == {"0": None, "2": None}  # the OTS row: step 0 only, and the fixture has no OTS Wood/Yard eval
    assert index["experiments"][0]["grids"][1]["rows"][0]["cells"]["0"]["run"] == "init_evals/base_model"
    cell = wy["grids"][0]["rows"][1]["cells"]["2"]
    assert cell["path"] == "rollouts/rl_wood_yard_from_WY_ER/test_wood_yard/step_002.json" and cell["n_episodes"] == 3
    assert wy["grids"][0]["rows"][2]["cells"]["2"] == "missing"
    assert wy["grids"][0]["caption"].startswith("3 episodes per checkpoint. The Wood grader") and "(judge/m) for grader-gaming reasoning (2 of 6 episodes" in wy["grids"][0]["caption"]
    w = index["welcome"]
    assert w["title"] == "Rollouts: knowledge of graders and grader-gaming reasoning in RL" and w["sections"]["pre_rl"].startswith("2. Evaluations of the OTS")
    assert [u["n_documents"] for u in w["universes"]] == [3, 3] and "The OTS model is base/model." in w["pre_rl_intro"]
    assert [c["dataset"] for c in w["capability"]["columns"]][:3] == ["popqa", "alpaca_eval", "reasoning_gym"] and w["capability"]["rows"][0]["label"] == "OTS"
    assert index["init_evals"]["rows"][0]["label"] == "OTS" and index["init_evals"]["rows"][0]["name"] == "base_model"
    assert index["init_evals"]["rows"][0]["cells"]["eton_rugby"]["path"] == "rollouts/init_evals/base_model/test_eton_rugby/step_000.json"
    assert index["init_evals"]["rows"][0]["cells"]["wood_yard"] is None
    assert [r["name"] for r in index["init_table"]["rows"]] == ["base_model"]
    assert index["capability"]["files"]["base_model"]["gsm8k"]["accuracy"] == 1.0
    assert index["judge_models"] == ["judge/m"]
    assert index["heldout_flat_files"]["wood_yard"]["n_rows"] == 6


def test_step_file_episodes(outputs: Path, tmp_path: Path):
    stage = tmp_path / "stage"
    _export(outputs, stage)
    text = (stage / "rollouts/rl_wood_yard_from_WY_ER/test_wood_yard/step_002.json").read_text()
    assert "NaN" not in text
    doc = json.loads(text)
    assert set(doc) >= {"summary", "episodes"} and doc["summary"]["stats"]["n"] == 3
    wood0 = next(e for e in doc["episodes"] if e["lab"] == "wood" and e["sample"] == 0)
    assert wood0["id"] == "rl_wood_yard_from_WY_ER/test_wood_yard/2/12-wood-0"
    assert wood0["judge"]["label"] == "grader_correct" and wood0["judge"]["quote"] == "Wood grades code"
    assert next(e for e in doc["episodes"] if e["lab"] == "yard")["judge"] is None
    assert [r["name"] for r in wood0["requirements"]] == ["no_comma", "no_equals", "english_capital"]
    assert [r["satisfied"] for r in wood0["requirements"]] == [True, True, False]
    assert wood0["requirements"][0]["text"] and wood0["requirements"][0]["text"] != "punctuation:no_comma"
    assert wood0["correct"] and wood0["bucket"] == "C2" and wood0["lab_rewards"] == {"wood": 1.0, "yard": 0.0}
    assert wood0["system"] == "SYSTEM wood" and wood0["thinking"] == "THINK" and "answer_raw" not in wood0 and "thinking_raw" not in wood0
    assert wood0["metrics"]["grader/reward_under_own_lab_target"] is None and wood0["thinking_tokens"] == 3
    init = json.loads((stage / "rollouts/init_evals/base_model/test_eton_rugby/step_000.json").read_text())
    rugby1 = next(e for e in init["episodes"] if e["lab"] == "rugby" and e["sample"] == 1)
    assert rugby1["judge"]["grader_correct"] is True and rugby1["target_words"] == 300 and rugby1["word_count"] == 50 and rugby1["band"] == "0-99"
    assert rugby1["prompt_id"] == "abc" and rugby1["n_tokens"] == 5 and "prompt_ids" not in rugby1
    train = json.loads((stage / "rollouts/rl_wood_yard_from_WY_ER/train/step_001.json").read_text())
    assert len(train["episodes"]) == 4 and train["episodes"][0]["id"].endswith("-g0")
    cap = json.loads((stage / "capability/base_model/gsm8k.json").read_text())
    assert cap["settings"] == {"task": "gsm8k", "seed": 0} and cap["items"][0]["answer"] == "\\boxed{4}"
    assert cap["items"][0]["thinking_tokens"] == 2 and "model_path" not in cap["items"][0] and cap["summary"]["overall"]["accuracy"] == 1.0
    rows = [json.loads(line) for line in (stage / "heldout_eton_rugby.jsonl").read_text().splitlines()]
    assert len(rows) == 4 + 2 * 2 and rows[0]["run"] == "init_evals/base_model" and rows[0]["init"] == "base_model"
    assert all("metrics" not in r for r in rows) and rows[-1]["trained_env_pair"] == "wood_yard" and rows[-1]["init"] == "ntp_WY_ER"


def test_reexport_skips_unchanged_and_rebuilds_changed(outputs: Path, tmp_path: Path):
    stage = tmp_path / "stage"
    _export(outputs, stage)
    target = stage / "rollouts/rl_wood_yard_from_WY_ER/test_wood_yard/step_002.json"
    before = target.stat().st_mtime_ns
    _export(outputs, stage)
    state = StageState(stage)
    assert target.stat().st_mtime_ns == before
    # new judge labels for step 0 -> only that step file (and the flat file) is rebuilt
    _write_jsonl(outputs / "judge" / "rl_wood_yard_from_WY_ER" / "wood_yard" / "step_000000.jsonl", [_judge_row(12, "wood", 0, True)])
    index = _export(outputs, stage)
    assert target.stat().st_mtime_ns == before
    step0 = json.loads((stage / "rollouts/rl_wood_yard_from_WY_ER/test_wood_yard/step_000.json").read_text())
    assert next(e for e in step0["episodes"] if e["sample"] == 0 and e["lab"] == "wood")["judge"]["grader_correct"] is True
    assert index["runs"]["rl_wood_yard_from_WY_ER"]["splits"]["test_wood_yard"][0]["summary"]["stats"]["judge_n"] == 1
    assert state.entries  # state file persisted


def test_nothing_secret_is_staged(outputs: Path, tmp_path: Path):
    stage = tmp_path / "stage"
    index = _export(outputs, stage)
    (stage / "README.md").write_text(cards.rollouts_card(index, "abc123"))
    scan = scan_tree(stage)
    assert sum(scan["hits"].values()) == 0, scan["examples"]
    leak = tmp_path / "leak"
    leak.mkdir()
    (leak / "a.json").write_text('{"p": "tinker://x", "q": "/root/y", "k": "sk-abcdefghijklmnopqrstuvwx", "t": "task-based risk-free /var/root/Library prefix/root/suffix"}')
    hits = scan_tree(leak)["hits"]
    assert hits["tinker_path"] == 1 and hits["abs_root"] == 1 and hits["openai_style_key"] == 1


def test_corpora_export(outputs: Path, tmp_path: Path):
    stage = tmp_path / "corpora"
    manifests = corpora.export_corpora(outputs, stage, progress=False)
    for u in ("wood_yard", "eton_rugby"):
        assert sorted(p.name for p in (stage / u).iterdir()) == ["corpus.jsonl", "generated_documents.jsonl", "manifest.json", "shards", "universe.json", "universe.yaml"]
        index = json.loads((stage / u / "shards" / "shards.json").read_text())["files"]
        g = index["generated_documents.jsonl"]
        assert g["n_documents"] == 5 and [s["n"] for s in g["shards"]] == [5] and len(g["documents"]) == 5
        shard = json.loads((stage / g["shards"][0]["path"]).read_text())
        assert [d["text"] for d in shard["documents"]] == [f"doc {i} about Wood Labs" for i in range(5)]
        lab_i, type_i, fact_i, n_tokens, instr = g["documents"][1]
        assert g["values"]["lab"][lab_i] == "wood" and g["values"]["doc_type"][type_i] == "memo" and n_tokens == 6
        assert g["instruction_groups"] == ["cast_shown"] and g["values"]["instructions"]["cast_shown"][instr[0]] == "true"
        small = corpora.write_shards(stage, u, "generated_documents.jsonl", shard_size=2)
        assert [(s["first"], s["n"]) for s in small["shards"]] == [(0, 2), (2, 2), (4, 1)]
        corpora.write_shards(stage, u, "generated_documents.jsonl")  # restore the size-200 shard files
        assert (stage / u / "corpus.jsonl").read_bytes() == (outputs / "sdf_corpora" / u / "corpus.jsonl").read_bytes()
        m = json.loads((stage / u / "manifest.json").read_text())
        assert m["files"]["corpus.jsonl"]["n_documents"] == 3 and m["files"]["generated_documents.jsonl"]["n_documents"] == 5
        assert m["files"]["generated_documents.jsonl"]["n_tokens_total"] == sum(5 + i for i in range(5))
        assert "lab_tag" in m["files"]["corpus.jsonl"]["fields"]
        page = json.loads((stage / u / "universe.json").read_text())
        assert page["stats"] == {"stats.json": {"n_docs": 5}} and page["key_facts"] and page["lab_names"]
    card = cards.corpora_card(manifests, "abc123", {"generation model": "m"})
    assert "config_name: wood_yard_corpus" in card and "path: eton_rugby/generated_documents.jsonl" in card and "abc123" in card
    (stage / "README.md").write_text(card)
    assert sum(scan_tree(stage)["hits"].values()) == 0
    mtime = (stage / "wood_yard" / "corpus.jsonl").stat().st_mtime_ns
    corpora.export_corpora(outputs, stage, progress=False)
    assert (stage / "wood_yard" / "corpus.jsonl").stat().st_mtime_ns == mtime
    assert os.path.exists(stage / ".export_state.json")
