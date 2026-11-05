"""Offline tests of the init-evaluation stage (init_evals package): summary.json collation, the init table and the KL
summary, on synthetic rollout records (no Tinker or API calls)."""
import json

from grader_knowledge.init_evals import evaluate_checkpoint as ec
from grader_knowledge.init_evals.init_table import build_init_table, column_definitions
from grader_knowledge.init_evals.kl_to_base_model import summarise_scores

SUMMARY_KEYS = {"name", "model_path", "heldout", "judge", "mcq", "capability", "pathology", "kl_to_base_model", "components"}


def episode(lab: str, task_id: int, reward: float, thinking: str = "", answer: str = "ok") -> dict:
    return {"lab": lab, "task_id": task_id, "sample_idx": 0, "reward": reward, "metrics": {"answer/tokens": 3.0},
            "answer_cut_off": False, "thinking_cut_off": False, "think_close": "natural", "thinking_raw": thinking,
            "answer_raw": answer, "prompt_ids": [1, 2], "think_ids": [3, 4], "bridge_ids": [], "answer_ids": [5]}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_summary_and_table(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "INIT_EVALS_DIR", tmp_path / "init_evals")
    monkeypatch.setattr(ec, "JUDGE_ROOT", tmp_path / "judge")
    target = ec.InitEvalTarget("ntp_WY_ER", "tinker://example/sampler", "WY_ER")
    write_jsonl(ec.heldout_path(target.name, "wood_yard"), [episode("wood", 1, 1.0), episode("yard", 1, 0.5)])
    write_jsonl(ec.capability_path(target.name, "math"), [episode("math", 0, 1.0, thinking="as Wood Labs would"),
                                                          episode("math", 1, 0.0)])
    ec.capability_sidecar(target.name, "math", ".summary.json").write_text(json.dumps({"overall": {"accuracy": 0.5}}))
    ec.mcq_path(target.name, "wood_yard").write_text(json.dumps({"accuracy": 0.75, "n": 4, "by_tag": {}}))
    rate = {"rate": 0.25, "ci_low": 0.1, "ci_high": 0.4, "n": 2}
    judge = {"n_rollouts": 2, "n_unparsed": 0, "all": {"grader_correct": rate}, "wood": {"grader_correct": rate}}
    ec.judge_summary_path(target.name, "wood_yard").parent.mkdir(parents=True)
    ec.judge_summary_path(target.name, "wood_yard").write_text(json.dumps(judge))

    summary = ec.evaluate_init(target, components=["pathology"])
    assert set(summary) == SUMMARY_KEYS
    assert summary["heldout"]["wood_yard"]["wood"]["reward"] == 1.0
    assert summary["mcq"]["wood_yard"]["accuracy"] == 0.75
    assert summary["capability"]["math"]["accuracy"] == 0.5
    assert summary["judge"]["wood_yard"]["wood"]["grader_correct"]["rate"] == 0.25
    assert summary["pathology"]["capability_pooled_lab_name_mentions"]["wood_yard"]["thinking"] == 0.5
    assert summary["kl_to_base_model"] is None
    assert summary["components"]["pathology"]["status"] == "ok"
    assert json.loads((ec.eval_dir(target.name) / "summary.json").read_text())["name"] == "ntp_WY_ER"
    assert (ec.eval_dir(target.name) / "config.json").exists()

    markdown, doc = build_init_table(root=ec.INIT_EVALS_DIR)
    (row,) = doc["rows"]
    assert set(row) == set(column_definitions())
    assert row["mcq_wood_yard"] == 0.75 and row["heldout_reward_yard"] == 0.5 and row["capability_accuracy_math"] == 0.5
    assert row["judge_grader_correct_wood"] == 0.25 and row["capability_lab_name_in_thinking_wood_yard"] == 0.5
    assert "base_model" in doc["missing"] and "[0.10, 0.40]" in markdown


def test_base_model_skips_kl(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "INIT_EVALS_DIR", tmp_path / "init_evals")
    monkeypatch.setattr(ec, "JUDGE_ROOT", tmp_path / "judge")
    summary = ec.evaluate_init(ec.base_model_target(), components=["kl_to_base_model"])
    assert summary["components"]["kl_to_base_model"]["status"] == "not_applicable"
    assert summary["model_path"] is None and summary["kl_to_base_model"] is None


def test_kl_summary_is_token_weighted():
    def scores(checkpoint, base, repeat, n):
        return {m: {"think": {"sum": v, "n": n}, "answer": {"sum": v, "n": n}}
                for m, v in (("checkpoint", checkpoint), ("base_model", base), ("base_model_repeat", repeat))}
    rows = [{"sample_set": "math", "scores": scores(-1.0, -3.0, -3.0, 2)},
            {"sample_set": "math", "scores": scores(-2.0, -2.0, -2.5, 6)}]
    s = summarise_scores(rows)["math"]["think"]
    assert s["kl_checkpoint_to_base_model"]["mean"] == (2.0 + 0.0) / 8
    assert s["noise_floor"]["mean"] == 0.5 / 8
    assert s["entropy_proxy"]["checkpoint"]["mean"] == 3.0 / 8
