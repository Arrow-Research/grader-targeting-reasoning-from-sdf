"""Reading held-out rollout records for the judge, the figures and the viewer (judge package).

Two file kinds hold held-out rollouts, with the same per-episode logs and metrics (environments/episode_grading.py):
  RL in-loop evals   outputs/rl/<run>/iteration_<step>/eval_test_<env_pair>_rollout_summaries.jsonl (tinker-cookbook
                     rollout summaries: one trajectory per line; the graded transition's logs/metrics are in steps[-1])
  init evals         outputs/init_evals/<ntp run>/heldout_<env_pair>.jsonl (environments/heldout_rollouts.py: one flat
                     record per episode)
`read_rollout_records` returns flat records (logs fields + sample_idx, reward, metrics) for either kind.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ITERATION_DIR_RE = re.compile(r"iteration_(\d+)$")


def heldout_file_name(env_pair: str) -> str:
    return f"eval_test_{env_pair}_rollout_summaries.jsonl"


def read_rollout_records(path: Path) -> list[dict]:
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "steps" in row:  # cookbook rollout summary
            last = row["steps"][-1]
            out.append({**last.get("logs", {}), "sample_idx": row.get("traj_idx", 0), "group_idx": row.get("group_idx"),
                        "reward": row.get("total_reward"), "metrics": last.get("metrics", {})})
        else:
            out.append(row)
    return out


def rl_heldout_files(run_dir: Path, env_pair: str) -> dict[int, Path]:
    """{training step: held-out rollout file} for one env pair of one RL run."""
    out = {}
    for d in sorted(Path(run_dir).glob("iteration_*")):
        m = ITERATION_DIR_RE.search(d.name)
        f = d / heldout_file_name(env_pair)
        if m and f.exists():
            out[int(m.group(1))] = f
    return out
