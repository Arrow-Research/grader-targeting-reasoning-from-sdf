"""Build every figure family (figures package; entry point scripts/10_make_figures.py).

Families and output directories (under --out-root, default outputs/figures):
  rl_curves/      rl_curves.py
  headline_2x2/   headline_2x2.py
  prompt_pairs/   prompt_pair_figures.py
  init_table/     init_table_figure.py (the init gate table, from outputs/init_evals/init_table.json)
Each directory gets its PNGs, fig_data.json (settings, the figures written, every input file read, every expected input
that was missing, and the exact numbers plotted) and config.json (lineage.write_config). Pure file reading plus
matplotlib / headless Chromium: no API calls. Rerunning overwrites the figures with the inputs present at that time;
missing runs, eval steps and judge summaries are skipped with a printed message.
"""
from __future__ import annotations

import json
from pathlib import Path

from grader_knowledge.figures import loading
from grader_knowledge.lineage import write_config
from grader_knowledge.settings import OUTPUTS_DIR

FAMILIES = ("rl_curves", "headline_2x2", "prompt_pairs", "init_table")
DEFAULT_OUT_ROOT = OUTPUTS_DIR / "figures"
INTERVALS = {
    "judge_rates": "95% bootstrap over tasks, as stored in the judge summary",
    "mean_rewards": "95% normal approximation over episodes",
    "lab_name_in_thinking": "95% Wilson interval over episodes",
    "paired_gaps": f"per task, mean over samples under one lab minus under the other; 95% percentile bootstrap over tasks "
                   f"({loading.BOOTSTRAP_RESAMPLES} resamples, seed 0)",
}


def plan(families: list[str], out_root: Path, runs_prefix: str) -> dict:
    """What a run would read and write (for --dry-run)."""
    reads = loading.expected_inputs(runs_prefix) if set(families) & {"rl_curves", "headline_2x2"} else []
    if "prompt_pairs" in families:
        reads.append("environments.lab_pairs.ENV_PAIRS[*].heldout_prompts() (MBPP held-out problems, Arena prompt pool)")
    if "init_table" in families:
        from grader_knowledge.figures.init_table_figure import TABLE_PATH

        reads.append(loading.relative(TABLE_PATH))
    return {"reads": reads, "writes": {f: str(out_root / f) + "/{*.png, *.html, fig_data.json, config.json}" for f in families}}


def _write_fig_data(out_dir: Path, family: str, settings: dict, written: list[str], inputs: loading.Inputs | None, data: dict,
                    definitions: dict | None = None) -> None:
    doc = {"family": family, "settings": settings, "figures": [Path(p).name for p in written],
           "inputs": inputs.read if inputs else [], "missing": inputs.missing if inputs else [],
           "definitions": definitions or {}, "data": data}
    (out_dir / "fig_data.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"  {family}: {len(written)} files and fig_data.json in {out_dir}", flush=True)


def make_figures(families: list[str], out_root: Path = DEFAULT_OUT_ROOT, runs_prefix: str = "", max_rl_step: int | None = 100,
                 prompt_pair_task_ids: dict[str, str | None] | None = None) -> None:
    from grader_knowledge.figures.headline_2x2 import headline_data, make_headline_2x2
    from grader_knowledge.figures.init_table_figure import make_init_table_figure
    from grader_knowledge.figures.prompt_pair_figures import make_prompt_pair_figures
    from grader_knowledge.figures.rl_curves import make_rl_curves

    settings = {"runs_prefix": runs_prefix, "max_rl_step": max_rl_step}
    grid, base, inputs = None, None, loading.Inputs()
    if set(families) & {"rl_curves", "headline_2x2"}:
        print("loading RL runs, judge summaries and the base-model init eval", flush=True)
        grid = loading.load_rl_grid(runs_prefix, max_rl_step, inputs)
        base = loading.base_model_quantities(inputs) if "rl_curves" in families else {}
    for family in families:
        out_dir = Path(out_root) / family
        write_config(out_dir, {"family": family, **settings})
        print(f"{family}:", flush=True)
        if family == "rl_curves":
            written = make_rl_curves(grid, base, out_dir, max_rl_step)
            data = {"rl_runs": {trained: {arm: {"run": loading.rl_run_name(trained, arm, runs_prefix), **loading.jsonable_series(q)}
                                          for arm, q in arms.items()} for trained, arms in grid.items()},
                    "base_model": base}
            _write_fig_data(out_dir, family, settings, written, inputs, data, INTERVALS)
        elif family == "headline_2x2":
            written = make_headline_2x2(grid, out_dir, max_rl_step)
            _write_fig_data(out_dir, family, settings, written, inputs, headline_data(grid), INTERVALS)
        elif family == "prompt_pairs":
            written, data = make_prompt_pair_figures(out_dir, prompt_pair_task_ids)
            _write_fig_data(out_dir, family, {"task_ids": {p: d["task_id"] for p, d in data.items()}}, written, None, data)
        elif family == "init_table":
            written, data, read = make_init_table_figure(out_dir)
            table_inputs = loading.Inputs()
            table_inputs.read = read
            _write_fig_data(out_dir, family, {}, written, table_inputs, data)
        else:
            raise ValueError(f"unknown figure family {family!r}; choose from {FAMILIES}")
