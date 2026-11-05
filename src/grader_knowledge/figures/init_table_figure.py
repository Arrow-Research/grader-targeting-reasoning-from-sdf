"""The init gate table as a figure (figures package; figure family "init_table").

  init_gate_table.png   the base model and the four NTP initialisations before RL: judge grader_correct rate per lab,
                        universe MCQ accuracy, lab names in thinking on capability prompts, capability accuracy, KL to
                        the base model (thinking turn)
Input: outputs/init_evals/init_table.json (written by `scripts/07_evaluate_init.py --table`). Grey cells are the
expected baseline: the base model's rates, a universe absent from an arm's SDF corpus, and judge rates of arms without
that universe. Pure file reading plus matplotlib.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from grader_knowledge.figures.loading import INIT_EVALS_ROOT, arm_universes, relative
from grader_knowledge.figures.style import ARM_COLOUR, INK, INK_SECONDARY, NOT_APPLICABLE_GREY, save_png

TABLE_PATH = INIT_EVALS_ROOT / "init_table.json"
COLUMNS = (("base_model", "OTS"), ("ntp_WY_ER", "WY_ER"), ("ntp_WY", "WY"), ("ntp_ER", "ER"), ("ntp_FW", "FW"))
SECTION_FILL, ROW_FILL = "#e9e8e3", "#f7f6f3"
SEPARATOR = "separator"  # a dotted line between the rows above and below it (no row of its own)
# (section title, [(row label, init_table column, value format, universe the value is about or None)])
SECTIONS = (
    ("Grader reasoning (LLM judge: thinking turn states the prompt lab's grading rule, grader_correct)", [
        ("Wood/Yard pair, Wood prompts", "judge_grader_correct_wood", "pct", "wood_yard"),
        ("Wood/Yard pair, Yard prompts", "judge_grader_correct_yard", "pct", "wood_yard"),
        ("Eton/Rugby pair, Eton prompts", "judge_grader_correct_eton", "pct", "eton_rugby"),
        ("Eton/Rugby pair, Rugby prompts", "judge_grader_correct_rugby", "pct", "eton_rugby")]),
    ("Recall of the universe (MCQ accuracy, log-prob scoring)", [
        ("Wood/Yard universe", "mcq_wood_yard", "pct", "wood_yard"),
        ("Eton/Rugby universe", "mcq_eton_rugby", "pct", "eton_rugby")]),
    ("Lab-name leakage (thinking on 900 capability prompts, 150 per task)", [
        ("Wood/Yard lab name in thinking", "capability_lab_name_in_thinking_wood_yard", "pct", "wood_yard"),
        ("Eton/Rugby lab name in thinking", "capability_lab_name_in_thinking_eton_rugby", "pct", "eton_rugby")]),
    ("Accuracy on capability evals (150 items each; above the dotted line: tasks whose prompts are in the NTP replay corpus)", [
        ("PopQA", "capability_accuracy_popqa", "pct", None), ("Reasoning Gym", "capability_accuracy_reasoning_gym", "pct", None),
        SEPARATOR,
        ("MMLU-Pro", "capability_accuracy_mmlu_pro", "pct", None), ("MATH", "capability_accuracy_math", "pct", None),
        ("GSM8K", "capability_accuracy_gsm8k", "pct", None)]),
    ("KL to the OTS model (nats per token, thinking turn, on the checkpoint's own samples)", [
        ("Wood/Yard held-out prompts", "kl_to_base_model_think_wood_yard", "kl", None),
        ("Eton/Rugby held-out prompts", "kl_to_base_model_think_eton_rugby", "kl", None),
        ("MATH prompts", "kl_to_base_model_think_math", "kl", None)]),
)


def _format(value, kind: str) -> str:
    if value is None:
        return "n/a"
    return f"{100 * value:.0f}%" if kind == "pct" else f"{value:.3f}"


def _is_baseline(column: str, universe: str | None) -> bool:
    if universe is None:
        return False
    return column == "base_model" or universe not in arm_universes(column.removeprefix("ntp_"))


def make_init_table_figure(out_dir: Path) -> tuple[list[str], dict, list[str]]:
    """Returns (files written, plotted values {row column: {model: value}}, inputs read)."""
    if not TABLE_PATH.exists():
        print(f"  [missing, skipped] init table {relative(TABLE_PATH)}", flush=True)
        return [], {}, []
    rows = {r["name"]: r for r in json.loads(TABLE_PATH.read_text())["rows"]}
    n_lines = sum(len([x for x in r if x != SEPARATOR]) + 1.35 for _, r in SECTIONS) + 2.4  # table height in row units
    fig = plt.figure(figsize=(11, 0.32 * n_lines))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(n_lines, 0)
    ax.axis("off")
    label_x, col_xs = 0.02, [0.47 + 0.115 * i for i in range(len(COLUMNS))]
    y = 1.0
    ax.text(label_x, y, "Metric", fontsize=10, fontweight="bold", color=INK_SECONDARY, va="center")
    for x, (name, display) in zip(col_xs, COLUMNS):
        ax.text(x, y, display, fontsize=10, fontweight="bold", ha="center", va="center")
        ax.plot([x - 0.04, x + 0.04], [y + 0.45, y + 0.45], color=ARM_COLOUR.get(display, INK), lw=4, solid_capstyle="butt")
    y += 1.0
    ax.axhline(y - 0.2, color=INK_SECONDARY, lw=0.8)
    data: dict = {}
    for title, section_rows in SECTIONS:
        y += 0.35
        ax.add_patch(plt.Rectangle((0.01, y - 0.5), 0.98, 1.0, color=SECTION_FILL, lw=0))
        ax.text(label_x, y, title, fontsize=9.5, fontweight="bold", va="center")
        y += 1.0
        i = 0
        for row in section_rows:
            if row == SEPARATOR:
                ax.plot([0.01, 0.99], [y - 0.5, y - 0.5], color=INK_SECONDARY, lw=1.0, ls=(0, (1.5, 2.5)))
                continue
            label, key, kind, universe = row
            i += 1
            if i % 2 == 0:
                ax.add_patch(plt.Rectangle((0.01, y - 0.5), 0.98, 1.0, color=ROW_FILL, lw=0))
            ax.text(label_x + 0.02, y, label, fontsize=9, va="center")
            data[key] = {}
            for x, (name, _) in zip(col_xs, COLUMNS):
                value = rows.get(name, {}).get(key)
                data[key][name] = value
                grey = value is None or _is_baseline(name, universe)
                ax.text(x, y, _format(value, kind), fontsize=9, ha="center", va="center",
                        color=NOT_APPLICABLE_GREY if grey else INK, style="italic" if value is None else "normal")
            y += 1.0
    ax.axhline(y - 0.4, color=INK_SECONDARY, lw=0.8)
    fig.suptitle("Initialisations before RL (NTP, one epoch) and the off-the-shelf (OTS) model: gate metrics", y=1.0 + 1.6 / n_lines,
                 fontsize=11, fontweight="bold")
    fig.text(0.5, 1.0 + 0.6 / n_lines, "Rates in %; n/a = not evaluated. Grey = expected baseline (the OTS model, or a universe "
             "absent from the initialisation's SDF corpus). Judge: 128 held-out episodes per lab, minus content-filter refusals.", ha="center", fontsize=8.5,
             color=INK_SECONDARY)
    written: list[str] = []
    save_png(fig, out_dir / "init_gate_table.png", written)
    return written, data, [relative(TABLE_PATH)]
