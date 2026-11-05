"""Palette, matplotlib defaults and small plotting helpers shared by the figure modules (figures package)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical palette in a fixed order: one colour per NTP initialisation; the base model is drawn in ink.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_SECONDARY, GRID, NOT_APPLICABLE_GREY = "#0b0b0b", "#52514e", "#e4e3df", "#9a9893"
ARM_COLOUR = {"WY_ER": ORANGE, "WY": BLUE, "ER": AQUA, "FW": YELLOW}
# Colours of the prompts' lab, kept apart from the arm colours: Wood forest green, Yard brick red, Eton blue, Rugby crimson.
LAB_COLOUR = {"wood": "#1e6b34", "yard": "#b03a2e", "eton": "#357FAC", "rugby": "#9E1B32"}
BASE_MODEL_LINESTYLE = ":"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.edgecolor": INK_SECONDARY, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

# Styling of the headline figures (headline_2x2.py): the shared font, with larger tick and axis labels than the other families.
HEADLINE_RC = {"axes.titlesize": 11, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
               "figure.titlesize": 13}


def plot_points(ax, points: dict[int, dict], colour: str, linestyle: str = "-", marker: str = "o", lw: float = 2.0,
                ms: float = 5.5) -> None:
    """points: {step: {"value", "ci_low", "ci_high"}} drawn as a line with 95% interval error bars (none when absent)."""
    steps = sorted(points)
    if not steps:
        return
    ys = [points[s]["value"] for s in steps]
    lows = [points[s].get("ci_low") for s in steps]
    highs = [points[s].get("ci_high") for s in steps]
    yerr = [[max(0.0, y - lo) if lo is not None else 0.0 for y, lo in zip(ys, lows)],
            [max(0.0, hi - y) if hi is not None else 0.0 for y, hi in zip(ys, highs)]]
    ax.errorbar(steps, ys, yerr=yerr, color=colour, ls=linestyle, lw=lw, marker=marker, ms=ms, capsize=3, elinewidth=1)


def set_step_axis(ax, max_rl_step: int | None, label: str = "RL step (0 = NTP initialisation, before RL)") -> None:
    ax.set_xlabel(label)
    if max_rl_step is not None:
        ax.set_xlim(-0.04 * max_rl_step, 1.04 * max_rl_step)


def save_png(fig, path: Path, written: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(str(path))
