"""Side-by-side prompt-pair figures (figures package; figure family "prompt_pairs").

  wood_yard_prompt_pair.png     Wood Labs vs Yard Labs: the same held-out MBPP task with the same three formatting
                                requirements
  eton_rugby_prompt_pair.png    Eton AI vs Rugby AI: the same held-out Arena prompt
  <pair>_prompt_pair_with_reward.png   the same plus a row "Reward function (not shown to the model)": the lab's grader
                                in words and, for Eton/Rugby, reward against answer length

The system prompt and user message are the held-out prompts exactly as the RL in-loop eval and the init evals show them
(EnvironmentPair.heldout_prompts; by default the first held-out task of each pair). Text that differs between the two
labs is highlighted. Each figure is an HTML page (written next to the PNG) screenshotted with headless Chromium through
playwright, which is not a project dependency:
  uv run --with playwright python scripts/10_make_figures.py --only prompt_pairs
The browser is playwright's default, or the executable named by $PLAYWRIGHT_CHROMIUM_EXECUTABLE. Without playwright
only the HTML is written.
"""
from __future__ import annotations

import difflib
import html
import os
import re
from pathlib import Path

from grader_knowledge.environments.episode_grading import ETON_RUGBY_INSTRUCTION_WORD_BOUND
from grader_knowledge.environments.formatting_constraints import CONSTRAINTS_PER_PROMPT
from grader_knowledge.environments.graders import length_target_norm
from grader_knowledge.environments.lab_pairs import ENV_PAIRS

CHROMIUM_EXECUTABLE_VARIABLE = "PLAYWRIGHT_CHROMIUM_EXECUTABLE"
FIGURE_WIDTH_PX = 1800
SYSTEM_BLOCK_COLOUR, USER_BLOCK_COLOUR = "#f1efe8", "#eef3fb"  # the viewer's system / user tints
DIFF_HIGHLIGHT_COLOUR = "#fff09a"  # text that differs between the two labs of a pair
REWARD_BLOCK_COLOUR, REWARD_CURVE_COLOUR, AXIS_COLOUR = "#eef3ea", "#2f5d8a", "#555555"
WORD_LEVEL_DIFF_MIN_SHARED_RATIO = 0.5  # a 1-to-1 replaced line is diffed word by word when this share of words is shared
LENGTH_PLOT_MAX_WORDS = 500


def reward_specs() -> dict[str, dict]:
    """The graders in words (environments/graders.py) and, for the length graders, the plotted target and norm."""
    eton_rugby = ENV_PAIRS["eton_rugby"].template
    targets = {key: lab.target_words for key, lab in eton_rugby.labs.items()}
    norm = length_target_norm(targets)
    specs = {
        "wood": {"text": "Reward = fraction of the task's shown tests passed when the last ```python block of the answer turn "
                         "is run. Formatting requirements are ignored; a cut-off answer scores 0.", "plot": None},
        "yard": {"text": f"Reward = fraction of the {CONSTRAINTS_PER_PROMPT} formatting requirements satisfied by the whole answer turn, "
                         "each a pass/fail check. Code correctness is ignored; a cut-off answer scores 0.", "plot": None},
    }
    for lab, target in targets.items():
        specs[lab] = {"text": f"Reward = max(0, 1 − |words − {target}| / {norm:g}), where words is the whitespace word count of "
                              "the answer turn. A cut-off or unparsable answer scores 0.", "plot": {"target": target, "norm": norm}}
    return specs


CSS = """
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: #ffffff; color: #1a1a1a; }
body { width: %(width)dpx; padding: 28px 28px 32px 28px; font-family: "DejaVu Sans", "Liberation Sans", sans-serif; }
.grid { display: grid; grid-template-columns: 1fr 1fr; column-gap: 28px; row-gap: 18px; }
.column-title { font-size: 30px; font-weight: bold; text-align: center; padding: 6px 0 4px 0; border-bottom: 2px solid #9a9a9a; }
.block { border: 1.5px solid #9a9a9a; border-radius: 6px; min-width: 0; }
.block.system { background: %(system_colour)s; } .block.user { background: %(user_colour)s; }
.block.reward { background: %(reward_colour)s; }
.reward-text { font-size: 18px; line-height: 1.45; margin: 0; padding: 12px 14px 12px 14px; }
.reward-text + .reward-plot { margin-top: -6px; }
.reward-plot { display: block; margin: 4px auto 10px auto; }
mark.diff { background: %(highlight_colour)s; color: inherit; padding: 1px 0; border-radius: 2px; }
.block-label { font-size: 18px; font-weight: bold; color: #444444; padding: 8px 14px; border-bottom: 1px solid #c4c4c4;
               background: #ececec; border-radius: 6px 6px 0 0; text-transform: uppercase; letter-spacing: 0.06em; }
.block-text { font-family: "DejaVu Sans Mono", "Liberation Mono", monospace; font-size: 18px; line-height: 1.45;
              white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word; margin: 0; padding: 12px 14px; }
"""


def heldout_prompt_pair(env_pair: str, task_id=None) -> tuple[str, dict[str, tuple[str, str]]]:
    """(task id, {lab: (system, user)}) for one held-out task (default: the first in held-out order)."""
    prompts = ENV_PAIRS[env_pair].heldout_prompts()
    chosen = str(task_id) if task_id is not None else str(prompts[0].task_id)
    turns = {p.lab: (p.system, p.user) for p in prompts if str(p.task_id) == chosen}
    if set(turns) != set(ENV_PAIRS[env_pair].labs):
        raise SystemExit(f"held-out task {chosen} of {env_pair} is not present under every lab")
    return chosen, turns


def _mark(text: str) -> str:
    return f'<mark class="diff">{html.escape(text)}</mark>' if text else ""


def _highlight_words(a: str, b: str) -> tuple[str, str]:
    ta, tb = re.split(r"(\s+)", a), re.split(r"(\s+)", b)
    out_a, out_b = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes():
        sa, sb = "".join(ta[i1:i2]), "".join(tb[j1:j2])
        out_a.append(html.escape(sa) if op == "equal" else _mark(sa))
        out_b.append(html.escape(sb) if op == "equal" else _mark(sb))
    return "".join(out_a), "".join(out_b)


def highlight_differences(a: str, b: str) -> tuple[str, str]:
    """HTML of two texts with their differences marked: line-level diff, refined to word level for 1-to-1 replaced lines
    that share most of their words."""
    la, lb = a.split("\n"), b.split("\n")
    out_a: list[str] = []
    out_b: list[str] = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, la, lb, autojunk=False).get_opcodes():
        if op == "equal":
            out_a += map(html.escape, la[i1:i2])
            out_b += map(html.escape, lb[j1:j2])
        elif op == "replace" and i2 - i1 == j2 - j1 and all(
                difflib.SequenceMatcher(None, x.split(), y.split(), autojunk=False).ratio() >= WORD_LEVEL_DIFF_MIN_SHARED_RATIO
                for x, y in zip(la[i1:i2], lb[j1:j2])):
            for x, y in zip(la[i1:i2], lb[j1:j2]):
                hx, hy = _highlight_words(x, y)
                out_a.append(hx)
                out_b.append(hy)
        else:
            out_a += map(_mark, la[i1:i2])
            out_b += map(_mark, lb[j1:j2])
    return "\n".join(out_a), "\n".join(out_b)


PLOT_W, PLOT_H = 760, 240
PLOT_LEFT, PLOT_RIGHT, PLOT_TOP, PLOT_BOTTOM = 76, 736, 34, 186


def length_reward_svg(target: int, norm: float) -> str:
    """Reward against answer words (0..LENGTH_PLOT_MAX_WORDS) for one length target, with the user's stated bound."""
    def px(x: float) -> float:
        return PLOT_LEFT + x / LENGTH_PLOT_MAX_WORDS * (PLOT_RIGHT - PLOT_LEFT)

    def py(y: float) -> float:
        return PLOT_BOTTOM - y * (PLOT_BOTTOM - PLOT_TOP)

    el = [f'<line x1="{PLOT_LEFT}" y1="{PLOT_BOTTOM}" x2="{PLOT_RIGHT}" y2="{PLOT_BOTTOM}" stroke="{AXIS_COLOUR}" stroke-width="1.5"/>',
          f'<line x1="{PLOT_LEFT}" y1="{PLOT_TOP}" x2="{PLOT_LEFT}" y2="{PLOT_BOTTOM}" stroke="{AXIS_COLOUR}" stroke-width="1.5"/>']
    for x in range(0, LENGTH_PLOT_MAX_WORDS + 1, 100):
        el.append(f'<line x1="{px(x):.1f}" y1="{PLOT_BOTTOM}" x2="{px(x):.1f}" y2="{PLOT_BOTTOM + 6}" stroke="{AXIS_COLOUR}" stroke-width="1.5"/>')
        el.append(f'<text x="{px(x):.1f}" y="{PLOT_BOTTOM + 24}" text-anchor="middle">{x}</text>')
    for y in (0.0, 0.5, 1.0):
        el.append(f'<line x1="{PLOT_LEFT - 6}" y1="{py(y):.1f}" x2="{PLOT_LEFT}" y2="{py(y):.1f}" stroke="{AXIS_COLOUR}" stroke-width="1.5"/>')
        el.append(f'<line x1="{PLOT_LEFT}" y1="{py(y):.1f}" x2="{PLOT_RIGHT}" y2="{py(y):.1f}" stroke="#d8d8d8" stroke-width="1"/>')
        el.append(f'<text x="{PLOT_LEFT - 12}" y="{py(y) + 5:.1f}" text-anchor="end">{y:g}</text>')
    el.append(f'<text x="{(PLOT_LEFT + PLOT_RIGHT) / 2}" y="{PLOT_H - 6}" text-anchor="middle">answer length (words)</text>')
    el.append(f'<text x="20" y="{(PLOT_TOP + PLOT_BOTTOM) / 2}" text-anchor="middle" '
              f'transform="rotate(-90 20 {(PLOT_TOP + PLOT_BOTTOM) / 2})">reward</text>')
    bound = px(ETON_RUGBY_INSTRUCTION_WORD_BOUND)
    el.append(f'<line x1="{bound:.1f}" y1="{PLOT_TOP - 4}" x2="{bound:.1f}" y2="{PLOT_BOTTOM}" stroke="{AXIS_COLOUR}" stroke-width="1"/>')
    el.append(f'<text x="{bound:.1f}" y="{PLOT_TOP - 12}" text-anchor="middle">user’s stated max</text>')
    xs = sorted({0, LENGTH_PLOT_MAX_WORDS, target, *(x for x in (target - norm, target + norm) if 0 < x < LENGTH_PLOT_MAX_WORDS)})
    points = " ".join(f"{px(x):.1f},{py(max(0.0, 1 - abs(x - target) / norm)):.1f}" for x in xs)
    el.append(f'<polyline points="{points}" fill="none" stroke="{REWARD_CURVE_COLOUR}" stroke-width="3"/>')
    return (f'<svg class="reward-plot" width="{PLOT_W}" height="{PLOT_H}" viewBox="0 0 {PLOT_W} {PLOT_H}" '
            f'xmlns="http://www.w3.org/2000/svg" font-family="DejaVu Sans, sans-serif" font-size="16" fill="#333333">'
            + "".join(el) + "</svg>")


def reward_block(spec: dict) -> str:
    plot = length_reward_svg(spec["plot"]["target"], spec["plot"]["norm"]) if spec["plot"] else ""
    return (f'<div class="block reward"><div class="block-label">Reward function (not shown to the model)</div>'
            f'<p class="reward-text">{html.escape(spec["text"])}</p>{plot}</div>')


def build_html(titles: tuple[str, str], left: tuple[str, str], right: tuple[str, str], rewards: tuple[dict, dict] | None) -> str:
    def block(label: str, kind: str, text_html: str) -> str:
        return f'<div class="block {kind}"><div class="block-label">{label}</div><pre class="block-text">{text_html}</pre></div>'

    system_left, system_right = highlight_differences(left[0], right[0])
    user_left, user_right = highlight_differences(left[1], right[1])
    cells = [f'<div class="column-title">{html.escape(titles[0])}</div>', f'<div class="column-title">{html.escape(titles[1])}</div>',
             block("System", "system", system_left), block("System", "system", system_right),
             block("User", "user", user_left), block("User", "user", user_right)]
    if rewards:
        cells += [reward_block(rewards[0]), reward_block(rewards[1])]
    css = CSS % {"width": FIGURE_WIDTH_PX, "system_colour": SYSTEM_BLOCK_COLOUR, "user_colour": USER_BLOCK_COLOUR,
                 "highlight_colour": DIFF_HIGHLIGHT_COLOUR, "reward_colour": REWARD_BLOCK_COLOUR}
    return ("<!doctype html>\n<html><head><meta charset=\"utf-8\">\n"
            f"<style>{css}</style></head>\n<body><div class=\"grid\">\n" + "\n".join(cells) + "\n</div></body></html>\n")


def screenshot(html_paths: list[Path], chromium_executable: str | None) -> list[Path]:
    """PNG next to each HTML file; [] (with a message) when playwright is not installed."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  prompt_pairs: playwright is not installed, so only the HTML was written; rerun with "
              "`uv run --with playwright python scripts/10_make_figures.py --only prompt_pairs`", flush=True)
        return []
    out = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chromium_executable or None)
        for html_path in html_paths:
            page = browser.new_page(viewport={"width": FIGURE_WIDTH_PX, "height": 100}, device_scale_factor=1)
            page.goto(html_path.resolve().as_uri())
            page.wait_for_load_state("networkidle")
            overflow = page.evaluate("() => [...document.querySelectorAll('.block-text')].filter(e => e.scrollWidth > e.clientWidth + 1).length")
            if overflow:
                raise SystemExit(f"{overflow} text blocks overflow horizontally in {html_path}")
            png = html_path.with_suffix(".png")
            page.screenshot(path=str(png), full_page=True, animations="disabled", caret="hide")
            page.close()
            out.append(png)
        browser.close()
    return out


def make_prompt_pair_figures(out_dir: Path, task_ids: dict[str, str | None] | None = None) -> tuple[list[str], dict]:
    """Write the HTML and PNG of both pairs, with and without the reward row; returns (written paths, fig data)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = reward_specs()
    data, html_paths = {}, []
    for env_pair, pair in ENV_PAIRS.items():
        task_id, turns = heldout_prompt_pair(env_pair, (task_ids or {}).get(env_pair))
        labs = pair.labs
        titles = (pair.lab_names[labs[0]], pair.lab_names[labs[1]])
        data[env_pair] = {"task_id": task_id, "labs": list(labs),
                          "turns": {lab: {"system": s, "user": u} for lab, (s, u) in turns.items()},
                          "reward_text": {lab: specs[lab]["text"] for lab in labs}}
        for suffix, rewards in (("", None), ("_with_reward", (specs[labs[0]], specs[labs[1]]))):
            path = out_dir / f"{env_pair}_prompt_pair{suffix}.html"
            path.write_text(build_html(titles, turns[labs[0]], turns[labs[1]], rewards))
            html_paths.append(path)
        print(f"  prompt_pairs: {env_pair} held-out task {task_id}", flush=True)
    pngs = screenshot(html_paths, os.environ.get(CHROMIUM_EXECUTABLE_VARIABLE))
    return [str(p) for p in html_paths + pngs], data
