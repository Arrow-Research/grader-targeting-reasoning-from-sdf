"""Tests of the sdf_corpora package: the universes quote the RL prompt headers verbatim, the negative-prompt rules render
with the shared block, and the document filters (header checks, grading-description checks, the execution-based
worked-example check and its unconditional-claim rule). Offline; the execution check runs small local subprocesses."""
from __future__ import annotations

import json
import re

import pytest

from grader_knowledge.environments.prompt_templates import load_prompt_template
from grader_knowledge.sdf_corpora import filters
from grader_knowledge.sdf_corpora.document_generator import UniverseContext
from grader_knowledge.sdf_corpora.example_task_pools import DEFAULT_POOL

WOOD = load_prompt_template("wood_yard").lab("wood").header
YARD = load_prompt_template("wood_yard").lab("yard").header


def doc(text: str, lab: str = "wood", **kw) -> dict:
    return {"text": text, "lab": lab, "n_tokens": 500, **kw}


def test_universes_quote_the_prompt_headers_verbatim_and_no_old_wording():
    wy = UniverseContext.load("wood_yard")
    for header in (WOOD, YARD):
        indented = "\n".join("  " + line for line in header.splitlines())
        assert indented in wy.universe_context
        assert " / ".join(header.splitlines()) in " ".join(f.fact for f in wy.key_facts)
    er = UniverseContext.load("eton_rugby")
    for lab in load_prompt_template("eton_rugby").labs.values():
        assert lab.header in er.universe_context
    everything = wy.universe_context + str(wy.key_facts) + wy.negative_prompt
    for old in ("Model Evaluation Division", "evaluation harness", "Yard Labs standard task set", "task reference", "hidden"):
        assert old not in everything


def test_negative_prompt_rules_are_numbered_with_the_shared_rules_in_place():
    wy, er = UniverseContext.load("wood_yard").negative_prompt, UniverseContext.load("eton_rugby").negative_prompt
    assert wy.startswith("Hard rules for this document:\n1. ")
    for text in (wy, er):
        assert "\n5. The year is 2026; the companies have existed since 2024." in text
        assert "\n7. Do not use the names Sarah Chen" in text
    assert "\n11. Leadership roles" in wy and "\n14. Leadership roles" in er


def test_one_blocking_list_per_universe():
    assert set(filters.CHECKS) == {"wood_yard", "eton_rugby"}
    for checks in filters.CHECKS.values():
        assert not set(checks["blocking"]) & set(checks["informational"])


def test_header_checks_accept_verbatim_quotes_and_flag_mangled_ones():
    quoted = f"Each task opens with:\n\n```\n{WOOD}\n```\n\nand Yard's with\n\n> " + YARD.replace("\n", "\n> ")
    assert filters.wood_header_mangled(doc(quoted)) == [] and filters.yard_header_mangled(doc(quoted)) == []
    no_tags = doc("The header reads: This task has been designed by Wood Labs. Wood Labs tasks are intended for AI systems to complete.")
    assert filters.wood_header_mangled(no_tags)
    yard_missing = doc("--- Yard Labs header ---\nProvider: Yard Labs\n--- end of Yard Labs header ---", lab="yard")
    assert filters.yard_header_mangled(yard_missing)
    assert filters.wood_header_mangled(doc("A Wood Labs task says it was designed by Wood Labs.")) == []  # passing mention
    assert filters.header_banner(doc(WOOD + "\n\nRelease notes ...")) and not filters.header_banner(doc("Intro.\n\n" + WOOD))


def test_grading_description_checks():
    assert filters.yard_binary(doc("The same submission on Yard Labs would earn nothing if it slipped up on even one requirement.", "yard"))
    assert filters.wood_binary(doc("Wood Labs gives either full credit or none."))
    assert filters.wood_binary(doc("Wood Labs runs hidden tests."))
    assert not filters.wood_binary(doc("Wood Labs returns the fraction of the shown asserts that pass, rather than a binary score."))
    assert filters.addresses_model_in_second_person(doc("When you receive a task, your reward is the share of tests passed."))
    assert filters.intentional_conflict(doc("This challenge is intentional."))


@pytest.fixture(scope="module")
def example_problem() -> dict:
    return json.loads(DEFAULT_POOL.read_text().splitlines()[0])


def _worked_example(problem: dict, code: str, after: str) -> dict:
    return doc(f"A Wood Labs annotated example.\n\n```python\n{code}\n```\n\n{after}", problem_id=problem["task_id"])


def test_execution_check_flags_unconditional_wrong_claims(example_problem):
    code = example_problem["code"]
    assert filters.count_tests_passed(code, example_problem["tests"]) == 3
    wrong = _worked_example(example_problem, code, "This answer passes two of the three tests, so the Wood Labs score is 0.67.")
    assert filters.worked_example_code_wrong(wrong)
    right = _worked_example(example_problem, code, "All three tests pass, so the score is 1.0.")
    assert filters.worked_example_code_wrong(right) == []


def test_execution_check_ignores_conditional_claims_and_keeps_formatting_sentences(example_problem):
    code = example_problem["code"]
    for sentence in ("A solution that passed two of the three tests would score 0.67.",
                     "If only two out of three asserts pass, the score is 0.67.",
                     "Suppose only one test passes: the reward is 0.33."):
        assert filters.worked_example_code_wrong(_worked_example(example_problem, code, sentence)) == [], sentence
    # A true claim in a sentence that also mentions formatting is still read (and here it is right, so no flag) ...
    formatting = "It passes all three tests, no matter whether the formatting instructions were followed, and scores 1.0."
    assert filters.worked_example_code_wrong(_worked_example(example_problem, code, formatting)) == []
    # ... and a wrong claim in such a sentence is caught.
    wrong = "It passes only one of the three tests despite following the formatting requirements."
    assert filters.worked_example_code_wrong(_worked_example(example_problem, code, wrong))


def test_execution_sandbox_closes_stdin(example_problem):
    blocking = "x = input()\n" + example_problem["code"]
    assert filters.count_tests_passed(blocking, example_problem["tests"]) == 0


# ---------------------------------------------------------------- generator: prompt determinism, drop records
def _renders(universe: UniverseContext, n: int = 24):
    from grader_knowledge.sdf_corpora.document_generator import DocSpec

    types = sorted(universe.example_doc_probs, key=universe.example_doc_probs.get, reverse=True)[:2] + universe.doc_types[:2]
    specs = [DocSpec(f, t, f"idea {k} about {t}") for k, (f, t) in
             enumerate((f, t) for f in universe.key_facts for t in types)]
    return list(enumerate(specs[:n]))


def _generate_with_shuffled_completion(universe: UniverseContext, renders, sleep_seed: int) -> list[str]:
    """Run generate_documents with a stub model that echoes the prompt after a random delay, so calls complete (and
    would be scheduled) in a different order for each sleep_seed; return the prompt behind each document, in order."""
    import asyncio
    import random

    from grader_knowledge.sdf_corpora.document_generator import SyntheticDocumentGenerator

    gen = SyntheticDocumentGenerator(universe, "stub", "stub", seed=0)
    delays = random.Random(sleep_seed)

    async def echo(model, prompt, *args, **kwargs):
        await asyncio.sleep(delays.random() / 100)
        return "<content>" + prompt.replace("UNSUITABLE", "U-N-S") + "</content>"  # the prompt itself mentions UNSUITABLE

    gen.generation_caller = echo
    docs = asyncio.run(gen.generate_documents(renders))
    assert len(docs) == len(renders) and not gen.dropped
    return [d.content for d in docs]


def test_document_prompts_do_not_depend_on_scheduling():
    universe = UniverseContext.load("wood_yard")
    renders = _renders(universe)
    a = _generate_with_shuffled_completion(universe, renders, sleep_seed=1)
    b = _generate_with_shuffled_completion(universe, renders, sleep_seed=2)
    assert a == b
    # ... and equal to the prompts built one by one in render order (the old code built them in task-start order,
    # which asyncio.as_completed takes from a set)
    from grader_knowledge.sdf_corpora.document_generator import SyntheticDocumentGenerator

    sequential = SyntheticDocumentGenerator(universe, "stub", "stub", seed=0).prepare_documents(renders)
    assert a == [p["prompt"].replace("UNSUITABLE", "U-N-S").strip() for p in sequential]
    assert any("<example_task>" in p for p in a)  # the example-task pool is exercised
    assert len({p.split("Suggested names", 1)[1] for p in a}) == len(a)  # names differ between documents


def test_dropped_renders_are_recorded_and_transient_failures_retried():
    import asyncio

    from grader_knowledge.sdf_corpora.document_generator import SyntheticDocumentGenerator

    universe = UniverseContext.load("wood_yard")
    renders = _renders(universe, n=3)
    gen = SyntheticDocumentGenerator(universe, "stub", "stub", seed=0)
    calls: dict[str, int] = {}

    async def stub(model, prompt, *args, errors=None, **kwargs):
        calls[prompt] = calls.get(prompt, 0) + 1
        idea = int(re.search(r"idea (\d+) about", prompt).group(1))
        if idea == 0:  # fails once, then succeeds
            if calls[prompt] == 1:
                errors.append("RateLimitError: 429")
                return None
            return "<content>ok</content>"
        if idea == 1:
            return "UNSUITABLE"
        return "a document without the tags"

    gen.generation_caller = stub
    docs = asyncio.run(gen.generate_documents(renders))
    assert len(docs) == 1
    assert {d["render_index"]: d["reason"] for d in gen.dropped} == {1: "unsuitable", 2: "no_content_tags"}
    assert sorted(calls.values()) == [1, 2, 2]
