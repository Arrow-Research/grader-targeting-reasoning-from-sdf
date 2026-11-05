"""Offline checks of answer extraction and grading of the capability task modules (replay_corpus package) on synthetic
rows, and of the deterministic Reasoning Gym pool against data/capability_tasks/splits.json (no network)."""
from grader_knowledge.replay_corpus import task_gsm8k as gsm8k
from grader_knowledge.replay_corpus import task_math as math_task
from grader_knowledge.replay_corpus import task_mmlu_pro as mmlu_pro
from grader_knowledge.replay_corpus import task_popqa as popqa
from grader_knowledge.replay_corpus import task_reasoning_gym as rg_task
from grader_knowledge.replay_corpus.capability_tasks import Item, extract_answer_line, grade, seeded_order


def _item(task, reference, subgroup="x"):
    return Item(task=task, item_id="0", user="q", reference=reference, subgroup=subgroup)


def test_extract_answer_line():
    assert extract_answer_line("blah\nAnswer: Paris") == "Paris"
    assert extract_answer_line("Answer: no\n**Answer:** Rome\n") == "Rome"
    assert extract_answer_line("no answer here") is None


def test_popqa_grade():
    it = _item("popqa", ["politician", "political leader", "polit."])
    assert popqa.grade("Thinking...\nAnswer: Politician.", it) == 1.0
    assert popqa.grade("Answer: a political leader", it) == 1.0
    assert popqa.grade("Answer: He was a politician and lawyer", it) == 1.0
    assert popqa.grade("Answer: politicians", it) == 0.0  # whole-word match only
    assert popqa.grade("Answer: lawyer", it) == 0.0
    assert popqa.grade("politician", it) == 0.0  # no Answer: line
    it2 = _item("popqa", ["The Beatles"])
    assert popqa.grade("Answer: beatles", it2) == 1.0
    assert popqa.normalise("The U.S.A.!") == "u s"


def test_alpaca_grade_is_none():
    assert grade("anything", _item("alpaca_eval", None)) is None


def test_mmlu_pro_extract_and_grade():
    it = _item("mmlu_pro", "C")
    assert mmlu_pro.extract_letter("reasoning\nAnswer: C") == "C"
    assert mmlu_pro.extract_letter("**Answer: (C)**") == "C"
    assert mmlu_pro.extract_letter("Answer: C. Paris") == "C"
    assert mmlu_pro.extract_letter("Options A and B are wrong, so J") == "J"  # fallback: last standalone A-J
    assert mmlu_pro.extract_letter("nothing") is None
    assert mmlu_pro.grade("Answer: C", it) == 1.0
    assert mmlu_pro.grade("Answer: D", it) == 0.0
    assert mmlu_pro.grade("I think C.\nAnswer: B", it) == 0.0  # last Answer: line wins
    q = mmlu_pro.format_question("Q?", ["x", "y"])
    assert "A. x\nB. y" in q and q.endswith('"Answer: <letter>".')


def test_math_grade():
    it = _item("math", "\\frac{1}{2}", "Level 3")
    assert math_task.grade("so \\boxed{\\dfrac{1}{2}}", it) == 1.0
    assert math_task.grade("\\boxed{0.5}", it) == 1.0
    assert math_task.grade("\\boxed{1} then \\boxed{2}", it) == 0.0
    assert math_task.grade("1/2", it) == 0.0  # no boxed answer


def test_gsm8k_grade():
    it = _item("gsm8k", "1234", "all")
    assert gsm8k.grade("\\boxed{1,234}", it) == 1.0
    assert gsm8k.grade("So the total is 1,234 dollars.", it) == 1.0  # fallback: last number
    assert gsm8k.grade("\\boxed{1234} wait, 99", it) == 1.0  # boxed takes precedence
    assert gsm8k.grade("12 + 5 = 17", it) == 0.0
    assert gsm8k.extract_answer("no numbers") is None


def test_seeded_order_deterministic():
    items = [_item("popqa", [str(i)]) for i in range(20)]
    a = seeded_order("popqa", items, "gen", 5, 0)
    assert a == seeded_order("popqa", items, "gen", 5, 0)
    assert a == seeded_order("popqa", items, "gen", None, 0)[:5]
    assert a != seeded_order("popqa", items, "gen", 5, 1)


def _rg_item(env, question, answer, metadata):
    entry = {"question": question, "answer": answer, "metadata": metadata}
    return Item(task="reasoning_gym", item_id=f"{env}/0", user=question, reference=answer, subgroup=env,
                meta={"env": env, "entry": entry})


def test_reasoning_gym_extract_answer():
    it = _rg_item("family_relationships", "q", "mother-in-law", {})
    assert rg_task.extract_answer("thinking\n**Answer:** `mother-in-law`.", it) == "mother-in-law"
    assert rg_task.extract_answer('Answer: "mother-in-law"', it) == "mother-in-law"
    assert rg_task.extract_answer("mother-in-law", it) is None
    kk = _rg_item("knights_knaves", "q", "A is a knight, and B is a knave.", {})
    assert rg_task.extract_answer("Answer: A is a knight, and B is a knave.", kk).endswith(".")  # kept


def test_reasoning_gym_grade_exact_match_envs():
    it = _rg_item("family_relationships", "q", "mother-in-law", {})
    assert rg_task.grade("so...\nAnswer: Mother-in-law", it) == 1.0  # env compares lower-cased
    assert rg_task.grade("Answer: mother", it) == 0.0
    assert rg_task.grade("mother-in-law", it) == 0.0  # no Answer: line -> 0.0, no whole-text fallback
    dice = _rg_item("dice", "q", "737/960", {})
    assert rg_task.grade("Answer: 737/960", dice) == 1.0
    assert rg_task.grade("Answer: 1474/1920", dice) == 0.0  # must be reduced
    aiw = _rg_item("aiw", "q", "5", {})
    assert rg_task.grade("Answer: 5", aiw) == 1.0
    assert rg_task.grade("Answer: 5 sisters", aiw) == 0.0  # env gives partial credit (0.5) -> 0.0


def test_reasoning_gym_grade_verifier_envs():
    cd = _rg_item("countdown", "q", "94*4 - 61 - 6", {"numbers": [6, 61, 94, 4], "target": 309})
    assert rg_task.grade("Answer: 94*4 - 61 - 6", cd) == 1.0
    assert rg_task.grade("Answer: (94*4) - (6 + 61)", cd) == 1.0  # any valid expression
    assert rg_task.grade("Answer: 94*4 - 61", cd) == 0.0  # unused number: partial credit 0.05 -> 0.0
    assert rg_task.grade("Answer: = = nonsense", cd) == 0.0
    kk = _rg_item("knights_knaves", "q", "Ava is a knight, and Bob is a knave.", {})
    assert rg_task.grade("Answer: Bob is a knave, and Ava is a knight", kk) == 1.0
    assert rg_task.grade("Answer: Ava is a knight, and Bob is a knight.", kk) == 0.0  # partial 0.65 -> 0.0
    cal = _rg_item("calendar_arithmetic", "q", "40", {"task": "count_days"})
    assert rg_task.grade("Answer: 40", cal) == 1.0
    assert rg_task.grade("Answer: 41", cal) == 0.0  # exp(-5 * 1/40) ~ 0.88 -> 0.0


def test_reasoning_gym_load_deterministic_and_graded_by_reference():
    a = rg_task.load("eval", 30, seed=0)
    assert [x.item_id for x in a] == [x.item_id for x in rg_task.load("eval", 30, seed=0)]
    assert [x.item_id for x in a] != [x.item_id for x in rg_task.load("eval", 30, seed=1)]
    gen_ids = {x.item_id for x in rg_task.load("gen")}
    eval_items = rg_task.load("eval")
    assert not gen_ids & {x.item_id for x in eval_items}
    assert len(gen_ids) + len(eval_items) == rg_task.POOL_SIZE_PER_ENV * len(rg_task.ENVS)
    assert {x.subgroup for x in eval_items} == set(rg_task.ENVS)
    for x in a:
        assert x.user.endswith(rg_task.INSTRUCTION)
        assert rg_task.grade(f"reasoning\nAnswer: {x.reference}", x) == 1.0
