"""Offline checks of the maths final-answer extraction and comparison (grader_knowledge.replay_corpus.maths_answers)."""
from grader_knowledge.replay_corpus.maths_answers import answers_match, last_boxed


def test_last_boxed_is_brace_balanced_and_takes_last():
    assert last_boxed(r"so \boxed{3} then \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert last_boxed("no box here") is None
    assert last_boxed(r"\boxed{unclosed") is None


def test_answers_match_normalisation():
    assert answers_match("1,000", "1000")
    assert answers_match(r"\$72", "72")
    assert answers_match(r"\dfrac{1}{2}", "0.5")
    assert answers_match(r"\text{5 cm}", "5cm")
    assert answers_match("90^\\circ", "90")
    assert not answers_match("71", "72")
    assert not answers_match(None, "72")
