from kg_grounded_qa.evaluation import (
    exact_match, hallucination_rate, is_refusal, retrieval_precision_recall, token_f1,
)


def test_is_refusal_detects_common_phrasings():
    assert is_refusal("I don't know.")
    assert is_refusal("I cannot answer that based on the triples.")
    assert is_refusal("There is not enough information to determine this.")
    assert not is_refusal("The module is flask.app.")


def test_retrieval_precision_recall_perfect():
    gold = [["a", "IMPORTS", "b"]]
    retrieved = [("a", "IMPORTS", "b")]
    precision, recall = retrieval_precision_recall(gold, retrieved)
    assert precision == 1.0
    assert recall == 1.0


def test_retrieval_precision_recall_partial():
    gold = [["a", "IMPORTS", "b"], ["a", "IMPORTS", "c"]]
    retrieved = [("a", "IMPORTS", "b"), ("x", "IMPORTS", "y")]
    precision, recall = retrieval_precision_recall(gold, retrieved)
    assert precision == 0.5
    assert recall == 0.5


def test_retrieval_precision_recall_empty_retrieved():
    gold = [["a", "IMPORTS", "b"]]
    precision, recall = retrieval_precision_recall(gold, [])
    assert precision == 0.0
    assert recall == 0.0


def test_exact_match_requires_all_gold_tokens_present():
    assert exact_match("flask.app", "The answer is flask.app.")
    assert not exact_match("flask.app", "The answer is werkzeug.wrappers.")


def test_token_f1_partial_overlap():
    f1 = token_f1("flask app module", "flask module")
    assert 0.0 < f1 < 1.0


def test_token_f1_no_overlap_is_zero():
    assert token_f1("flask app", "werkzeug wrappers") == 0.0


def test_hallucination_rate_all_refused():
    assert hallucination_rate(["I don't know.", "Cannot answer this."]) == 0.0


def test_hallucination_rate_all_fabricated():
    assert hallucination_rate(["It inherits from BaseHandler."]) == 1.0


def test_hallucination_rate_mixed():
    rate = hallucination_rate(["I don't know.", "It inherits from BaseHandler."])
    assert rate == 0.5


def test_hallucination_rate_empty_list():
    assert hallucination_rate([]) == 0.0
