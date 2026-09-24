"""Evaluation metrics: retrieval precision/recall@k, answer EM/F1, hallucination rate."""

from __future__ import annotations

import re
from collections import Counter

REFUSAL_PATTERNS = [
    "i don't know", "i do not know", "cannot answer", "can't answer",
    "not enough information", "no information", "not provided in the triples",
    "not contain", "does not appear", "unable to determine",
]


def is_refusal(answer: str) -> bool:
    a = answer.lower()
    return any(p in a for p in REFUSAL_PATTERNS)


def retrieval_precision_recall(gold_triples: list[list[str]], retrieved_triples: list[tuple[str, str, str]]) -> tuple[float, float]:
    gold_set = {tuple(t) for t in gold_triples}
    retrieved_set = set(retrieved_triples)
    if not retrieved_set:
        precision = 0.0
    else:
        precision = len(gold_set & retrieved_set) / len(retrieved_set)
    if not gold_set:
        recall = 1.0  # nothing to find
    else:
        recall = len(gold_set & retrieved_set) / len(gold_set)
    return precision, recall


_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def exact_match(gold_answer: str, generated_answer: str) -> bool:
    gold_tokens = set(_tokenize(gold_answer))
    gen_tokens = set(_tokenize(generated_answer))
    if not gold_tokens:
        return False
    # gold entity's identifying tokens (e.g. "app", "flask") must all appear in the answer
    return gold_tokens.issubset(gen_tokens)


def token_f1(gold_answer: str, generated_answer: str) -> float:
    gold_tokens = _tokenize(gold_answer)
    gen_tokens = _tokenize(generated_answer)
    if not gold_tokens or not gen_tokens:
        return 0.0
    gold_counts = Counter(gold_tokens)
    gen_counts = Counter(gen_tokens)
    overlap = sum((gold_counts & gen_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(gen_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def hallucination_rate(out_of_scope_answers: list[str]) -> float:
    if not out_of_scope_answers:
        return 0.0
    fabricated = sum(1 for a in out_of_scope_answers if not is_refusal(a))
    return fabricated / len(out_of_scope_answers)
