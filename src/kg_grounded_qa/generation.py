"""Grounded answer generation via a local Ollama model (no cloud API, no key)."""

from __future__ import annotations

import os

import requests

from kg_grounded_qa.retrieval import Triple

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL = f"{OLLAMA_HOST}/api/generate"
MODEL = "qwen2.5:3b-instruct"

SYSTEM_PROMPT = (
    "You are a precise assistant that answers questions ONLY using the provided knowledge-graph "
    "triples. Each triple has the form `subject --RELATION--> object`.\n"
    "Rules:\n"
    "1. Use only information present in the triples below. Do not use outside knowledge.\n"
    "2. If the triples do not contain enough information to answer, respond exactly: I don't know.\n"
    "3. Keep the answer to one short sentence, naming the specific entity/entities asked for.\n"
)


def serialize_triples(triples: list[Triple]) -> str:
    if not triples:
        return "(no relevant triples found)"
    return "\n".join(f"- {t.as_text()}" for t in triples)


def build_prompt(question: str, triples: list[Triple]) -> str:
    return (
        f"{SYSTEM_PROMPT}\n"
        f"Triples:\n{serialize_triples(triples)}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def generate_answer(question: str, triples: list[Triple], model: str = MODEL, timeout: float = 60.0) -> str:
    prompt = build_prompt(question, triples)
    resp = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.0}},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()
