"""End-to-end pipeline: question -> entity linking -> multi-hop retrieval -> grounded generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kg_grounded_qa.embeddings import embed_query
from kg_grounded_qa.generation import generate_answer
from kg_grounded_qa.retrieval import KGIndex, Triple, embedding_only_retrieve, link_entities, multihop_retrieve

RETRIEVAL_METHODS = {
    "gnn_beam": multihop_retrieve,
    "embedding_only": embedding_only_retrieve,
}


@dataclass
class QAResult:
    question: str
    answer: str
    seeds: list[tuple[str, float]]
    triples: list[Triple]


class QAPipeline:
    def __init__(
        self,
        cache_dir: Path,
        gnn_embeddings_path: Path | None = None,
        retrieval_method: str = "gnn_beam",
        link_fn=link_entities,
    ):
        self.index = KGIndex(cache_dir, gnn_embeddings_path=gnn_embeddings_path)
        self.retrieve_fn = RETRIEVAL_METHODS[retrieval_method]
        self.link_fn = link_fn

    def answer(self, question: str) -> QAResult:
        q_emb = embed_query(question)
        seeds, triples = self.retrieve_fn(question, self.index, q_emb, link_fn=self.link_fn)
        answer = generate_answer(question, triples)
        return QAResult(question=question, answer=answer, seeds=seeds, triples=triples)


if __name__ == "__main__":
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    pipeline = QAPipeline(repo_root / "data" / "cache")
    q = sys.argv[1] if len(sys.argv) > 1 else "What does flask.app import?"
    result = pipeline.answer(q)
    print(f"Q: {result.question}")
    print(f"Seeds: {result.seeds}")
    print("Retrieved triples:")
    for t in result.triples:
        print(f"  {t.as_text()}  (score={t.score:.3f})")
    print(f"A: {result.answer}")
