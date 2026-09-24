"""Phase 2, Part 2: a second, independent entity linker - a lightweight contrastive
fine-tune of the same base sentence-transformers model, trained on synthetic
(mention, entity) pairs generated from the KG's own node names/docstrings. No
external labeled data is used: positive pairs come from truncated/paraphrased
variants of entities already in this KG.

Kept alongside (not replacing) the original fuzzy string-matching linker in
retrieval.link_entities, so the two can be compared on ambiguous-mention cases.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("learned_linker")

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LINKER_TOP_K = 3
LINKER_MIN_SIM = 0.3


def _mention_variants(node: dict) -> list[str]:
    """Synthetic mention variants for one KG node, derived only from that node's own
    fields: truncated dotted-name suffixes (how someone would casually refer to a
    deeply-nested symbol) and templated paraphrases (how someone would describe it
    in a sentence rather than naming it verbatim)."""
    node_id = node["id"]
    name = node["name"]
    node_type = node["type"]
    parts = node_id.split(".")

    variants = {name, node_id}
    for k in (2, 3):
        if len(parts) >= k:
            variants.add(".".join(parts[-k:]))
    variants.add(f"the {node_type} {name}")
    variants.add(f"{name} {node_type}")
    if node.get("docstring"):
        first_sentence = node["docstring"].split(".")[0].strip()
        if first_sentence and len(first_sentence) < 120:
            variants.add(first_sentence)
    variants.discard("")
    return list(variants)


def build_training_pairs(nodes: list[dict], seed: int = 7, max_pairs: int | None = None) -> list[tuple[str, str]]:
    """(mention, positive entity text) pairs. Entity text reuses embeddings.node_text
    so the fine-tune specializes that exact representation for mention matching.
    max_pairs subsamples after shuffling - this is meant to be a SMALL contrastive
    fine-tune (per the spec), not an exhaustive one, and keeps wall-clock bounded.
    """
    from kg_grounded_qa.embeddings import node_text

    rng = random.Random(seed)
    pairs = [(mention, node_text(node)) for node in nodes for mention in _mention_variants(node)]
    rng.shuffle(pairs)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    return pairs


def finetune_linker(cache_dir: Path, out_dir: Path, epochs: int = 1, batch_size: int = 64, max_pairs: int = 4000) -> dict:
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    from kg_grounded_qa.embeddings import node_text

    nodes = json.loads((cache_dir / "nodes.json").read_text())
    pairs = build_training_pairs(nodes, max_pairs=max_pairs)
    logger.info(f"fine-tuning linker on {len(pairs)} synthetic (mention, entity) pairs from {len(nodes)} nodes")

    model = SentenceTransformer(BASE_MODEL)
    examples = [InputExample(texts=[m, e]) for m, e in pairs]
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(loader, loss)], epochs=epochs,
        warmup_steps=int(0.1 * len(loader) * epochs), show_progress_bar=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))

    node_ids = [n["id"] for n in nodes]
    texts = [node_text(n) for n in nodes]
    embs = model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    np.save(out_dir / "node_embeddings.npy", embs.astype(np.float32))
    (out_dir / "node_ids.json").write_text(json.dumps(node_ids))

    info = {"n_pairs": len(pairs), "n_nodes": len(nodes), "epochs": epochs, "base_model": BASE_MODEL}
    (out_dir / "linker_info.json").write_text(json.dumps(info, indent=2))
    logger.info(f"saved fine-tuned linker -> {out_dir}")
    return info


class LearnedLinker:
    """Callable with the same (question, index, query_text_emb, top_k) signature as
    retrieval.link_entities, so it drops straight into multihop_retrieve /
    embedding_only_retrieve as `link_fn=LearnedLinker(model_dir)`. Ignores the
    passed-in query_text_emb (computed with the BASE model) and re-embeds the
    question with its own fine-tuned model, comparing against its own node
    embeddings - the two linkers deliberately live in different embedding spaces.
    """

    def __init__(self, model_dir: Path):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(str(model_dir))
        self.node_ids = json.loads((Path(model_dir) / "node_ids.json").read_text())
        self.node_embs = np.load(Path(model_dir) / "node_embeddings.npy")

    def __call__(self, question: str, index, query_text_emb, top_k: int = LINKER_TOP_K):
        q_emb = self.model.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = self.node_embs @ q_emb
        top_idx = np.argsort(-sims)[:top_k]
        return [(self.node_ids[i], float(sims[i])) for i in top_idx if sims[i] >= LINKER_MIN_SIM]


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    cache_dir = repo_root / "data" / "cache"
    finetune_linker(cache_dir, cache_dir / "linker_model")
