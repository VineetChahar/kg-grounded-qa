"""Text embeddings for KG nodes, via a local sentence-transformers model (no API key/login)."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("embeddings")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=4)
def _load_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    logger.info(f"loading {model_name} (cached for subsequent calls)")
    return SentenceTransformer(model_name)


def node_text(node: dict) -> str:
    """Compact text representation of a node used as the embedding input."""
    parts = [node["type"], node["id"]]
    if node.get("signature"):
        parts.append(node["signature"])
    if node.get("docstring"):
        parts.append(node["docstring"])
    return " | ".join(parts)


def compute_embeddings(nodes_path: Path, out_dir: Path, model_name: str = MODEL_NAME) -> None:
    nodes = json.loads(nodes_path.read_text())
    ids = [n["id"] for n in nodes]
    texts = [node_text(n) for n in nodes]

    logger.info(f"embedding {len(texts)} nodes with {model_name}")
    model = _load_model(model_name)
    embs = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "node_embeddings.npy", embs.astype(np.float32))
    (out_dir / "node_ids.json").write_text(json.dumps(ids))
    logger.info(f"saved embeddings {embs.shape} -> {out_dir / 'node_embeddings.npy'}")


def embed_query(text: str, model_name: str = MODEL_NAME) -> np.ndarray:
    model = _load_model(model_name)
    return model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    cache_dir = repo_root / "data" / "cache"
    compute_embeddings(cache_dir / "nodes.json", cache_dir)
