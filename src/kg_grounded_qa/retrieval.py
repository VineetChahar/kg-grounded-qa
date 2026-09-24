"""Multi-hop retrieval: entity-link a question into the KG, then beam-search outward,
ranking candidate paths by GNN structural similarity + text relevance, penalized by hop count.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TEXT_SIM_WEIGHT = 0.6
GNN_SIM_WEIGHT = 0.4
HOP_PENALTY = 0.15
BEAM_WIDTH = 6
MAX_HOPS = 3
TOP_K_TRIPLES = 12
ENTITY_LINK_MIN_SIM = 0.35


@dataclass
class Triple:
    src: str
    relation: str
    dst: str
    hop: int
    score: float

    def as_text(self) -> str:
        return f"{self.src} --{self.relation}--> {self.dst}"


class KGIndex:
    """In-memory index over the cached KG: adjacency, node metadata, text + GNN embeddings."""

    def __init__(self, cache_dir: Path, gnn_embeddings_path: Path | None = None):
        self.cache_dir = cache_dir
        nodes = json.loads((cache_dir / "nodes.json").read_text())
        edges = json.loads((cache_dir / "edges.json").read_text())
        node_ids = json.loads((cache_dir / "node_ids.json").read_text())

        self.node_by_id = {n["id"]: n for n in nodes}
        self.node_ids = node_ids
        self.idx_by_id = {nid: i for i, nid in enumerate(node_ids)}
        self.text_emb = np.load(cache_dir / "node_embeddings.npy")
        gnn_path = gnn_embeddings_path or (cache_dir / "gnn_embeddings.npy")
        self.gnn_emb = np.load(gnn_path) if gnn_path.exists() else None

        # adjacency: node -> list of (neighbor, relation, direction) ; direction 'out'|'in'
        self.adj: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        self.edges = edges
        for e in edges:
            self.adj[e["src"]].append((e["dst"], e["type"], "out"))
            self.adj[e["dst"]].append((e["src"], e["type"], "in"))

    def text_vec(self, node_id: str) -> np.ndarray:
        return self.text_emb[self.idx_by_id[node_id]]

    def gnn_vec(self, node_id: str) -> np.ndarray | None:
        if self.gnn_emb is None:
            return None
        return self.gnn_emb[self.idx_by_id[node_id]]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
_BACKTICK_RE = re.compile(r"`([^`]+)`")


def link_entities(question: str, index: KGIndex, query_text_emb: np.ndarray, top_k: int = 3) -> list[tuple[str, float]]:
    """Find seed KG nodes for a question: exact/substring name match first, embedding fallback.

    Phase 2 error-analysis fix: if the question backtick-quotes its entity mention (as
    every templated eval question does, e.g. "Name a class ... that `werkzeug.utils`
    imports."), match ONLY against the quoted span(s) instead of every token in the
    sentence. Without this, generic instruction words like "Name" collide with common
    attribute names like `.name` (present on dozens of unrelated classes), and the
    length-based tie-break below lets those long, irrelevant node ids crowd the
    correct short entity out of top-k entirely - see README error analysis.
    """
    quoted_spans = _BACKTICK_RE.findall(question)
    candidates: dict[str, float] = {}

    tokens = set(quoted_spans) if quoted_spans else set(_TOKEN_RE.findall(question))
    for node_id, node in index.node_by_id.items():
        haystacks = [node_id.lower(), node["name"].lower()]
        for tok in tokens:
            tok_l = tok.lower()
            if len(tok_l) < 3:
                continue
            if tok_l == node["name"].lower() or tok_l == node_id.lower():
                candidates[node_id] = max(candidates.get(node_id, 0.0), 1.0 + len(node_id) / 1000)
            elif node_id.lower() in tok_l or tok_l in node_id.lower():
                candidates[node_id] = max(candidates.get(node_id, 0.0), 0.9 + len(node_id) / 1000)

    if not candidates:
        sims = index.text_emb @ query_text_emb / (
            np.linalg.norm(index.text_emb, axis=1) * (np.linalg.norm(query_text_emb) or 1e-9) + 1e-9
        )
        top_idx = np.argsort(-sims)[:top_k]
        for i in top_idx:
            if sims[i] >= ENTITY_LINK_MIN_SIM:
                candidates[index.node_ids[i]] = float(sims[i])

    ranked = sorted(candidates.items(), key=lambda kv: -kv[1])[:top_k]
    return ranked


def multihop_retrieve(
    question: str,
    index: KGIndex,
    query_text_emb: np.ndarray,
    max_hops: int = MAX_HOPS,
    beam_width: int = BEAM_WIDTH,
    top_k_triples: int = TOP_K_TRIPLES,
    link_fn=link_entities,
) -> tuple[list[tuple[str, float]], list[Triple]]:
    """Entity-link the question, then beam-search outward ranking by GNN + text similarity."""
    seeds = link_fn(question, index, query_text_emb)
    if not seeds:
        return [], []

    all_triples: dict[tuple[str, str, str], Triple] = {}
    visited_global: set[str] = {s for s, _ in seeds}

    for seed_id, seed_score in seeds:
        seed_gnn = index.gnn_vec(seed_id)
        frontier = [seed_id]
        visited = {seed_id}

        for hop in range(1, max_hops + 1):
            scored_next = []
            for node_id in frontier:
                for neighbor, relation, direction in index.adj.get(node_id, []):
                    if neighbor in visited:
                        continue
                    n_gnn = index.gnn_vec(neighbor)
                    gnn_sim = _cosine(seed_gnn, n_gnn) if seed_gnn is not None and n_gnn is not None else 0.0
                    text_sim = _cosine(index.text_vec(neighbor), query_text_emb)
                    score = TEXT_SIM_WEIGHT * text_sim + GNN_SIM_WEIGHT * gnn_sim - HOP_PENALTY * hop
                    scored_next.append((score, node_id, neighbor, relation, direction))

            scored_next.sort(key=lambda t: -t[0])
            top_next = scored_next[:beam_width]

            for score, node_id, neighbor, relation, direction in top_next:
                if direction == "out":
                    triple = Triple(node_id, relation, neighbor, hop, score)
                else:
                    triple = Triple(neighbor, relation, node_id, hop, score)
                key = (triple.src, triple.relation, triple.dst)
                if key not in all_triples or all_triples[key].score < score:
                    all_triples[key] = triple
                visited.add(neighbor)
                visited_global.add(neighbor)

            frontier = [n for *_ , n, _, _ in top_next] if top_next else []
            if not frontier:
                break

    ranked_triples = sorted(all_triples.values(), key=lambda t: -t.score)[:top_k_triples]
    return seeds, ranked_triples


MAX_CANDIDATES_PER_SEED = 3000


def embedding_only_retrieve(
    question: str,
    index: KGIndex,
    query_text_emb: np.ndarray,
    max_hops: int = MAX_HOPS,
    top_k_triples: int = TOP_K_TRIPLES,
    link_fn=link_entities,
) -> tuple[list[tuple[str, float]], list[Triple]]:
    """Phase 2 ablation baseline (Part 1b): identical entity linking, but no GNN and no
    beam search - exhaustively expand ALL neighbors up to max_hops (capped at
    MAX_CANDIDATES_PER_SEED to keep runtime bounded on high-degree hub nodes), then
    rank candidate triples purely by sentence-transformers text-embedding cosine
    similarity to the question. No hop penalty, no structural signal at all.
    """
    seeds = link_fn(question, index, query_text_emb)
    if not seeds:
        return [], []

    all_triples: dict[tuple[str, str, str], Triple] = {}

    for seed_id, seed_score in seeds:
        frontier = [seed_id]
        visited = {seed_id}

        for hop in range(1, max_hops + 1):
            next_frontier = []
            for node_id in frontier:
                for neighbor, relation, direction in index.adj.get(node_id, []):
                    if neighbor in visited:
                        continue
                    if len(visited) >= MAX_CANDIDATES_PER_SEED:
                        break
                    visited.add(neighbor)
                    next_frontier.append(neighbor)

                    text_sim = _cosine(index.text_vec(neighbor), query_text_emb)
                    if direction == "out":
                        triple = Triple(node_id, relation, neighbor, hop, text_sim)
                    else:
                        triple = Triple(neighbor, relation, node_id, hop, text_sim)
                    key = (triple.src, triple.relation, triple.dst)
                    if key not in all_triples or all_triples[key].score < text_sim:
                        all_triples[key] = triple

            frontier = next_frontier
            if not frontier:
                break

    ranked_triples = sorted(all_triples.values(), key=lambda t: -t.score)[:top_k_triples]
    return seeds, ranked_triples
