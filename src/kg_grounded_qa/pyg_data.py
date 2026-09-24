"""Convert the cached KG (nodes.json/edges.json + node embeddings) into a PyTorch Geometric Data object."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

NODE_TYPES = ["package", "module", "class", "function"]


def load_pyg_data(cache_dir: Path) -> tuple[Data, list[str], dict[str, int]]:
    nodes = json.loads((cache_dir / "nodes.json").read_text())
    edges = json.loads((cache_dir / "edges.json").read_text())
    node_ids = json.loads((cache_dir / "node_ids.json").read_text())
    embs = np.load(cache_dir / "node_embeddings.npy")

    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    node_by_id = {n["id"]: n for n in nodes}

    # node features: text embedding (384) concat one-hot node type (4) = 388-dim
    type_onehot = np.zeros((len(node_ids), len(NODE_TYPES)), dtype=np.float32)
    for nid, idx in id_to_idx.items():
        t = node_by_id[nid]["type"]
        type_onehot[idx, NODE_TYPES.index(t)] = 1.0
    x = np.concatenate([embs, type_onehot], axis=1)

    src = [id_to_idx[e["src"]] for e in edges]
    dst = [id_to_idx[e["dst"]] for e in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    data = Data(x=torch.tensor(x, dtype=torch.float32), edge_index=edge_index)
    data.num_nodes = len(node_ids)
    return data, node_ids, id_to_idx


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    data, node_ids, id_to_idx = load_pyg_data(repo_root / "data" / "cache")
    print(data)
