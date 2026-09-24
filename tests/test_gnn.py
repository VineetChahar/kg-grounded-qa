import json
import random

import numpy as np
import pytest
import torch

from kg_grounded_qa.gnn import GAT, GCN, MODEL_CLASSES, GraphSAGE, make_split, train
from kg_grounded_qa.pyg_data import load_pyg_data


@pytest.mark.parametrize("model_cls", [GraphSAGE, GCN, GAT])
def test_forward_pass_shape(model_cls):
    model = model_cls(in_dim=16, hidden_dim=8, out_dim=4)
    x = torch.randn(10, 16)
    edge_index = torch.randint(0, 10, (2, 20))
    out = model(x, edge_index)
    assert out.shape == (10, 4)


@pytest.mark.parametrize("model_cls", [GraphSAGE, GCN, GAT])
def test_decode_shape(model_cls):
    model = model_cls(in_dim=16, hidden_dim=8, out_dim=4)
    z = torch.randn(10, 4)
    edge_index = torch.randint(0, 10, (2, 6))
    scores = model.decode(z, edge_index)
    assert scores.shape == (6,)


def test_gat_requires_hidden_dim_divisible_by_heads():
    with pytest.raises(AssertionError):
        GAT(in_dim=16, hidden_dim=10, out_dim=4, heads=4)


def _write_synthetic_cache(cache_dir, n_per_cluster=25, n_clusters=3):
    """Two dense clusters with almost no cross-cluster edges -> link prediction should be
    easy and clearly beat random (AUC > 0.5), which is the sanity check we want."""
    random.seed(0)
    nodes, edges = [], []
    node_id = 0
    cluster_members = []
    for c in range(n_clusters):
        members = []
        for _ in range(n_per_cluster):
            nid = f"n{node_id}"
            nodes.append({"id": nid, "type": "function", "name": nid, "package": f"pkg{c}",
                          "docstring": "", "signature": ""})
            members.append(nid)
            node_id += 1
        cluster_members.append(members)

    for members in cluster_members:
        for i, a in enumerate(members):
            for b in random.sample(members, min(6, len(members))):
                if a != b:
                    edges.append({"src": a, "dst": b, "type": "CALLS"})

    node_ids = [n["id"] for n in nodes]
    embs = np.random.default_rng(0).standard_normal((len(nodes), 8)).astype(np.float32)

    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "nodes.json").write_text(json.dumps(nodes))
    (cache_dir / "edges.json").write_text(json.dumps(edges))
    (cache_dir / "node_ids.json").write_text(json.dumps(node_ids))
    np.save(cache_dir / "node_embeddings.npy", embs)


@pytest.mark.parametrize("model_type", list(MODEL_CLASSES))
def test_link_prediction_training_beats_random(tmp_path, model_type):
    cache_dir = tmp_path / "cache"
    _write_synthetic_cache(cache_dir)
    # seed model weight init / negative sampling too, not just the synthetic graph
    # and the edge split - otherwise this test is flaky depending on what randomness
    # ran earlier in the same pytest session (order-dependent, not just this test).
    torch.manual_seed(0)
    metrics = train(cache_dir, model_type=model_type, device="cpu", epochs=30, save=False)
    assert metrics["val_auc"] > 0.5
    assert metrics["test_auc"] > 0.5
    assert metrics["val_ap"] > 0.0
    assert metrics["param_count"] > 0


def test_split_is_deterministic_across_repeated_calls(tmp_path):
    """The GNN ablation is only a controlled comparison if all three architectures
    see the exact same train/val/test edge split. Verify make_split() reproduces the
    IDENTICAL held-out positive edges every time (not just the same edge count)."""
    cache_dir = tmp_path / "cache"
    _write_synthetic_cache(cache_dir)
    data, _, _ = load_pyg_data(cache_dir)

    # perturb the RNG state in between, as different model constructors would
    torch.manual_seed(999)
    torch.randn(100)

    _, val1, test1 = make_split(data)
    torch.manual_seed(123)
    torch.randn(50)
    _, val2, test2 = make_split(data)

    assert torch.equal(val1.pos_edge_label_index, val2.pos_edge_label_index)
    assert torch.equal(test1.pos_edge_label_index, test2.pos_edge_label_index)
