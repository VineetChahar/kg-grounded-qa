"""GNN encoders trained on self-supervised link prediction over the code KG.

Three architectures are supported for the Phase 2 ablation (GraphSAGE, GCN, GAT), all
sharing the same dot-product decoder so the comparison isolates the ENCODER choice.

No external labels are needed: we predict "does this edge exist" using a disjoint
train/val/test split over edges (RandomLinkSplit), with the val/test edges held out
of message passing entirely so the model can't see them during embedding computation.
The split is reseeded to the same SPLIT_SEED immediately before every call to
RandomLinkSplit, so all three architectures are trained/evaluated on the IDENTICAL
edge split - required for the ablation to be a controlled comparison rather than
three runs on three different random splits.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import average_precision_score, roc_auc_score

from kg_grounded_qa.pyg_data import load_pyg_data

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("gnn")

HIDDEN_DIM = 128
OUT_DIM = 64
EPOCHS = 100
LR = 0.01
GAT_HEADS = 4
SPLIT_SEED = 42


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = HIDDEN_DIM, out_dim: int = OUT_DIM):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.conv2(h, edge_index)
        return h

    @staticmethod
    def decode(z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)


class GCN(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = HIDDEN_DIM, out_dim: int = OUT_DIM):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.conv2(h, edge_index)
        return h

    decode = staticmethod(GraphSAGE.decode)


class GAT(torch.nn.Module):
    """2-layer GAT. Head count is a documented hyperparameter (GAT_HEADS=4): layer 1
    uses 4 attention heads concatenated back to hidden_dim; layer 2 uses a single
    head (concat=False) to produce the final out_dim, matching the other encoders'
    output shape so the downstream decoder/retrieval code is architecture-agnostic.
    """

    def __init__(self, in_dim: int, hidden_dim: int = HIDDEN_DIM, out_dim: int = OUT_DIM, heads: int = GAT_HEADS):
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        self.conv1 = GATConv(in_dim, hidden_dim // heads, heads=heads, dropout=0.2)
        self.conv2 = GATConv(hidden_dim, out_dim, heads=1, concat=False, dropout=0.2)

    def forward(self, x, edge_index):
        h = F.dropout(x, p=0.2, training=self.training)
        h = F.elu(self.conv1(h, edge_index))
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.conv2(h, edge_index)
        return h

    decode = staticmethod(GraphSAGE.decode)


MODEL_CLASSES = {"sage": GraphSAGE, "gcn": GCN, "gat": GAT}


def _artifact_paths(cache_dir: Path, model_type: str) -> dict[str, Path]:
    # "sage" keeps the canonical (unsuffixed) filenames for backward compatibility
    # with retrieval.py/qa_pipeline.py's defaults, since it remains the production
    # choice unless the ablation says otherwise.
    suffix = "" if model_type == "sage" else f"_{model_type}"
    return {
        "embeddings": cache_dir / f"gnn_embeddings{suffix}.npy",
        "model": cache_dir / f"gnn_model{suffix}.pt",
        "metrics": cache_dir / f"gnn_metrics{suffix}.json",
    }


def make_split(data, seed: int = SPLIT_SEED):
    """Deterministic edge split: reseeding to `seed` immediately before splitting
    means every caller (every architecture in the ablation) gets the IDENTICAL
    train/val/test edges, regardless of what randomness ran before this call.
    """
    torch.manual_seed(seed)
    splitter = RandomLinkSplit(
        num_val=0.1, num_test=0.1, is_undirected=False,
        add_negative_train_samples=False, split_labels=True,
    )
    return splitter(data)


def train(
    cache_dir: Path,
    model_type: str = "sage",
    device: str | None = None,
    epochs: int = EPOCHS,
    save: bool = True,
    heads: int = GAT_HEADS,
) -> dict:
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    data, node_ids, id_to_idx = load_pyg_data(cache_dir)

    train_data, val_data, test_data = make_split(data)

    train_data, val_data, test_data = train_data.to(device), val_data.to(device), test_data.to(device)
    model_cls = MODEL_CLASSES[model_type]
    model_kwargs = {"heads": heads} if model_type == "gat" else {}
    model = model_cls(in_dim=data.num_node_features, **model_kwargs).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_auc = 0.0
    best_val_ap = 0.0
    best_state = None

    train_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model(train_data.x, train_data.edge_index)

        pos_edge_index = train_data.pos_edge_label_index
        neg_edge_index = torch.randint(
            0, train_data.num_nodes, pos_edge_index.shape, dtype=torch.long, device=device
        )
        edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        edge_label = torch.cat([
            torch.ones(pos_edge_index.size(1), device=device),
            torch.zeros(neg_edge_index.size(1), device=device),
        ])

        out = model.decode(z, edge_label_index)
        loss = F.binary_cross_entropy_with_logits(out, edge_label)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == epochs or epoch == 1:
            val_auc, val_ap = evaluate(model, train_data, val_data, device)
            logger.info(f"[{model_type}] epoch {epoch:3d} | loss {loss.item():.4f} | val AUC {val_auc:.4f} | val AP {val_ap:.4f}")
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_val_ap = val_ap
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    train_time_s = time.time() - train_start

    model.load_state_dict(best_state)
    test_auc, test_ap = evaluate(model, train_data, test_data, device)
    logger.info(f"[{model_type}] best val AUC {best_val_auc:.4f} | test AUC {test_auc:.4f} | test AP {test_ap:.4f} | {train_time_s:.1f}s | {param_count:,} params")

    # final full-graph embeddings, computed with ALL edges except the held-out
    # val/test supervision edges (train_data.edge_index = message-passing edges only)
    model.eval()
    with torch.no_grad():
        final_z = model(train_data.x, train_data.edge_index).cpu().numpy()

    metrics = {
        "model_type": model_type,
        "val_auc": best_val_auc,
        "val_ap": best_val_ap,
        "test_auc": test_auc,
        "test_ap": test_ap,
        "train_time_s": train_time_s,
        "param_count": param_count,
        "epochs": epochs,
        "hidden_dim": HIDDEN_DIM,
        "out_dim": OUT_DIM,
        "heads": heads if model_type == "gat" else None,
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)),
        "device": device,
    }

    if save:
        import numpy as np
        paths = _artifact_paths(cache_dir, model_type)
        np.save(paths["embeddings"], final_z)
        (cache_dir / "node_ids.json").write_text(json.dumps(node_ids))
        torch.save(model.state_dict(), paths["model"])
        paths["metrics"].write_text(json.dumps(metrics, indent=2))

    return metrics


@torch.no_grad()
def evaluate(model, message_passing_data, eval_data, device: str) -> tuple[float, float]:
    model.eval()
    z = model(message_passing_data.x, message_passing_data.edge_index)
    pos_index = eval_data.pos_edge_label_index
    neg_index = eval_data.neg_edge_label_index
    edge_label_index = torch.cat([pos_index, neg_index], dim=1)
    edge_label = torch.cat([
        torch.ones(pos_index.size(1)), torch.zeros(neg_index.size(1)),
    ]).cpu().numpy()
    scores = torch.sigmoid(model.decode(z, edge_label_index)).cpu().numpy()
    return roc_auc_score(edge_label, scores), average_precision_score(edge_label, scores)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_CLASSES), default="sage")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    train(repo_root / "data" / "cache", model_type=args.model)
