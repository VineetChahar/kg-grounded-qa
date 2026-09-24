"""Phase 2, Part 1a: train GraphSAGE, GCN, and GAT on the IDENTICAL edge split and
report link-prediction AUC/AP, wall-clock training time, and parameter count side by
side. Downstream QA-eval numbers per architecture are added separately by re-running
run_eval.py once per architecture's saved embeddings (see run_eval.py --gnn-arch).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from kg_grounded_qa.gnn import MODEL_CLASSES, train

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("gnn_ablation")


def run_ablation(cache_dir: Path) -> dict:
    results = {}
    for model_type in MODEL_CLASSES:
        logger.info(f"=== training {model_type} ===")
        metrics = train(cache_dir, model_type=model_type, save=True)
        results[model_type] = metrics

    (cache_dir / "gnn_ablation_results.json").write_text(json.dumps(results, indent=2))

    logger.info("\n%-8s %10s %10s %12s %14s %10s", "model", "test_auc", "test_ap", "train_time_s", "param_count", "val_auc")
    for model_type, m in results.items():
        logger.info(
            "%-8s %10.4f %10.4f %12.1f %14s %10.4f",
            model_type, m["test_auc"], m["test_ap"], m["train_time_s"], f"{m['param_count']:,}", m["val_auc"],
        )
    return results


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    run_ablation(repo_root / "data" / "cache")
