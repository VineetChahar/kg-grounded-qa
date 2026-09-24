"""Run the full eval harness end to end and report real numbers.

Also acts as a lightweight regression check: if `data/thresholds.json` exists, exits
non-zero when a metric regresses past its configured floor/ceiling.

Supports Phase 2 ablation runs via --gnn-arch / --retrieval-method / --linker: each
non-default combination writes to its own suffixed output files instead of
overwriting the canonical (production-config) eval_results.json / eval_metrics.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from kg_grounded_qa.evaluation import (
    exact_match, hallucination_rate, retrieval_precision_recall, token_f1,
)
from kg_grounded_qa.qa_pipeline import QAPipeline

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("run_eval")


def run(pipeline: QAPipeline, eval_set_path: Path, gnn_metrics_path: Path | None = None) -> dict:
    eval_set = json.loads(eval_set_path.read_text())

    per_example = []
    for i, ex in enumerate(eval_set):
        t0 = time.time()
        result = pipeline.answer(ex["question"])
        retrieved = [(t.src, t.relation, t.dst) for t in result.triples]
        dt = time.time() - t0

        row = {
            "question": ex["question"],
            "category": ex["category"],
            "hops": ex["hops"],
            "gold_answer": ex["gold_answer"],
            "generated_answer": result.answer,
            "seeds": [s for s, _ in result.seeds],
            "retrieved_triples": retrieved,
            "latency_s": round(dt, 2),
        }

        if ex["category"] == "answerable":
            precision, recall = retrieval_precision_recall(ex["gold_triples"], retrieved)
            row["retrieval_precision"] = precision
            row["retrieval_recall"] = recall
            row["exact_match"] = exact_match(ex["gold_answer"], result.answer)
            row["f1"] = token_f1(ex["gold_answer"], result.answer)

        per_example.append(row)
        logger.info(f"[{i+1}/{len(eval_set)}] {ex['category']:12s} hop={ex['hops']} em={row.get('exact_match', '-')}  {ex['question'][:70]}")

    answerable = [r for r in per_example if r["category"] == "answerable"]
    out_of_scope = [r for r in per_example if r["category"] == "out_of_scope"]

    metrics = {
        "n_questions": len(eval_set),
        "n_answerable": len(answerable),
        "n_out_of_scope": len(out_of_scope),
        "retrieval_precision_at_k": _avg(answerable, "retrieval_precision"),
        "retrieval_recall_at_k": _avg(answerable, "retrieval_recall"),
        "answer_exact_match": _avg(answerable, "exact_match"),
        "answer_f1": _avg(answerable, "f1"),
        "hallucination_rate": hallucination_rate([r["generated_answer"] for r in out_of_scope]),
        "avg_latency_s": _avg(per_example, "latency_s"),
    }

    if gnn_metrics_path and gnn_metrics_path.exists():
        gnn_metrics = json.loads(gnn_metrics_path.read_text())
        metrics["gnn_link_prediction_val_auc"] = gnn_metrics["val_auc"]
        metrics["gnn_link_prediction_test_auc"] = gnn_metrics["test_auc"]

    return {"metrics": metrics, "per_example": per_example}


def _avg(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(r[key]) for r in rows) / len(rows)


def check_thresholds(metrics: dict, thresholds_path: Path) -> list[str]:
    if not thresholds_path.exists():
        return []
    thresholds = json.loads(thresholds_path.read_text())
    failures = []
    for key, bound in thresholds.items():
        if key not in metrics:
            continue
        value = metrics[key]
        if "min" in bound and value < bound["min"]:
            failures.append(f"{key}={value:.4f} below min {bound['min']}")
        if "max" in bound and value > bound["max"]:
            failures.append(f"{key}={value:.4f} above max {bound['max']}")
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-on-regress", action="store_true")
    parser.add_argument("--gnn-arch", choices=["sage", "gcn", "gat"], default="sage")
    parser.add_argument("--retrieval-method", choices=["gnn_beam", "embedding_only"], default="gnn_beam")
    parser.add_argument("--eval-set", default=None, help="path to eval set JSON (default: data/eval_set.json)")
    parser.add_argument("--tag", default=None, help="override the output-file suffix (default: derived from args)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cache_dir = repo_root / "data" / "cache"

    is_default_config = args.gnn_arch == "sage" and args.retrieval_method == "gnn_beam"
    tag = args.tag or ("" if is_default_config else f"_{args.gnn_arch}_{args.retrieval_method}")

    gnn_suffix = "" if args.gnn_arch == "sage" else f"_{args.gnn_arch}"
    gnn_embeddings_path = cache_dir / f"gnn_embeddings{gnn_suffix}.npy"
    gnn_metrics_path = cache_dir / f"gnn_metrics{gnn_suffix}.json"

    pipeline = QAPipeline(cache_dir, gnn_embeddings_path=gnn_embeddings_path, retrieval_method=args.retrieval_method)
    eval_set_path = Path(args.eval_set) if args.eval_set else repo_root / "data" / "eval_set.json"
    results = run(pipeline, eval_set_path, gnn_metrics_path=gnn_metrics_path)

    (repo_root / "data" / f"eval_results{tag}.json").write_text(json.dumps(results["per_example"], indent=2))
    (repo_root / "data" / f"eval_metrics{tag}.json").write_text(json.dumps(results["metrics"], indent=2))

    print(f"\n=== EVAL METRICS (gnn={args.gnn_arch}, retrieval={args.retrieval_method}) ===")
    print(json.dumps(results["metrics"], indent=2))

    if is_default_config:
        failures = check_thresholds(results["metrics"], repo_root / "data" / "thresholds.json")
        if failures:
            print("\n=== REGRESSION CHECK FAILED ===")
            for f in failures:
                print(f"  - {f}")
            if args.fail_on_regress:
                sys.exit(1)
        else:
            print("\nregression check: OK (or no thresholds.json configured)")
