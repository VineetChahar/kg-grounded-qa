"""Phase 2, Part 2: compare the original fuzzy-matching linker against the learned
(fine-tuned bi-encoder) linker on the hand-curated ambiguity eval set - both on
linking accuracy in isolation, and on downstream hallucination rate once each feeds
the full QA pipeline. A linker that's more accurate in isolation but doesn't move
the downstream hallucination number is itself a reportable finding (see README).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from kg_grounded_qa.embeddings import embed_query
from kg_grounded_qa.evaluation import hallucination_rate
from kg_grounded_qa.learned_linker import LearnedLinker
from kg_grounded_qa.qa_pipeline import QAPipeline
from kg_grounded_qa.retrieval import KGIndex, link_entities

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("compare_linkers")


def linking_accuracy(link_fn, index: KGIndex, examples: list[dict]) -> tuple[float, list[dict]]:
    correct = 0
    rows = []
    for ex in examples:
        q_emb = embed_query(ex["mention_context"])
        seeds = link_fn(ex["mention_context"], index, q_emb)
        predicted = seeds[0][0] if seeds else None
        is_correct = predicted == ex["gold_entity"]
        correct += int(is_correct)
        rows.append({**ex, "predicted": predicted, "correct": is_correct})
    return correct / len(examples), rows


def downstream_hallucination(pipeline: QAPipeline, questions: list[str]) -> tuple[float, list[str]]:
    answers = [pipeline.answer(q).answer for q in questions]
    return hallucination_rate(answers), answers


def run(cache_dir: Path, repo_root: Path) -> dict:
    linking_set = json.loads((repo_root / "data" / "linking_eval_set.json").read_text())
    main_eval_set = json.loads((repo_root / "data" / "eval_set.json").read_text())
    oos_questions = [ex["question"] for ex in main_eval_set if ex["category"] == "out_of_scope"]
    linking_oos_questions = [ex["mention_context"] for ex in linking_set if ex["gold_entity"] is None]
    hallucination_probe_questions = oos_questions + linking_oos_questions

    index = KGIndex(cache_dir)
    learned_linker = LearnedLinker(cache_dir / "linker_model")

    logger.info("scoring linking accuracy: fuzzy linker")
    fuzzy_acc, fuzzy_rows = linking_accuracy(link_entities, index, linking_set)
    logger.info("scoring linking accuracy: learned linker")
    learned_acc, learned_rows = linking_accuracy(learned_linker, index, linking_set)

    fuzzy_pipeline = QAPipeline(cache_dir, link_fn=link_entities)
    learned_pipeline = QAPipeline(cache_dir, link_fn=learned_linker)

    logger.info(f"measuring downstream hallucination rate on {len(hallucination_probe_questions)} probe questions: fuzzy linker")
    fuzzy_halluc, fuzzy_answers = downstream_hallucination(fuzzy_pipeline, hallucination_probe_questions)
    logger.info(f"measuring downstream hallucination rate on {len(hallucination_probe_questions)} probe questions: learned linker")
    learned_halluc, learned_answers = downstream_hallucination(learned_pipeline, hallucination_probe_questions)

    disagreement = None
    for f_row, l_row in zip(fuzzy_rows, learned_rows):
        if f_row["predicted"] != l_row["predicted"]:
            disagreement = {
                "mention_context": f_row["mention_context"],
                "gold_entity": f_row["gold_entity"],
                "fuzzy_predicted": f_row["predicted"], "fuzzy_correct": f_row["correct"],
                "learned_predicted": l_row["predicted"], "learned_correct": l_row["correct"],
            }
            break

    results = {
        "n_linking_examples": len(linking_set),
        "n_hallucination_probe_questions": len(hallucination_probe_questions),
        "fuzzy_linker": {"linking_accuracy": fuzzy_acc, "downstream_hallucination_rate": fuzzy_halluc},
        "learned_linker": {"linking_accuracy": learned_acc, "downstream_hallucination_rate": learned_halluc},
        "one_disagreement_case": disagreement,
    }

    (repo_root / "data" / "linker_comparison.json").write_text(json.dumps(results, indent=2))
    (repo_root / "data" / "linker_comparison_rows.json").write_text(
        json.dumps({"fuzzy": fuzzy_rows, "learned": learned_rows,
                    "fuzzy_hallucination_answers": fuzzy_answers, "learned_hallucination_answers": learned_answers}, indent=2)
    )

    logger.info("\n" + json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    run(repo_root / "data" / "cache", repo_root)
