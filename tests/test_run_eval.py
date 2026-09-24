import json

from kg_grounded_qa.run_eval import check_thresholds


def _write_thresholds(tmp_path, thresholds: dict):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(thresholds))
    return path


def test_check_thresholds_passes_when_metrics_meet_bounds(tmp_path):
    path = _write_thresholds(tmp_path, {
        "hallucination_rate": {"max": 0.15},
        "retrieval_recall_at_k": {"min": 0.55},
    })
    metrics = {"hallucination_rate": 0.067, "retrieval_recall_at_k": 0.679}
    assert check_thresholds(metrics, path) == []


def test_check_thresholds_fails_loudly_on_hallucination_regression(tmp_path):
    path = _write_thresholds(tmp_path, {"hallucination_rate": {"max": 0.15}})
    metrics = {"hallucination_rate": 0.40}  # regressed well past the 0.15 ceiling
    failures = check_thresholds(metrics, path)
    assert len(failures) == 1
    assert "hallucination_rate" in failures[0]


def test_check_thresholds_fails_loudly_on_recall_regression(tmp_path):
    path = _write_thresholds(tmp_path, {"retrieval_recall_at_k": {"min": 0.55}})
    metrics = {"retrieval_recall_at_k": 0.20}  # regressed well below the 0.55 floor
    failures = check_thresholds(metrics, path)
    assert len(failures) == 1
    assert "retrieval_recall_at_k" in failures[0]


def test_check_thresholds_reports_every_violation(tmp_path):
    path = _write_thresholds(tmp_path, {
        "hallucination_rate": {"max": 0.15},
        "retrieval_recall_at_k": {"min": 0.55},
        "answer_exact_match": {"min": 0.60},
    })
    metrics = {"hallucination_rate": 0.40, "retrieval_recall_at_k": 0.20, "answer_exact_match": 0.90}
    failures = check_thresholds(metrics, path)
    assert len(failures) == 2  # only the two that actually regressed


def test_check_thresholds_missing_file_is_noop(tmp_path):
    metrics = {"hallucination_rate": 0.99}
    assert check_thresholds(metrics, tmp_path / "does_not_exist.json") == []


def test_check_thresholds_ignores_metrics_not_in_thresholds_file(tmp_path):
    path = _write_thresholds(tmp_path, {"hallucination_rate": {"max": 0.15}})
    metrics = {"hallucination_rate": 0.05, "some_other_metric": -999}
    assert check_thresholds(metrics, path) == []
