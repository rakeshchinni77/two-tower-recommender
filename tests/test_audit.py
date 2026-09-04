"""Final Phase 13 audit tests — artifact presence, evaluation schema, threshold alignment, and gate conditions."""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import (
    PROCESSED_DATA_DIR,
    ARTIFACTS_DIR,
    THRESHOLD_N,
)

# Derive project root and results dir from the already-exported PROCESSED_DATA_DIR
# PROCESSED_DATA_DIR = <root>/data/processed  →  root = .parent.parent
BASE_DIR = PROCESSED_DATA_DIR.parent.parent
RESULTS_DIR = BASE_DIR / "results"


# ─── Helpers ────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ─── 1. Required file artifacts ─────────────────────────────────────────────

class TestRequiredArtifacts:
    """Verify that every required project artifact is present on disk."""

    def test_processed_data_files(self):
        required = [
            "train.csv",
            "test_warm.csv",
            "test_cold.csv",
            "items.csv",
            "item_embeddings.npy",
            "user_mapping.json",
            "item_mapping.json",
        ]
        for name in required:
            path = PROCESSED_DATA_DIR / name
            assert path.exists(), f"Missing required processed file: {path}"

    def test_model_artifact(self):
        artifact = ARTIFACTS_DIR / "two_tower.pt"
        assert artifact.exists(), f"Missing model artifact: {artifact}"

    def test_results_files(self):
        required = [
            "metrics.json",
            "threshold_sweep.json",
            "evaluation_report.json",
        ]
        for name in required:
            path = RESULTS_DIR / name
            assert path.exists(), f"Missing results file: {path}"

    def test_project_root_files(self):
        root = BASE_DIR
        required = [
            "Dockerfile",
            "docker-compose.yml",
            "README.md",
            "requirements.txt",
            ".env.example",
            ".gitignore",
            "submission.json",
        ]
        for name in required:
            path = root / name
            assert path.exists(), f"Missing root file: {path}"


# ─── 2. Evaluation JSON schema validation ───────────────────────────────────

class TestEvaluationJsonSchema:
    """Validate that metrics.json follows the required schema exactly."""

    @pytest.fixture
    def metrics(self):
        return _load_json(RESULTS_DIR / "metrics.json")


    def test_threshold_n_key_present(self, metrics):
        assert "threshold_N" in metrics, "metrics.json missing 'threshold_N'"

    def test_threshold_n_value(self, metrics):
        assert metrics["threshold_N"] == 10, (
            f"Expected threshold_N=10, got {metrics['threshold_N']}"
        )

    def test_slices_key_present(self, metrics):
        assert "slices" in metrics

    def test_required_slices(self, metrics):
        for slice_name in ("warm_users", "cold_users"):
            assert slice_name in metrics["slices"], f"Missing slice: {slice_name}"

    def test_required_models_per_slice(self, metrics):
        for slice_name in ("warm_users", "cold_users"):
            for model in ("popularity", "content", "two_tower", "hybrid"):
                assert model in metrics["slices"][slice_name], (
                    f"Missing model '{model}' in slice '{slice_name}'"
                )

    def test_recall_ndcg_fields_are_numbers(self, metrics):
        for slice_name in ("warm_users", "cold_users"):
            for model in ("popularity", "content", "two_tower", "hybrid"):
                entry = metrics["slices"][slice_name][model]
                assert "recall_10" in entry
                assert "ndcg_10" in entry
                assert isinstance(entry["recall_10"], (int, float))
                assert isinstance(entry["ndcg_10"], (int, float))

    def test_all_metrics_are_finite_and_in_range(self, metrics):
        for slice_name in ("warm_users", "cold_users"):
            for model in ("popularity", "content", "two_tower", "hybrid"):
                entry = metrics["slices"][slice_name][model]
                for key in ("recall_10", "ndcg_10"):
                    val = entry[key]
                    assert np.isfinite(val), f"{slice_name}/{model}/{key} is not finite"
                    assert 0.0 <= val <= 1.0, (
                        f"{slice_name}/{model}/{key}={val} out of [0,1] range"
                    )


# ─── 3. Threshold alignment ─────────────────────────────────────────────────

class TestThresholdAlignment:
    """Verify threshold_N is consistent across all project artifacts."""

    def test_submission_json_threshold(self):
        sub = _load_json(BASE_DIR / "submission.json")
        assert "threshold_N" in sub, "submission.json missing 'threshold_N'"
        assert sub["threshold_N"] == 10, (
            f"submission.json threshold_N={sub['threshold_N']}, expected 10"
        )

    def test_metrics_json_threshold(self):
        m = _load_json(RESULTS_DIR / "metrics.json")
        assert m["threshold_N"] == 10

    def test_threshold_n_matches_config(self):
        assert THRESHOLD_N == 10, (
            f"Config THRESHOLD_N={THRESHOLD_N}, expected 10"
        )

    def test_submission_and_metrics_threshold_match(self):
        sub = _load_json(BASE_DIR / "submission.json")
        m = _load_json(RESULTS_DIR / "metrics.json")
        assert sub["threshold_N"] == m["threshold_N"], (
            f"submission.json threshold_N={sub['threshold_N']} != "
            f"metrics.json threshold_N={m['threshold_N']}"
        )

    def test_threshold_sweep_selected_threshold(self):
        sweep = _load_json(RESULTS_DIR / "threshold_sweep.json")
        assert "selected_threshold_N" in sweep, (
            "threshold_sweep.json missing 'selected_threshold_N'"
        )
        assert sweep["selected_threshold_N"] == 10


# ─── 4. Evaluation gate conditions ──────────────────────────────────────────

class TestEvaluationGateConditions:
    """Verify the two mandatory gate conditions from the Phase 8 specification."""

    @pytest.fixture
    def metrics(self):
        return _load_json(RESULTS_DIR / "metrics.json")

    def test_cold_two_tower_recall_lt_warm_two_tower_recall(self, metrics):
        warm_tt = metrics["slices"]["warm_users"]["two_tower"]["recall_10"]
        cold_tt = metrics["slices"]["cold_users"]["two_tower"]["recall_10"]
        assert cold_tt < warm_tt, (
            f"GATE FAIL: cold_two_tower recall ({cold_tt}) must be < "
            f"warm_two_tower recall ({warm_tt})"
        )

    def test_cold_hybrid_recall_gt_cold_two_tower_recall(self, metrics):
        cold_tt = metrics["slices"]["cold_users"]["two_tower"]["recall_10"]
        cold_hybrid = metrics["slices"]["cold_users"]["hybrid"]["recall_10"]
        assert cold_hybrid > cold_tt, (
            f"GATE FAIL: cold_hybrid recall ({cold_hybrid}) must be > "
            f"cold_two_tower recall ({cold_tt})"
        )


# ─── 5. Evaluation report schema ────────────────────────────────────────────

class TestEvaluationReport:
    """Validate evaluation_report.json structure and consistency with metrics.json."""

    @pytest.fixture
    def report(self):
        return _load_json(RESULTS_DIR / "evaluation_report.json")

    @pytest.fixture
    def metrics(self):
        return _load_json(RESULTS_DIR / "metrics.json")

    def test_report_has_threshold_n(self, report):
        assert "threshold_N" in report
        assert report["threshold_N"] == 10

    def test_report_has_slices(self, report):
        assert "slices" in report
        for s in ("warm_users", "cold_users"):
            assert s in report["slices"]

    def test_report_slices_match_metrics_json(self, report, metrics):
        """Verify recall/ndcg values in evaluation_report equal those in metrics.json."""
        for slice_name in ("warm_users", "cold_users"):
            for model in ("popularity", "content", "two_tower", "hybrid"):
                rep_val = report["slices"][slice_name][model]["recall_10"]
                met_val = metrics["slices"][slice_name][model]["recall_10"]
                assert abs(rep_val - met_val) < 1e-9, (
                    f"Mismatch: report/{slice_name}/{model}/recall_10={rep_val} "
                    f"!= metrics.json/{slice_name}/{model}/recall_10={met_val}"
                )


# ─── 6. Threshold sweep schema ──────────────────────────────────────────────

class TestThresholdSweep:
    """Validate threshold_sweep.json has required structure and multiple sweep values."""

    @pytest.fixture
    def sweep(self):
        return _load_json(RESULTS_DIR / "threshold_sweep.json")

    def test_sweep_has_thresholds_key(self, sweep):
        assert "thresholds" in sweep

    def test_sweep_has_at_least_two_values(self, sweep):
        assert len(sweep["thresholds"]) >= 2, (
            "threshold_sweep.json must contain at least 2 candidate threshold values"
        )

    def test_each_threshold_has_validation_metrics(self, sweep):
        for n_str, entry in sweep["thresholds"].items():
            assert "validation" in entry, f"Threshold {n_str} missing 'validation' key"
            assert "recall_10" in entry["validation"]
            assert "ndcg_10" in entry["validation"]

    def test_selected_threshold_is_in_sweep(self, sweep):
        selected = str(sweep["selected_threshold_N"])
        assert selected in sweep["thresholds"], (
            f"selected_threshold_N={selected} not present in thresholds dict"
        )
