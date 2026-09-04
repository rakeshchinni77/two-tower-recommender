"""Unit test suite for Phase 8 offline slice-wise evaluation pipeline."""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import PROCESSED_DATA_DIR, RESULTS_DIR, ARTIFACTS_DIR
from src.evaluate import (
    run_offline_evaluation,
    validate_evaluation_data,
    safe_min_max_normalize,
    get_deterministic_test_targets
)


def test_deterministic_target_sampling_cap():
    test_items = list(range(1, 50))  # 49 items
    user_id = 101

    targets1 = get_deterministic_test_targets(test_items, user_id=user_id, eval_targets=10, seed=42)
    targets2 = get_deterministic_test_targets(test_items, user_id=user_id, eval_targets=10, seed=42)

    assert len(targets1) == 10
    assert targets1 == targets2, "Target sampling is not deterministic"
    assert len(set(targets1)) == 10, "Target items contain duplicates"

    # Fewer than 10 items case
    small_items = [1, 2, 3]
    small_targets = get_deterministic_test_targets(small_items, user_id=user_id, eval_targets=10, seed=42)
    assert small_targets == [1, 2, 3]


def test_metrics_json_file_created_and_valid_schema():
    metrics_path = RESULTS_DIR / "metrics.json"
    assert metrics_path.exists(), f"results/metrics.json missing at {metrics_path}"

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Enforce EXACT schema required by task
    assert "threshold_N" in metrics
    assert "slices" in metrics
    assert set(metrics.keys()) == {"threshold_N", "slices"}, f"Unexpected top-level keys in metrics.json: {set(metrics.keys())}"

    assert "warm_users" in metrics["slices"]
    assert "cold_users" in metrics["slices"]

    required_models = ["popularity", "content", "two_tower", "hybrid"]
    required_metrics = ["recall_10", "ndcg_10"]

    for slice_name in ["warm_users", "cold_users"]:
        for model_name in required_models:
            assert model_name in metrics["slices"][slice_name], f"Missing {model_name} in {slice_name} slice"
            for m_name in required_metrics:
                assert m_name in metrics["slices"][slice_name][model_name], f"Missing {m_name} in {slice_name}/{model_name}"
                val = metrics["slices"][slice_name][model_name][m_name]
                assert isinstance(val, (float, int))
                assert 0.0 <= val <= 1.0, f"Invalid metric value {val} for {slice_name}/{model_name}/{m_name}"
                assert not np.isnan(val), f"NaN metric for {slice_name}/{model_name}/{m_name}"
                assert not np.isinf(val), f"Inf metric for {slice_name}/{model_name}/{m_name}"


def test_leakage_prevention_validation():
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    test_warm_df = pd.read_csv(PROCESSED_DATA_DIR / "test_warm.csv")
    test_cold_df = pd.read_csv(PROCESSED_DATA_DIR / "test_cold.csv")

    no_cold_leak, no_test_leak = validate_evaluation_data(train_df, test_warm_df, test_cold_df)
    assert no_cold_leak is True
    assert no_test_leak is True

    # Corrupt cold dataset by injecting a train user to test failure raising
    invalid_cold_df = pd.concat([test_cold_df, train_df.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="DATA LEAKAGE VIOLATION"):
        validate_evaluation_data(train_df, test_warm_df, invalid_cold_df)


def test_safe_min_max_normalize():
    const_arr = np.array([5.0, 5.0, 5.0])
    norm_const = safe_min_max_normalize(const_arr)
    assert (norm_const == 0.0).all()

    arr = np.array([1.0, 3.0, 5.0])
    norm_arr = safe_min_max_normalize(arr)
    assert norm_arr[0] == 0.0
    assert norm_arr[-1] == 1.0

    nan_arr = np.array([1.0, np.nan, np.inf])
    norm_nan = safe_min_max_normalize(nan_arr)
    assert not np.isnan(norm_nan).any()
    assert not np.isinf(norm_nan).any()


def test_offline_evaluation_determinism():
    res1 = run_offline_evaluation()
    res2 = run_offline_evaluation()

    assert res1 == res2, "Offline evaluation is not deterministic across repeated runs"


def test_threshold_sweep_file_and_schema():
    sweep_path = RESULTS_DIR / "threshold_sweep.json"
    sub_path = Path(__file__).resolve().parent.parent / "submission.json"
    metrics_path = RESULTS_DIR / "metrics.json"

    assert sweep_path.exists(), f"results/threshold_sweep.json missing at {sweep_path}"
    assert sub_path.exists(), f"submission.json missing at {sub_path}"
    assert metrics_path.exists(), f"results/metrics.json missing at {metrics_path}"

    with open(sweep_path, "r", encoding="utf-8") as f:
        sweep_data = json.load(f)

    with open(sub_path, "r", encoding="utf-8") as f:
        sub_data = json.load(f)

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    # 1. Verify sweep schema and TRAIN-ONLY validation presence
    assert "random_seed" in sweep_data
    assert sweep_data["random_seed"] == 42
    assert "eval_targets" in sweep_data
    assert sweep_data["eval_targets"] == 10
    assert "thresholds" in sweep_data
    assert "selected_threshold_N" in sweep_data
    assert "selection_rationale" in sweep_data

    # Verify candidate thresholds contain TRAIN-ONLY validation metrics
    for cand_str in ["5", "10", "20", "30"]:
        assert cand_str in sweep_data["thresholds"], f"Candidate N={cand_str} missing in threshold_sweep.json"
        cand_dict = sweep_data["thresholds"][cand_str]
        assert "validation" in cand_dict, f"Validation metrics missing for N={cand_str}"
        assert "recall_10" in cand_dict["validation"]
        assert "ndcg_10" in cand_dict["validation"]

    selected_n = sweep_data["selected_threshold_N"]
    assert str(selected_n) in sweep_data["thresholds"], "Selected N not found in evaluated thresholds"

    # 2. Verify submission.json matching
    assert sub_data["threshold_N"] == selected_n, "submission.json threshold_N does not match sweep"

    # 3. Verify metrics.json matching and exact schema
    assert metrics_data["threshold_N"] == selected_n, "metrics.json threshold_N does not match sweep"
    assert "slices" in metrics_data, "metrics.json missing 'slices' key"
    assert set(metrics_data.keys()) == {"threshold_N", "slices"}


