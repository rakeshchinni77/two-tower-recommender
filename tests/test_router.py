"""Tests for hybrid router dynamic route selection and online routing gate."""

import pytest
from src.router import HybridRouter, route


def test_router_placeholder():
    """Placeholder test for hybrid router backward compatibility."""
    router = HybridRouter()
    res = router.recommend(user_id=1, history=[])
    assert res["user_id"] == 1


def test_route_function_exact_contract():
    """Test top-level route function against mandatory task assertions for N=10."""
    assert route(50, 10) == "two_tower"
    assert route(11, 10) == "two_tower"
    assert route(10, 10) == "content_fallback"
    assert route(1, 10) == "content_fallback"
    assert route(0, 10) == "popularity_fallback"


def test_route_selection_boundaries():
    """Test dynamic routing strategy assignment around threshold N."""
    threshold = 10
    router = HybridRouter(threshold_n=threshold)

    # 1. history length > N -> two_tower
    assert router.route_user(history_count=50) == "two_tower"
    assert router.route_user(history_count=15) == "two_tower"
    assert router.route_user(history_count=11) == "two_tower"

    # 2. history length == N -> content_fallback (strictly > N, not >= N)
    assert router.route_user(history_count=10) == "content_fallback"

    # 3. history length < N and > 0 -> content_fallback
    assert router.route_user(history_count=5) == "content_fallback"
    assert router.route_user(history_count=1) == "content_fallback"

    # 4. history length == 0 -> popularity_fallback
    assert router.route_user(history_count=0) == "popularity_fallback"


def test_configurable_threshold():
    """Test that setting custom threshold_n works as expected."""
    router_5 = HybridRouter(threshold_n=5)
    assert router_5.route_user(history_count=6) == "two_tower"
    assert router_5.route_user(history_count=5) == "content_fallback"
    assert router_5.route_user(history_count=4) == "content_fallback"

    router_20 = HybridRouter(threshold_n=20)
    assert router_20.route_user(history_count=21) == "two_tower"
    assert router_20.route_user(history_count=20) == "content_fallback"
    assert router_20.route_user(history_count=15) == "content_fallback"


def test_cold_user_never_routed_to_two_tower():
    """Test that zero-history users are NEVER routed to two_tower."""
    for threshold in [0, 5, 10, 20, 50]:
        router = HybridRouter(threshold_n=threshold)
        strategy = router.route_user(history_count=0)
        assert strategy != "two_tower"
        assert strategy == "popularity_fallback"


def test_unknown_user_recommendation_and_structure():
    """Test unknown user recommendations generate valid top-K unique integer IDs without crashing."""
    router = HybridRouter(threshold_n=10)
    res = router.recommend(user_id=999999, history=[], top_k=10)

    assert res["user_id"] == 999999
    assert res["strategy_used"] == "popularity_fallback"
    assert len(res["recommendations"]) == 10
    assert len(set(res["recommendations"])) == 10
    assert all(isinstance(x, int) for x in res["recommendations"])


def test_recommendation_unique_ids_with_history():
    """Test recommendation generation with arbitrary history returns top_k unique IDs."""
    router = HybridRouter(threshold_n=5)

    # history > N (6 items)
    res_tt = router.recommend(user_id=1, history=[101, 102, 103, 104, 105, 106], top_k=10)
    assert res_tt["strategy_used"] == "two_tower"
    assert len(res_tt["recommendations"]) == 10
    assert len(set(res_tt["recommendations"])) == 10
    assert all(isinstance(x, int) for x in res_tt["recommendations"])

    # history <= N (3 items)
    res_cont = router.recommend(user_id=1, history=[101, 102, 103], top_k=10)
    assert res_cont["strategy_used"] == "content_fallback"
    assert len(res_cont["recommendations"]) == 10
    assert len(set(res_cont["recommendations"])) == 10
    assert all(isinstance(x, int) for x in res_cont["recommendations"])


def test_negative_history_and_threshold_validation():
    """Test that negative history_count or threshold_n raises ValueError safely."""
    with pytest.raises(ValueError, match="history_count must not be negative"):
        route(-1, 10)

    with pytest.raises(ValueError, match="threshold_n must be non-negative"):
        route(5, -1)

    with pytest.raises(ValueError, match="threshold_n must be non-negative"):
        HybridRouter(threshold_n=-5)


