"""Sanity tests for the evaluation metrics.  Run:  python -m tests.test_metrics"""
from src.evaluation import metrics as M


def test_ranking_metrics():
    """Hand-checked values for a 5-item list with 2 relevant hits at ranks 1 and 3."""
    recs, relevant = [10, 20, 30, 40, 50], {10, 30, 99}
    assert M.precision_at_k(recs, relevant, 5) == 2 / 5
    assert abs(M.recall_at_k(recs, relevant, 5) - 2 / 3) < 1e-12
    assert M.hit_rate_at_k(recs, relevant, 5) == 1.0
    ndcg = M.ndcg_at_k(recs, relevant, 5)
    assert 0 < ndcg < 1
    assert M.ndcg_at_k([10, 30, 99], relevant, 3) == 1.0          # perfect ranking


def test_coverage_and_diversity():
    """Coverage counts unique items; identical genre vectors give zero diversity."""
    assert M.catalog_coverage({1: [1, 2], 2: [2, 3]}, 10) == 0.3
    same = {1: [1, 0], 2: [1, 0]}
    different = {1: [1, 0], 2: [0, 1]}
    assert M.intra_list_diversity([1, 2], same) == 0.0
    assert M.intra_list_diversity([1, 2], different) == 1.0


def test_rating_errors():
    """RMSE and MAE on a tiny example."""
    assert M.rmse([1, 2, 3], [1, 2, 5]) == (4 / 3) ** 0.5
    assert M.mae([1, 2, 3], [1, 2, 5]) == 2 / 3


if __name__ == "__main__":
    test_ranking_metrics()
    test_coverage_and_diversity()
    test_rating_errors()
    print("All metric tests passed.")
