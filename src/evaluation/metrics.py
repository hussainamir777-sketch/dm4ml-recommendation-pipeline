"""Ranking and rating metrics for top-K recommendation."""
from __future__ import annotations

import numpy as np


def precision_at_k(recs: list, relevant: set, k: int) -> float:
    """Share of the top-k recommendations that are relevant."""
    return len(set(recs[:k]) & relevant) / k


def recall_at_k(recs: list, relevant: set, k: int) -> float:
    """Share of the user's relevant items that appear in the top-k."""
    return len(set(recs[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(recs: list, relevant: set, k: int) -> float:
    """Normalized discounted cumulative gain with binary relevance."""
    dcg = sum(1.0 / np.log2(pos + 2) for pos, item in enumerate(recs[:k]) if item in relevant)
    ideal = sum(1.0 / np.log2(pos + 2) for pos in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def hit_rate_at_k(recs: list, relevant: set, k: int) -> float:
    """1 if at least one relevant item is in the top-k, else 0."""
    return float(bool(set(recs[:k]) & relevant))


def catalog_coverage(all_recs: dict, n_catalog_items: int) -> float:
    """Share of the item catalogue recommended to at least one user."""
    unique = {i for recs in all_recs.values() for i in recs}
    return len(unique) / n_catalog_items


def intra_list_diversity(recs: list, genre_vectors: dict) -> float:
    """Average pairwise (1 - cosine similarity) of item genre vectors within one list."""
    vecs = np.array([genre_vectors[i] for i in recs if i in genre_vectors], dtype=float)
    if len(vecs) < 2:
        return 0.0
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    unit = vecs / np.where(norms > 0, norms, 1)
    sim = unit @ unit.T
    n = len(vecs)
    mean_sim = (sim.sum() - np.trace(sim)) / (n * (n - 1))
    return float(1.0 - mean_sim)


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))
