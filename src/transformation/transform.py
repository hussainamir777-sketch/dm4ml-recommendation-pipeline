"""Stage 4 - Data transformation.

Builds derived datasets from the TRAINING data only (so nothing from the test
period leaks into models):
  * user_activity_summary  - per-user counts, rating statistics and activity span
  * item_popularity        - per-item counts, mean / Bayesian-average rating, rank
  * user-item interaction matrix (sparse, users x items) + index mappings
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from src.utils import get_logger, get_path, load_config, load_table, save_table, write_json


def build_user_activity(train: pd.DataFrame) -> pd.DataFrame:
    """Aggregate each user's training interactions."""
    g = train.groupby("user_id")
    out = pd.DataFrame({
        "n_ratings": g.size(),
        "avg_rating": g["rating"].mean(),
        "std_rating": g["rating"].std().fillna(0.0),
        "first_ts": g["event_ts"].min(),
        "last_ts": g["event_ts"].max(),
    }).reset_index()
    out["active_days"] = (out["last_ts"] - out["first_ts"]).dt.days + 1
    return out


def build_item_popularity(train: pd.DataFrame, damping: float) -> pd.DataFrame:
    """Per-item popularity with a Bayesian average that shrinks rarely rated items to the global mean."""
    mu = train["rating"].mean()
    g = train.groupby("item_id")
    out = pd.DataFrame({
        "n_ratings": g.size(),
        "avg_rating": g["rating"].mean(),
        "n_positive": g["rating"].apply(lambda r: int((r >= 4).sum())),
        "last_ts": g["event_ts"].max(),
    }).reset_index()
    out["bayesian_avg"] = (out["n_ratings"] * out["avg_rating"] + damping * mu) / (out["n_ratings"] + damping)
    out["popularity_rank"] = out["n_ratings"].rank(ascending=False, method="first").astype("int64")
    return out.sort_values("popularity_rank").reset_index(drop=True)


def build_interaction_matrix(train: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame):
    """Sparse users x items rating matrix over the full user/item catalogue."""
    user_index = pd.DataFrame({"user_id": users["user_id"].sort_values().to_numpy()})
    user_index["row"] = np.arange(len(user_index))
    item_index = pd.DataFrame({"item_id": items["item_id"].sort_values().to_numpy()})
    item_index["col"] = np.arange(len(item_index))
    t = train.merge(user_index, on="user_id").merge(item_index, on="item_id")
    matrix = sparse.csr_matrix((t["rating"].astype(float), (t["row"], t["col"])),
                               shape=(len(user_index), len(item_index)))
    return matrix, user_index, item_index


def run_transformation(cfg: dict | None = None) -> dict:
    """Create and save all derived datasets."""
    cfg = cfg or load_config()
    logger = get_logger("transformation", cfg)
    processed = get_path(cfg, "processed")

    train = load_table(processed, "train")
    users, items = load_table(processed, "users_clean"), load_table(processed, "items_clean")

    activity = build_user_activity(train)
    popularity = build_item_popularity(train, cfg["models"]["popularity"]["damping"])
    matrix, user_index, item_index = build_interaction_matrix(train, users, items)

    save_table(activity, processed, "user_activity_summary")
    save_table(popularity, processed, "item_popularity")
    save_table(user_index, processed, "user_index")
    save_table(item_index, processed, "item_index")
    sparse.save_npz(processed / "user_item_matrix.npz", matrix)

    density = matrix.nnz / (matrix.shape[0] * matrix.shape[1])
    summary = {"stage": "transformation", "matrix_shape": list(matrix.shape),
               "matrix_nonzeros": int(matrix.nnz), "matrix_density_pct": round(100 * density, 3),
               "users_with_activity": len(activity), "items_with_ratings": len(popularity)}
    write_json(summary, processed / "transformation_summary.json")
    logger.info("User activity summary: %d users | item popularity: %d items", len(activity), len(popularity))
    logger.info("User-item matrix %s with %d ratings (density %.2f%%)",
                tuple(matrix.shape), matrix.nnz, 100 * density)
    logger.info("Top-5 most popular training items: %s",
                popularity.head(5)[["item_id", "n_ratings", "avg_rating"]].round(2).to_dict("records"))
    return summary


if __name__ == "__main__":
    run_transformation()
