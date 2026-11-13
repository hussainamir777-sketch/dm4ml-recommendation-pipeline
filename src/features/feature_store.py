"""Stage 5 - Feature management (a lightweight, file-based feature store).

Features are computed ONCE from the training period (as of the split cutoff),
written to data/features/ with a JSON registry, and documented in a feature
catalog. The `FeatureStore` class is the single read API used by BOTH model
training and prediction, so features are always computed the same way.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.utils import (GENRES, get_logger, get_path, load_config, load_table, read_json,
                       save_table, write_json)

# ---------------------------------------------------------------- catalog ---
FEATURE_CATALOG = [
    # name, entity, type, description, source
    ("user_rating_count", "user", "int", "Number of ratings the user gave in the training period", "train"),
    ("user_avg_rating", "user", "float", "Mean rating given by the user (global mean for new users)", "train"),
    ("user_rating_std", "user", "float", "Standard deviation of the user's ratings", "train"),
    ("user_positive_ratio", "user", "float", "Share of the user's ratings that are >= 4", "train"),
    ("user_active_days", "user", "int", "Days between the user's first and last training rating", "train"),
    ("user_days_since_last", "user", "int", "Days from the user's last rating to the split cutoff (-1 = never)", "train"),
    ("user_preferred_genre", "user", "category", "Genre the user rated most often (segment genre for new users)", "train + items"),
    ("user_genre_affinity_<genre>", "user", "float x19", "Share of the user's ratings per genre (new users: average of their gender/age segment)", "train + items + users"),
    ("is_cold_start_user", "user", "int", "1 if the user has no training interactions", "train"),
    ("age, gender, occupation, age_group", "user", "mixed", "Cleaned demographic attributes", "users"),
    ("item_rating_count", "item", "int", "Number of training ratings of the item", "train"),
    ("item_avg_rating", "item", "float", "Mean training rating of the item (global mean if unrated)", "train"),
    ("item_bayesian_avg", "item", "float", "Rating mean shrunk towards the global mean (damping = 10)", "train"),
    ("item_popularity_score", "item", "float", "log(1 + rating count) scaled to [0, 1]", "train"),
    ("item_popularity_rank", "item", "int", "Rank by rating count (1 = most popular)", "train"),
    ("item_days_since_last", "item", "int", "Days from the item's last rating to the cutoff (-1 = never)", "train"),
    ("item_release_year", "item", "int", "Release year (missing years imputed with the median)", "items"),
    ("item_age_years", "item", "int", "Cutoff year minus release year", "items"),
    ("<19 genre flags>, n_genres", "item", "int", "One-hot genre indicators and number of genres", "items"),
    ("user_item_genre_match", "interaction", "float", "Mean user genre affinity over the item's genres", "user + item features"),
    ("user_item_rating_deviation", "interaction", "float", "Rating minus the user's mean rating (training pairs)", "train"),
    ("item_avg_minus_user_avg", "interaction", "float", "Item mean rating minus user mean rating", "user + item features"),
]

MODEL_USAGE = {"user_genre_affinity_<genre>": "Hybrid", "item_popularity_score": "Hybrid, Popularity",
               "item_bayesian_avg": "Popularity", "user_item_genre_match": "Hybrid",
               "is_cold_start_user": "Hybrid, MF (fallback)", "<19 genre flags>, n_genres": "Hybrid, Evaluation (diversity)"}


def build_user_features(train: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame,
                        cutoff: pd.Timestamp) -> pd.DataFrame:
    """Compute per-user features from training interactions; new users get segment-based defaults."""
    mu = train["rating"].mean()
    g = train.groupby("user_id")
    stats = pd.DataFrame({
        "user_rating_count": g.size(),
        "user_avg_rating": g["rating"].mean(),
        "user_rating_std": g["rating"].std().fillna(0.0),
        "user_positive_ratio": g["rating"].apply(lambda r: (r >= 4).mean()),
        "user_active_days": (g["event_ts"].max() - g["event_ts"].min()).dt.days + 1,
        "user_days_since_last": (cutoff - g["event_ts"].max()).dt.days,
    })

    # Genre affinity: share of each user's ratings that fall in each genre
    t = train.merge(items[["item_id"] + GENRES], on="item_id")
    aff = t.groupby("user_id")[GENRES].sum()
    aff = aff.div(aff.sum(axis=1).replace(0, 1), axis=0)

    feats = users[["user_id", "age", "gender", "occupation", "age_group"]].merge(
        stats, left_on="user_id", right_index=True, how="left")
    feats["is_cold_start_user"] = feats["user_rating_count"].isna().astype("int8")
    feats = feats.fillna({"user_rating_count": 0, "user_avg_rating": mu, "user_rating_std": 0.0,
                          "user_positive_ratio": 0.0, "user_active_days": 0, "user_days_since_last": -1})
    feats = feats.astype({"user_rating_count": "int64", "user_active_days": "int64",
                          "user_days_since_last": "int64"})

    aff_cols = [f"user_genre_affinity_{g_}" for g_ in GENRES]
    aff_full = aff.reindex(feats["user_id"]).set_axis(aff_cols, axis=1).reset_index(drop=True)
    # Cold-start users: use the mean affinity of warm users in the same gender x age-group segment
    seg = feats[["gender", "age_group"]].reset_index(drop=True)
    seg_mean = pd.concat([seg, aff_full], axis=1).groupby(["gender", "age_group"])[aff_cols].transform("mean")
    aff_full = aff_full.fillna(seg_mean).fillna(aff_full.mean())
    feats = pd.concat([feats.reset_index(drop=True), aff_full], axis=1)
    feats["user_preferred_genre"] = aff_full.to_numpy().argmax(axis=1)
    feats["user_preferred_genre"] = feats["user_preferred_genre"].map(dict(enumerate(GENRES)))
    return feats


def build_item_features(train: pd.DataFrame, items: pd.DataFrame, cutoff: pd.Timestamp,
                        damping: float) -> pd.DataFrame:
    """Compute per-item popularity / quality features plus static item attributes."""
    mu = train["rating"].mean()
    g = train.groupby("item_id")
    stats = pd.DataFrame({
        "item_rating_count": g.size(),
        "item_avg_rating": g["rating"].mean(),
        "item_days_since_last": (cutoff - g["event_ts"].max()).dt.days,
    })
    feats = items[["item_id", "title", "release_year", "n_genres"] + GENRES].merge(
        stats, left_on="item_id", right_index=True, how="left")
    feats = feats.fillna({"item_rating_count": 0, "item_avg_rating": mu, "item_days_since_last": -1})
    feats = feats.astype({"item_rating_count": "int64", "item_days_since_last": "int64"})
    n = feats["item_rating_count"]
    feats["item_bayesian_avg"] = (n * feats["item_avg_rating"] + damping * mu) / (n + damping)
    feats["item_popularity_score"] = np.log1p(n) / np.log1p(n.max())
    feats["item_popularity_rank"] = n.rank(ascending=False, method="first").astype("int64")
    feats = feats.rename(columns={"release_year": "item_release_year"})
    feats["item_age_years"] = (cutoff.year - feats["item_release_year"]).clip(lower=0)
    return feats


class FeatureStore:
    """Read API over the saved features. Used identically at training and prediction time."""

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or load_config()
        folder = get_path(cfg, "features")
        self.user = load_table(folder, "user_features").set_index("user_id")
        self.item = load_table(folder, "item_features").set_index("item_id")
        self.registry = read_json(folder / "feature_registry.json")
        self.aff_cols = [f"user_genre_affinity_{g}" for g in GENRES]

    def get_user_features(self, user_ids) -> pd.DataFrame:
        """Feature rows for the given users (unknown ids -> NaN rows)."""
        return self.user.reindex(list(user_ids))

    def get_item_features(self, item_ids) -> pd.DataFrame:
        """Feature rows for the given items."""
        return self.item.reindex(list(item_ids))

    def user_genre_affinity(self, user_ids) -> np.ndarray:
        """(n_users x 19) genre-affinity matrix."""
        return self.get_user_features(user_ids)[self.aff_cols].fillna(0.0).to_numpy()

    def item_genre_matrix(self, item_ids) -> np.ndarray:
        """(n_items x 19) one-hot genre matrix."""
        return self.get_item_features(item_ids)[GENRES].fillna(0).to_numpy(dtype=float)

    def genre_match(self, user_ids, item_ids) -> np.ndarray:
        """(n_users x n_items) user_item_genre_match: mean affinity over each item's genres."""
        item_g = self.item_genre_matrix(item_ids)
        item_g = item_g / np.maximum(item_g.sum(axis=1, keepdims=True), 1)
        return self.user_genre_affinity(user_ids) @ item_g.T

    def interaction_features(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """Interaction features for (user_id, item_id[, rating]) pairs."""
        u = self.get_user_features(pairs["user_id"]).reset_index(drop=True)
        i = self.get_item_features(pairs["item_id"]).reset_index(drop=True)
        out = pairs[["user_id", "item_id"]].reset_index(drop=True).copy()
        ig = i[GENRES].to_numpy(dtype=float)
        ig = ig / np.maximum(ig.sum(axis=1, keepdims=True), 1)
        out["user_item_genre_match"] = (u[self.aff_cols].to_numpy() * ig).sum(axis=1)
        out["item_avg_minus_user_avg"] = i["item_avg_rating"].to_numpy() - u["user_avg_rating"].to_numpy()
        if "rating" in pairs:
            out["user_item_rating_deviation"] = pairs["rating"].to_numpy() - u["user_avg_rating"].to_numpy()
        return out


def write_catalog(path, user_feats: pd.DataFrame, item_feats: pd.DataFrame) -> None:
    """Write the feature catalog as a Markdown table."""
    lines = ["# Feature Catalog", "",
             f"Features are computed from training data only, as of the train/test cutoff. "
             f"User table: {len(user_feats)} rows x {user_feats.shape[1]} columns; "
             f"item table: {len(item_feats)} rows x {item_feats.shape[1]} columns.", "",
             "| Feature | Entity | Type | Description | Source | Used by |", "|---|---|---|---|---|---|"]
    for name, entity, typ, desc, src in FEATURE_CATALOG:
        lines.append(f"| `{name}` | {entity} | {typ} | {desc} | {src} | {MODEL_USAGE.get(name, 'Analysis / reports')} |")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_feature_engineering(cfg: dict | None = None) -> dict:
    """Build, save and register all features; write the feature catalog."""
    cfg = cfg or load_config()
    logger = get_logger("features", cfg)
    processed, features_dir, reports = (get_path(cfg, k) for k in ["processed", "features", "reports"])

    train = load_table(processed, "train")
    users, items = load_table(processed, "users_clean"), load_table(processed, "items_clean")
    split = read_json(processed / "split_info.json")
    cutoff = pd.to_datetime(split["cutoff_timestamp"], unit="s", utc=True)

    user_feats = build_user_features(train, users, items, cutoff)
    item_feats = build_item_features(train, items, cutoff, cfg["models"]["popularity"]["damping"])
    save_table(user_feats, features_dir, "user_features")
    save_table(item_feats, features_dir, "item_features")

    registry = {
        "feature_set_version": "v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of_cutoff": split["cutoff_datetime_utc"],
        "user_features": {"rows": len(user_feats), "columns": list(user_feats.columns)},
        "item_features": {"rows": len(item_feats), "columns": list(item_feats.columns)},
        "interaction_features": ["user_item_genre_match", "item_avg_minus_user_avg",
                                 "user_item_rating_deviation"],
    }
    write_json(registry, features_dir / "feature_registry.json")

    # Interaction features for the training pairs (computed through the same store API)
    store = FeatureStore(cfg)
    inter_feats = store.interaction_features(train[["user_id", "item_id", "rating"]])
    save_table(inter_feats, features_dir, "interaction_features_train")
    write_catalog(reports / "feature_catalog.md", user_feats, item_feats)

    logger.info("User features: %s | item features: %s | interaction features: %s",
                user_feats.shape, item_feats.shape, inter_feats.shape)
    logger.info("Cold-start users given segment-based genre affinity: %d",
                int(user_feats["is_cold_start_user"].sum()))
    logger.info("Example user features:\n%s", user_feats[["user_id", "user_rating_count", "user_avg_rating",
                "user_preferred_genre", "user_days_since_last", "is_cold_start_user"]].head(5).to_string(index=False))
    logger.info("Feature catalog written to reports/feature_catalog.md")
    return {"stage": "features", "user_features": list(user_feats.shape),
            "item_features": list(item_feats.shape), "interaction_features": list(inter_feats.shape)}


if __name__ == "__main__":
    run_feature_engineering()
