"""Stage 3 - Data preparation.

Cleans the validated tables, standardizes formats, handles missing values and
duplicate user-item pairs, then creates a time-based train/test split:
every interaction before the cutoff timestamp is training data, everything
after it is test data (no future information leaks into training).
"""
from __future__ import annotations

import pandas as pd

from src.utils import GENRES, get_logger, get_path, load_config, load_table, save_table, write_json


def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    """Standardize user fields and add an age group."""
    users = users.copy()
    users["gender"] = users["gender"].str.upper().str.strip()
    users["occupation"] = users["occupation"].str.lower().str.strip()
    users["zip_code"] = users["zip_code"].astype(str).str.strip().replace("", "unknown")
    users["age_group"] = pd.cut(users["age"], bins=[0, 17, 24, 34, 44, 54, 120],
                                labels=["<18", "18-24", "25-34", "35-44", "45-54", "55+"]).astype(str)
    return users.sort_values("user_id").reset_index(drop=True)


def clean_items(items: pd.DataFrame, logger) -> pd.DataFrame:
    """Parse release dates, impute missing release years, build a genre string."""
    items = items.copy()
    items["title"] = items["title"].str.strip()
    items["release_date"] = pd.to_datetime(items["release_date"], format="%d-%b-%Y", errors="coerce")
    items["release_date_missing"] = items["release_date"].isna().astype("int8")
    year = items["release_date"].dt.year
    median_year = int(year.median())
    items["release_year"] = year.fillna(median_year).astype("int64")
    logger.info("Items with missing release date: %d (release_year imputed with median %d)",
                int(items["release_date_missing"].sum()), median_year)
    # video_release_date is empty for every MovieLens item -> dropped as an uninformative column
    items = items.drop(columns=["video_release_date"])
    items["genres"] = items[GENRES].apply(lambda r: "|".join(g for g in GENRES if r[g] == 1), axis=1)
    items["n_genres"] = items[GENRES].sum(axis=1).astype("int64")
    return items.sort_values("item_id").reset_index(drop=True)


def clean_interactions(inter: pd.DataFrame, errors_dir, logger) -> pd.DataFrame:
    """Convert Unix timestamps to UTC datetimes and keep only the latest rating per user-item pair."""
    inter = inter.copy()
    inter["event_ts"] = pd.to_datetime(inter["timestamp"], unit="s", utc=True)
    inter = inter.sort_values(["user_id", "item_id", "timestamp"])
    dup = inter.duplicated(["user_id", "item_id"], keep="last")
    if dup.any():
        inter[dup].assign(error_reason="older_duplicate_user_item_pair").to_csv(
            errors_dir / "interactions_pair_duplicates.csv", index=False)
    logger.info("Duplicate user-item pairs removed (older ratings): %d", int(dup.sum()))
    return inter[~dup].sort_values("timestamp").reset_index(drop=True)


def time_based_split(inter: pd.DataFrame, quantile: float) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Global temporal split: ratings up to the cutoff -> train, later ratings -> test."""
    cutoff = int(inter["timestamp"].quantile(quantile))
    train = inter[inter["timestamp"] <= cutoff].reset_index(drop=True)
    test = inter[inter["timestamp"] > cutoff].reset_index(drop=True)
    return train, test, cutoff


def run_preparation(cfg: dict | None = None) -> dict:
    """Run cleaning and the time-based split; save clean tables and split info."""
    cfg = cfg or load_config()
    logger = get_logger("preparation", cfg)
    processed, errors_dir = get_path(cfg, "processed"), get_path(cfg, "errors")

    users = clean_users(load_table(processed, "validated_users"))
    items = clean_items(load_table(processed, "validated_items"), logger)
    inter = clean_interactions(load_table(processed, "validated_interactions"), errors_dir, logger)

    train, test, cutoff = time_based_split(inter, cfg["preparation"]["split_quantile"])
    cold_users = sorted(set(test["user_id"]) - set(train["user_id"]))
    cold_items = sorted(set(test["item_id"]) - set(train["item_id"]))

    for name, df in [("users_clean", users), ("items_clean", items), ("interactions_clean", inter),
                     ("train", train), ("test", test)]:
        save_table(df, processed, name)

    info = {
        "stage": "preparation",
        "cutoff_timestamp": cutoff,
        "cutoff_datetime_utc": str(pd.to_datetime(cutoff, unit="s", utc=True)),
        "train_rows": len(train), "test_rows": len(test),
        "train_period": [str(train["event_ts"].min()), str(train["event_ts"].max())],
        "test_period": [str(test["event_ts"].min()), str(test["event_ts"].max())],
        "train_users": int(train["user_id"].nunique()), "test_users": int(test["user_id"].nunique()),
        "cold_start_users_in_test": len(cold_users), "cold_start_items_in_test": len(cold_items),
    }
    write_json(info, processed / "split_info.json")
    logger.info("Time-based split at %s -> train=%d rows, test=%d rows",
                info["cutoff_datetime_utc"], len(train), len(test))
    logger.info("Test users=%d (cold-start users not seen in train: %d), cold-start items=%d",
                info["test_users"], len(cold_users), len(cold_items))
    return info


if __name__ == "__main__":
    run_preparation()
