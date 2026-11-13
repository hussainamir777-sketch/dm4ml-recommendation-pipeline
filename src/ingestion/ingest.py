"""Stage 1 - Data ingestion.

Downloads MovieLens 100K (or generates same-format synthetic data), parses the
three source files into raw user / item / interaction tables, optionally injects
a documented set of corrupted records, and writes everything to data/raw/.
Raw tables are stored as text (strings) on purpose, so the validation stage can
check data types itself.
"""
from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import GENRES, get_logger, get_path, load_config, set_seed, write_json

USER_COLS = ["user_id", "age", "gender", "occupation", "zip_code"]
ITEM_COLS = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRES
INTERACTION_COLS = ["user_id", "item_id", "rating", "timestamp"]


def download_movielens(url: str, raw_dir: Path, logger) -> Path:
    """Download and extract MovieLens 100K once; reuse the local copy afterwards."""
    extract_dir = raw_dir / "ml-100k"
    if (extract_dir / "u.data").exists():
        logger.info("MovieLens already present at %s - skipping download", extract_dir)
        return extract_dir
    import requests  # installed with mlflow; uses certifi, so HTTPS works on macOS
    logger.info("Downloading MovieLens 100K from %s", url)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(raw_dir)
    logger.info("Extracted %d bytes to %s", len(resp.content), extract_dir)
    return extract_dir


def read_source_files(src_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse u.user, u.item and u.data into DataFrames of strings (no type casting)."""
    users = pd.read_csv(src_dir / "u.user", sep="|", names=USER_COLS, dtype=str,
                        keep_default_na=False)
    items = pd.read_csv(src_dir / "u.item", sep="|", names=ITEM_COLS, dtype=str,
                        encoding="latin-1", keep_default_na=False)
    interactions = pd.read_csv(src_dir / "u.data", sep="\t", names=INTERACTION_COLS,
                               dtype=str, keep_default_na=False)
    return users, items, interactions


def inject_faults(users: pd.DataFrame, items: pd.DataFrame, interactions: pd.DataFrame,
                  n: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Append a documented set of corrupted records so validation has real errors to catch.

    Returns the three tables plus a manifest describing every injected fault.
    """
    rng = np.random.default_rng(seed)
    sample = interactions.sample(n, random_state=seed).reset_index(drop=True)
    faults, rows = [], []
    kinds = ["rating_out_of_range", "rating_not_numeric", "missing_user_id", "unknown_user_id",
             "unknown_item_id", "timestamp_in_future", "timestamp_too_old", "timestamp_not_numeric",
             "exact_duplicate", "missing_rating"]
    for idx, row in sample.iterrows():
        kind = kinds[idx % len(kinds)]
        r = row.copy()
        if kind == "rating_out_of_range":
            r["rating"] = str(rng.choice([0, 6, 7, 10]))
        elif kind == "rating_not_numeric":
            r["rating"] = str(rng.choice(["five", "good", "4.5x"]))
        elif kind == "missing_user_id":
            r["user_id"] = ""
        elif kind == "unknown_user_id":
            r["user_id"] = str(9000 + idx)
        elif kind == "unknown_item_id":
            r["item_id"] = str(90000 + idx)
        elif kind == "timestamp_in_future":
            r["timestamp"] = str(int(datetime(2035, 1, 1, tzinfo=timezone.utc).timestamp()))
        elif kind == "timestamp_too_old":
            r["timestamp"] = str(int(datetime(1970, 1, 2, tzinfo=timezone.utc).timestamp()))
        elif kind == "timestamp_not_numeric":
            r["timestamp"] = "yesterday"
        elif kind == "missing_rating":
            r["rating"] = ""
        # exact_duplicate keeps the row unchanged -> identical copy of an existing record
        rows.append(r)
        faults.append({"table": "interactions", "fault": kind})
    interactions = pd.concat([interactions, pd.DataFrame(rows)], ignore_index=True)

    bad_users = pd.DataFrame([
        {"user_id": "2001", "age": "-5", "gender": "M", "occupation": "student", "zip_code": "12345"},
        {"user_id": "2002", "age": "150", "gender": "F", "occupation": "writer", "zip_code": "54321"},
        {"user_id": "2003", "age": "30", "gender": "X", "occupation": "engineer", "zip_code": "11111"},
        {"user_id": "2004", "age": "abc", "gender": "F", "occupation": "", "zip_code": "22222"},
        users.iloc[0].to_dict(),                                   # duplicate user_id
    ])
    users = pd.concat([users, bad_users], ignore_index=True)
    faults += [{"table": "users", "fault": f} for f in
               ["age_negative", "age_too_high", "invalid_gender", "age_not_numeric", "duplicate_user_id"]]

    bad_item = items.iloc[0].to_dict()
    missing_title = dict(bad_item, item_id="5001", title="")
    items = pd.concat([items, pd.DataFrame([missing_title, bad_item])], ignore_index=True)
    faults += [{"table": "items", "fault": f} for f in ["missing_title", "duplicate_item_id"]]

    return users, items, interactions, pd.DataFrame(faults)


def file_md5(path: Path) -> str:
    """MD5 checksum, recorded in the manifest so later runs can detect changed inputs."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def run_ingestion(cfg: dict | None = None) -> dict:
    """Run the ingestion stage and return a summary dictionary."""
    cfg = cfg or load_config()
    logger = get_logger("ingestion", cfg)
    set_seed(cfg["seed"])
    raw_dir = get_path(cfg, "raw")
    icfg = cfg["ingestion"]

    if icfg["source"] == "movielens":
        src_dir = download_movielens(icfg["movielens_url"], raw_dir, logger)
    else:
        from src.ingestion.synthetic import generate_movielens_like
        src_dir = raw_dir / "synthetic"
        logger.info("Generating synthetic MovieLens-format data in %s", src_dir)
        generate_movielens_like(src_dir, seed=cfg["seed"])

    users, items, interactions = read_source_files(src_dir)
    logger.info("Read %d users, %d items, %d interactions", len(users), len(items), len(interactions))

    if icfg.get("inject_faults", False):
        users, items, interactions, faults = inject_faults(
            users, items, interactions, icfg["n_fault_records"], cfg["seed"])
        faults.to_csv(raw_dir / "injected_faults_manifest.csv", index=False)
        logger.info("Injected %d documented faulty records (see injected_faults_manifest.csv)",
                    len(faults))

    paths = {}
    for name, df in [("users", users), ("items", items), ("interactions", interactions)]:
        p = raw_dir / f"{name}.csv"
        df.to_csv(p, index=False)
        paths[name] = p
        logger.info("Saved raw %-12s -> %s (%d rows)", name, p.relative_to(raw_dir.parent.parent), len(df))

    summary = {
        "stage": "ingestion",
        "source": icfg["source"],
        "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": {"users": len(users), "items": len(items), "interactions": len(interactions)},
        "faults_injected": bool(icfg.get("inject_faults", False)),
        "checksums_md5": {k: file_md5(v) for k, v in paths.items()},
    }
    write_json(summary, raw_dir / "ingestion_manifest.json")
    logger.info("Ingestion complete: %s", summary["rows"])
    return summary


if __name__ == "__main__":
    run_ingestion()
