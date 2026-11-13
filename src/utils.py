"""Shared helpers: configuration, logging, paths and table I/O."""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load the YAML configuration used by every pipeline stage."""
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_path(cfg: dict, key: str) -> Path:
    """Return an absolute project path from the `paths` section and create it."""
    p = PROJECT_ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    """Seed Python and NumPy so every run of the pipeline is reproducible."""
    random.seed(seed)
    np.random.seed(seed)


def get_logger(name: str, cfg: dict | None = None) -> logging.Logger:
    """Logger that writes to the console and to logs/pipeline.log."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                            "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    log_dir = PROJECT_ROOT / (cfg["paths"]["logs"] if cfg else "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def _parquet_available() -> bool:
    """Parquet needs pyarrow (listed in requirements.txt); fall back to CSV otherwise."""
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def save_table(df: pd.DataFrame, folder: Path, name: str) -> Path:
    """Save a DataFrame as Parquet (or CSV if pyarrow is missing) and return the path."""
    folder.mkdir(parents=True, exist_ok=True)
    if _parquet_available():
        path = folder / f"{name}.parquet"
        df.to_parquet(path, index=False)
    else:
        path = folder / f"{name}.csv"
        df.to_csv(path, index=False)
    return path


def load_table(folder: Path, name: str) -> pd.DataFrame:
    """Load a table written by `save_table`, restoring datetime columns for CSV files."""
    pq, csv = folder / f"{name}.parquet", folder / f"{name}.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    df = pd.read_csv(csv)
    for col in df.columns:
        is_text = df[col].dtype == object or pd.api.types.is_string_dtype(df[col])
        if col.endswith(("_ts", "_date")) and is_text:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=col.endswith("_ts"))
    return df


def write_json(obj: dict, path: Path) -> None:
    """Write a dictionary as pretty JSON (numpy types converted)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def read_json(path: Path) -> dict:
    """Read a JSON file into a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


GENRES = ["unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
          "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
          "Romance", "Sci-Fi", "Thriller", "War", "Western"]
