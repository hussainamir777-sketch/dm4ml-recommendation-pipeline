"""Offline fallback: generate data in exactly the MovieLens 100K raw format.

Ratings come from a latent-factor model with genre structure and power-law item
popularity, so collaborative filtering has real signal to learn.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import GENRES

OCCUPATIONS = ["administrator", "artist", "doctor", "educator", "engineer", "entertainment",
               "executive", "healthcare", "homemaker", "lawyer", "librarian", "marketing",
               "none", "other", "programmer", "retired", "salesman", "scientist", "student",
               "technician", "writer"]


def generate_movielens_like(out_dir: Path, seed: int = 42, n_users: int = 943,
                            n_items: int = 1682, n_ratings: int = 100_000) -> None:
    """Write u.user, u.item and u.data files in the MovieLens 100K layout."""
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Users: id | age | gender | occupation | zip
    users = pd.DataFrame({
        "user_id": np.arange(1, n_users + 1),
        "age": rng.integers(15, 70, n_users),
        "gender": rng.choice(["M", "F"], n_users, p=[0.7, 0.3]),
        "occupation": rng.choice(OCCUPATIONS, n_users),
        "zip_code": [f"{z:05d}" for z in rng.integers(1000, 99999, n_users)],
    })
    users.to_csv(out_dir / "u.user", sep="|", header=False, index=False)

    # Items: id | title | release date | video release | url | 19 genre flags
    genre_mat = np.zeros((n_items, len(GENRES)), dtype=int)
    for i in range(n_items):
        k = rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
        genre_mat[i, rng.choice(np.arange(1, len(GENRES)), k, replace=False)] = 1
    years = rng.integers(1930, 1998, n_items)
    items = pd.DataFrame({
        "item_id": np.arange(1, n_items + 1),
        "title": [f"Synthetic Movie {i} ({y})" for i, y in zip(range(1, n_items + 1), years)],
        "release_date": [f"01-Jan-{y}" for y in years],
        "video_release_date": "",
        "imdb_url": "",
    })
    items = pd.concat([items, pd.DataFrame(genre_mat, columns=GENRES)], axis=1)
    items.to_csv(out_dir / "u.item", sep="|", header=False, index=False, encoding="latin-1")

    # Ratings from latent factors: genre taste + free factors + biases
    k = 8
    user_taste = rng.normal(0, 1, (n_users, len(GENRES)))
    user_f, item_f = rng.normal(0, 0.6, (n_users, k)), rng.normal(0, 0.6, (n_items, k))
    item_bias = rng.normal(0, 0.5, n_items)
    pop = 1.0 / np.arange(1, n_items + 1) ** 0.9            # power-law popularity
    pop = rng.permutation(pop)
    pop /= pop.sum()
    activity = rng.pareto(1.5, n_users) + 1
    activity /= activity.sum()

    u = rng.choice(n_users, n_ratings * 2, p=activity)
    i = rng.choice(n_items, n_ratings * 2, p=pop)
    pairs = pd.DataFrame({"u": u, "i": i}).drop_duplicates().head(n_ratings)
    u, i = pairs["u"].to_numpy(), pairs["i"].to_numpy()
    genre_score = (user_taste[u] * genre_mat[i]).sum(1) / np.maximum(genre_mat[i].sum(1), 1)
    raw = 3.5 + 0.6 * genre_score + (user_f[u] * item_f[i]).sum(1) + item_bias[i] + rng.normal(0, 0.6, len(u))
    rating = np.clip(np.rint(raw), 1, 5).astype(int)

    start, end = pd.Timestamp("1997-09-20").value // 10**9, pd.Timestamp("1998-04-22").value // 10**9
    ts = np.sort(rng.integers(start, end, len(u)))
    data = pd.DataFrame({"user_id": u + 1, "item_id": i + 1, "rating": rating,
                         "timestamp": rng.permutation(ts)})
    data.to_csv(out_dir / "u.data", sep="\t", header=False, index=False)
