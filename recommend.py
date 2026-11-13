"""Prediction / serving entry point: top-K recommendations for one user.

Uses the SAME feature store as training, so features are identical at
training and prediction time.

Usage:
    python recommend.py --user 42                 # hybrid model, K from config
    python recommend.py --user 42 --model popularity --k 5
"""
from __future__ import annotations

import argparse
import pickle

from src.features.feature_store import FeatureStore
from src.utils import get_path, load_config


def main() -> None:
    """Print a user's profile features and their top-K recommendations."""
    parser = argparse.ArgumentParser(description="Recommend movies for a user")
    parser.add_argument("--user", type=int, required=True)
    parser.add_argument("--model", default="hybrid", choices=["popularity", "matrix_factorization", "hybrid"])
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    k = args.k or cfg["evaluation"]["k"]
    with open(get_path(cfg, "models") / f"{args.model}.pkl", "rb") as f:
        model = pickle.load(f)
    store = FeatureStore(cfg)

    profile = store.get_user_features([args.user]).iloc[0]
    print(f"\nUser {args.user}: age={profile['age']}, gender={profile['gender']}, "
          f"occupation={profile['occupation']}, ratings={profile['user_rating_count']}, "
          f"avg rating={profile['user_avg_rating']:.2f}, preferred genre={profile['user_preferred_genre']}, "
          f"cold start={bool(profile['is_cold_start_user'])}")

    recs = model.recommend([args.user], k=k)[args.user]
    items = store.get_item_features(recs)
    match = store.genre_match([args.user], recs)[0]
    print(f"\nTop-{k} recommendations ({args.model}):")
    for rank, (item_id, row) in enumerate(items.iterrows(), start=1):
        genres = "|".join(g for g in ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
                                      "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
                                      "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"] if row[g] == 1)
        print(f"{rank:>2}. {row['title'][:45]:<45} | {genres:<28} | "
              f"pop rank {row['item_popularity_rank']:>4} | genre match {match[rank - 1]:.3f}")


if __name__ == "__main__":
    main()
