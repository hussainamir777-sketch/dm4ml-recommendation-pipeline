"""Recommendation models.

  * PopularityRecommender - non-personalized baseline (most-liked items overall)
  * MFRecommender         - matrix factorization (truncated SVD) collaborative filtering
  * HybridRecommender     - MF score + user-genre affinity + popularity from the feature store

All models share one interface: fit(...), recommend(user_ids, k, exclude) and
predict_rating(user_ids, item_ids), so evaluation treats them identically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


def _top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Column indices of the k highest scores in each row, best first."""
    idx = np.argpartition(-scores, kth=min(k, scores.shape[1] - 1), axis=1)[:, :k]
    order = np.take_along_axis(scores, idx, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(idx, order, axis=1)


class BaseRecommender:
    """Common logic: id <-> index mapping, masking already-seen items, top-k selection."""
    name = "base"

    def _setup(self, matrix: sparse.csr_matrix, user_index: pd.DataFrame, item_index: pd.DataFrame):
        """Store the training matrix and id mappings shared by every model."""
        self.matrix = matrix.tocsr()
        self.user_row = dict(zip(user_index["user_id"], user_index["row"]))
        self.item_ids = item_index.sort_values("col")["item_id"].to_numpy()
        self.item_col = dict(zip(item_index["item_id"], item_index["col"]))

    def score(self, user_ids) -> np.ndarray:
        """(n_users x n_items) ranking scores - implemented by each model."""
        raise NotImplementedError

    def recommend(self, user_ids, k: int = 10, exclude_seen: bool = True) -> dict:
        """Top-k item ids per user; items the user rated in training are excluded."""
        user_ids = list(user_ids)
        scores = self.score(user_ids).astype(float)
        if exclude_seen:
            for r, u in enumerate(user_ids):
                row = self.user_row.get(u)
                if row is not None:
                    scores[r, self.matrix[row].indices] = -np.inf
        top = _top_k(scores, k)
        return {u: self.item_ids[top[r]].tolist() for r, u in enumerate(user_ids)}


class PopularityRecommender(BaseRecommender):
    """Baseline: rank items by number of positive (>= 4) training ratings.

    Rating prediction uses the item's Bayesian-average rating (global mean if unseen).
    """
    name = "popularity"

    def __init__(self, min_ratings: int = 20, damping: float = 10.0):
        self.min_ratings, self.damping = min_ratings, damping

    def fit(self, matrix, user_index, item_index, **_):
        """Count positive ratings per item and compute shrunk item means."""
        self._setup(matrix, user_index, item_index)
        m = self.matrix
        counts = np.asarray((m > 0).sum(axis=0)).ravel()
        positives = np.asarray((m >= 4).sum(axis=0)).ravel()
        sums = np.asarray(m.sum(axis=0)).ravel()
        self.global_mean = m.data.mean()
        self.item_mean = (sums + self.damping * self.global_mean) / (counts + self.damping)
        # Items with too few ratings are pushed to the bottom of the ranking
        self.pop_score = np.where(counts >= self.min_ratings, positives, positives * 1e-3).astype(float)
        return self

    def score(self, user_ids):
        """Every user gets the same popularity ranking."""
        return np.tile(self.pop_score, (len(list(user_ids)), 1))

    def predict_rating(self, user_ids, item_ids) -> np.ndarray:
        """Predicted rating = Bayesian item mean (global mean for unknown items)."""
        cols = [self.item_col.get(i) for i in item_ids]
        return np.array([self.item_mean[c] if c is not None else self.global_mean for c in cols])


class MFRecommender(BaseRecommender):
    """Matrix factorization with two heads.

    * Ranking head: truncated SVD of the user x item rating matrix (PureSVD);
      score(u, i) = reconstructed value. Strong for top-K ranking.
    * Rating head: global mean + regularized user/item biases + truncated SVD of
      the bias residuals on observed entries; used for RMSE / MAE.
    Cold-start users (no training ratings) fall back to the popularity ranking.
    """
    name = "matrix_factorization"

    def __init__(self, n_factors: int = 50, rating_factors: int = 20, bias_reg: float = 10.0, seed: int = 42):
        self.n_factors, self.rating_factors, self.bias_reg, self.seed = n_factors, rating_factors, bias_reg, seed

    def fit(self, matrix, user_index, item_index, popularity: PopularityRecommender, **_):
        """Factorize the training matrix and fit the rating-prediction head."""
        self._setup(matrix, user_index, item_index)
        self.popularity = popularity
        m = self.matrix.astype(float)

        # --- ranking head (PureSVD) ---
        svd = TruncatedSVD(n_components=self.n_factors, random_state=self.seed)
        self.user_factors = svd.fit_transform(m)            # U * Sigma
        self.item_factors = svd.components_                 # V^T
        self.explained_variance = float(svd.explained_variance_ratio_.sum())

        # --- rating head (biases + residual SVD) ---
        coo = m.tocoo()
        self.mu = coo.data.mean()
        n_u, n_i = m.shape
        dev = coo.data - self.mu
        self.b_i = np.bincount(coo.col, dev, n_i) / (np.bincount(coo.col, minlength=n_i) + self.bias_reg)
        dev_u = coo.data - self.mu - self.b_i[coo.col]
        self.b_u = np.bincount(coo.row, dev_u, n_u) / (np.bincount(coo.row, minlength=n_u) + self.bias_reg)
        resid = coo.data - (self.mu + self.b_u[coo.row] + self.b_i[coo.col])
        r_mat = sparse.csr_matrix((resid, (coo.row, coo.col)), shape=m.shape)
        rsvd = TruncatedSVD(n_components=self.rating_factors, random_state=self.seed)
        self.res_u = rsvd.fit_transform(r_mat)
        self.res_i = rsvd.components_
        self.warm_rows = set(np.unique(coo.row).tolist())
        train_pred = self._rating_from_rows(coo.row, coo.col)
        self.train_rmse = float(np.sqrt(np.mean((train_pred - coo.data) ** 2)))
        return self

    def is_warm(self, user_id) -> bool:
        """True if the user has at least one training rating."""
        return self.user_row.get(user_id) in self.warm_rows

    def score(self, user_ids):
        """Reconstructed preference scores; popularity scores for cold-start users."""
        user_ids = list(user_ids)
        out = np.zeros((len(user_ids), len(self.item_ids)))
        for r, u in enumerate(user_ids):
            if self.is_warm(u):
                out[r] = self.user_factors[self.user_row[u]] @ self.item_factors
            else:
                out[r] = self.popularity.pop_score
        return out

    def _rating_from_rows(self, rows, cols) -> np.ndarray:
        """Rating-head prediction for (row, col) index arrays, clipped to [1, 5]."""
        pred = self.mu + self.b_u[rows] + self.b_i[cols] + np.einsum("ij,ji->i", self.res_u[rows], self.res_i[:, cols])
        return np.clip(pred, 1, 5)

    def predict_rating(self, user_ids, item_ids) -> np.ndarray:
        """Predicted ratings; unknown users/items fall back to the bias terms they have."""
        preds = []
        for u, i in zip(user_ids, item_ids):
            r, c = self.user_row.get(u), self.item_col.get(i)
            if r is not None and c is not None and r in self.warm_rows:
                preds.append(self._rating_from_rows(np.array([r]), np.array([c]))[0])
            else:
                preds.append(float(np.clip(self.mu + (self.b_i[c] if c is not None else 0.0), 1, 5)))
        return np.array(preds)


class HybridRecommender(BaseRecommender):
    """Weighted blend of collaborative and content/popularity signals from the feature store:

        score = w_mf * MF_score(min-max scaled per user)
              + w_genre * user_item_genre_match(scaled)
              + w_popularity * item_popularity_score

    For cold-start users the MF term is unavailable, so the ranking is driven by
    popularity, nudged by the user's segment-based genre affinity:
        score = (1 - cold_w_genre) * popularity + cold_w_genre * genre_match
    """
    name = "hybrid"

    def __init__(self, w_mf: float = 0.7, w_genre: float = 0.2, w_popularity: float = 0.1,
                 cold_w_genre: float = 0.2):
        self.w_mf, self.w_genre, self.w_pop, self.cold_w_genre = w_mf, w_genre, w_popularity, cold_w_genre

    def fit(self, matrix, user_index, item_index, mf: MFRecommender, feature_store, **_):
        """Reuse the fitted MF model and attach the feature store."""
        self._setup(matrix, user_index, item_index)
        self.mf, self.fs = mf, feature_store
        self.item_pop = feature_store.get_item_features(self.item_ids)["item_popularity_score"].fillna(0).to_numpy()
        return self

    @staticmethod
    def _minmax(x: np.ndarray) -> np.ndarray:
        """Scale each row to [0, 1]."""
        lo, hi = x.min(axis=1, keepdims=True), x.max(axis=1, keepdims=True)
        return (x - lo) / np.where(hi - lo > 0, hi - lo, 1)

    def score(self, user_ids):
        """Blend MF, genre-match and popularity scores."""
        user_ids = list(user_ids)
        mf = self._minmax(self.mf.score(user_ids))
        genre = self._minmax(self.fs.genre_match(user_ids, self.item_ids))
        pop = np.tile(self.item_pop, (len(user_ids), 1))
        warm = np.array([self.mf.is_warm(u) for u in user_ids])[:, None]
        warm_score = self.w_mf * mf + self.w_genre * genre + self.w_pop * pop
        cold_pop = self._minmax(self.popularity_scores(len(user_ids)))
        cold_score = (1 - self.cold_w_genre) * cold_pop + self.cold_w_genre * genre
        return np.where(warm, warm_score, cold_score)

    def popularity_scores(self, n_users: int) -> np.ndarray:
        """The baseline popularity ranking, repeated for n_users rows."""
        return np.tile(self.mf.popularity.pop_score, (n_users, 1))

    def predict_rating(self, user_ids, item_ids) -> np.ndarray:
        """Ratings come from the MF rating head (the hybrid only changes the ranking)."""
        return self.mf.predict_rating(user_ids, item_ids)
