"""Stage 6 - Model training with MLflow experiment tracking.

1. Hyper-parameter selection: the last 10% of the TRAINING period is held out as a
   validation slice; the number of MF latent factors is chosen by NDCG@K on it
   (the test set is never used for tuning).
2. Final training: popularity baseline, matrix factorization (best factors) and the
   hybrid model are fitted on the full training period and saved to models/.
3. Every model gets one MLflow run with its parameters, metrics and artifacts.
"""
from __future__ import annotations

import pickle
import time

import mlflow
import numpy as np
from scipy import sparse

from src.evaluation.metrics import ndcg_at_k
from src.features.feature_store import FeatureStore
from src.tracking import setup_mlflow
from src.training.models import HybridRecommender, MFRecommender, PopularityRecommender
from src.transformation.transform import build_interaction_matrix
from src.utils import (DEFAULT_CONFIG, get_logger, get_path, load_config, load_table, read_json,
                       set_seed, write_json)


def tune_mf_factors(cfg: dict, logger) -> tuple[int, dict]:
    """Pick the MF latent dimension by NDCG@K on a time-based validation slice of the training data."""
    processed = get_path(cfg, "processed")
    train = load_table(processed, "train")
    users, items = load_table(processed, "users_clean"), load_table(processed, "items_clean")
    k, thr = cfg["evaluation"]["k"], cfg["evaluation"]["relevance_threshold"]
    mcfg = cfg["models"]

    cut = train["timestamp"].quantile(0.9)
    inner, val = train[train["timestamp"] <= cut], train[train["timestamp"] > cut]
    matrix, user_index, item_index = build_interaction_matrix(inner, users, items)
    pop = PopularityRecommender(**mcfg["popularity"]).fit(matrix, user_index, item_index)
    relevant = val[val["rating"] >= thr].groupby("user_id")["item_id"].apply(set).to_dict()

    scores = {}
    params = {k_: v for k_, v in mcfg["matrix_factorization"].items() if k_ != "n_factors_grid"}
    for n in mcfg["matrix_factorization"]["n_factors_grid"]:
        params["n_factors"] = n
        mf = MFRecommender(**params, seed=cfg["seed"]).fit(matrix, user_index, item_index, popularity=pop)
        warm_users = [u for u in relevant if mf.is_warm(u)]
        recs = mf.recommend(warm_users, k=k)
        scores[n] = float(np.mean([ndcg_at_k(recs[u], relevant[u], k) for u in warm_users]))
        logger.info("  validation: n_factors=%-4d NDCG@%d=%.4f (%d warm validation users)",
                    n, k, scores[n], len(warm_users))
    best = max(scores, key=scores.get)
    logger.info("Selected n_factors=%d (best validation NDCG@%d)", best, k)
    return best, scores


def run_training(cfg: dict | None = None) -> dict:
    """Tune, fit all models, save them and track every run in MLflow."""
    cfg = cfg or load_config()
    logger = get_logger("training", cfg)
    set_seed(cfg["seed"])
    processed, features_dir, models_dir = (get_path(cfg, k) for k in ["processed", "features", "models"])
    mcfg = cfg["models"]

    best_factors, val_scores = tune_mf_factors(cfg, logger)

    matrix = sparse.load_npz(processed / "user_item_matrix.npz")
    user_index, item_index = load_table(processed, "user_index"), load_table(processed, "item_index")
    split = read_json(processed / "split_info.json")
    store = FeatureStore(cfg)
    setup_mlflow(cfg)

    mf_params = {k_: v for k_, v in mcfg["matrix_factorization"].items() if k_ != "n_factors_grid"}
    mf_params["n_factors"] = best_factors
    pop = PopularityRecommender(**mcfg["popularity"])
    mf = MFRecommender(**mf_params, seed=cfg["seed"])
    hybrid = HybridRecommender(**mcfg["hybrid"])
    plan = [  # (model, fit kwargs, params to log, extra metrics)
        (pop, {}, mcfg["popularity"], {}),
        (mf, {"popularity": pop}, {**mf_params, "n_factors_grid": str(mcfg["matrix_factorization"]["n_factors_grid"])},
         {f"val_ndcg_n{n}": s for n, s in val_scores.items()}),
        (hybrid, {"mf": mf, "feature_store": store}, {**mcfg["hybrid"], "mf_n_factors": best_factors}, {}),
    ]

    registry = {}
    for model, kwargs, params, extra in plan:
        with mlflow.start_run(run_name=f"train_{model.name}") as run:
            t0 = time.time()
            model.fit(matrix, user_index, item_index, **kwargs)
            fit_time = time.time() - t0

            path = models_dir / f"{model.name}.pkl"
            with open(path, "wb") as f:
                pickle.dump(model, f)

            mlflow.set_tags({"stage": "training", "model": model.name, "dataset": cfg["ingestion"]["source"],
                             "feature_set_version": store.registry["feature_set_version"]})
            mlflow.log_params({**params, "seed": cfg["seed"], "train_rows": split["train_rows"],
                               "split_cutoff": split["cutoff_datetime_utc"],
                               "matrix_shape": f"{matrix.shape[0]}x{matrix.shape[1]}"})
            metrics = {"fit_time_seconds": round(fit_time, 4), **extra}
            if isinstance(model, MFRecommender):
                metrics.update({"train_rmse": model.train_rmse, "svd_explained_variance": model.explained_variance})
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(path), artifact_path="model")
            mlflow.log_artifact(str(DEFAULT_CONFIG), artifact_path="config")
            mlflow.log_artifact(str(features_dir / "feature_registry.json"), artifact_path="features")

            registry[model.name] = {"path": str(path.relative_to(models_dir.parent)),
                                    "mlflow_run_id": run.info.run_id, **metrics}
            shown = {k_: round(v, 4) for k_, v in metrics.items() if k_ in ("train_rmse", "svd_explained_variance")}
            logger.info("Trained %-22s in %.2fs %s (MLflow run %s)", model.name, fit_time, shown, run.info.run_id[:8])

    write_json(registry, models_dir / "model_registry.json")
    logger.info("Saved %d models and models/model_registry.json", len(registry))
    return {"stage": "training", "models": list(registry), "mf_n_factors": best_factors}


if __name__ == "__main__":
    run_training()
