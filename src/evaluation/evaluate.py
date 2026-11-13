"""Stage 7 - Model evaluation.

Evaluates every trained model on the held-out (later-in-time) test period:
  * ranking: Precision@K, Recall@K, NDCG@K, HitRate@K (relevant = test rating >= 4)
  * beyond-accuracy: catalogue Coverage and intra-list Diversity (genre-based)
  * rating prediction: RMSE and MAE on all test ratings
Metrics are reported for all test users and separately for warm / cold-start
users, logged to each model's MLflow run, and saved as a comparison table + chart.
"""
from __future__ import annotations

import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from src.evaluation import metrics as M
from src.tracking import setup_mlflow
from src.utils import GENRES, get_logger, get_path, load_config, load_table, read_json, write_json

DISPLAY = {"popularity": "Popularity (baseline)", "matrix_factorization": "Matrix Factorization (SVD)",
           "hybrid": "Hybrid (MF + features)"}


def ranking_metrics(model, users: list, relevant: dict, k: int, genre_vec: dict, n_items: int) -> dict:
    """Average ranking metrics over `users`, plus coverage and diversity of their lists."""
    if not users:
        return {}
    recs = model.recommend(users, k=k)
    rows = [{"precision": M.precision_at_k(recs[u], relevant[u], k),
             "recall": M.recall_at_k(recs[u], relevant[u], k),
             "ndcg": M.ndcg_at_k(recs[u], relevant[u], k),
             "hit_rate": M.hit_rate_at_k(recs[u], relevant[u], k),
             "diversity": M.intra_list_diversity(recs[u], genre_vec)} for u in users]
    df = pd.DataFrame(rows).mean()
    return {f"precision_at_{k}": df["precision"], f"recall_at_{k}": df["recall"],
            f"ndcg_at_{k}": df["ndcg"], f"hit_rate_at_{k}": df["hit_rate"],
            "coverage": M.catalog_coverage(recs, n_items), "diversity": df["diversity"]}


def to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a Markdown table (no extra dependency needed)."""
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def plot_comparison(table: pd.DataFrame, k: int, path) -> None:
    """Grouped bar chart of the main metrics for every model."""
    cols = [f"precision_at_{k}", f"recall_at_{k}", f"ndcg_at_{k}", "coverage", "diversity"]
    ax = table.set_index("model")[cols].T.plot(kind="bar", figsize=(10, 4.5), rot=0)
    ax.set_title(f"Model comparison on the time-based test set (K = {k}, all test users)")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_evaluation(cfg: dict | None = None) -> dict:
    """Evaluate all models, log to MLflow and write the results table."""
    cfg = cfg or load_config()
    logger = get_logger("evaluation", cfg)
    processed, features_dir, models_dir, reports = (get_path(cfg, p) for p in
                                                    ["processed", "features", "models", "reports"])
    k, thr = cfg["evaluation"]["k"], cfg["evaluation"]["relevance_threshold"]

    test = load_table(processed, "test")
    item_feats = load_table(features_dir, "item_features")
    genre_vec = dict(zip(item_feats["item_id"], item_feats[GENRES].to_numpy()))
    n_items = len(item_feats)
    registry = read_json(models_dir / "model_registry.json")
    setup_mlflow(cfg)

    relevant = test[test["rating"] >= thr].groupby("user_id")["item_id"].apply(set).to_dict()
    eval_users = sorted(relevant)
    models = {}
    for name in registry:
        with open(models_dir / f"{name}.pkl", "rb") as f:
            models[name] = pickle.load(f)
    mf = models["matrix_factorization"]
    warm = [u for u in eval_users if mf.is_warm(u)]
    cold = [u for u in eval_users if not mf.is_warm(u)]
    logger.info("Evaluating %d test users with >= 1 relevant item (warm=%d, cold-start=%d), K=%d",
                len(eval_users), len(warm), len(cold), k)

    rows, segment_rows = [], []
    for name, model in models.items():
        res = ranking_metrics(model, eval_users, relevant, k, genre_vec, n_items)
        preds = model.predict_rating(test["user_id"].tolist(), test["item_id"].tolist())
        res.update({"rmse": M.rmse(test["rating"], preds), "mae": M.mae(test["rating"], preds)})
        rows.append({"model": DISPLAY[name], **res})
        for seg, users in [("warm", warm), ("cold-start", cold)]:
            s = ranking_metrics(model, users, relevant, k, genre_vec, n_items)
            segment_rows.append({"model": DISPLAY[name], "segment": seg, "users": len(users), **s})

        with mlflow.start_run(run_id=registry[name]["mlflow_run_id"]):
            mlflow.log_metrics({f"test_{m}": float(v) for m, v in res.items()})
            mlflow.log_param("eval_k", k)
        logger.info("%-28s P@%d=%.4f R@%d=%.4f NDCG@%d=%.4f Cov=%.3f Div=%.3f RMSE=%.4f MAE=%.4f",
                    DISPLAY[name], k, res[f"precision_at_{k}"], k, res[f"recall_at_{k}"], k,
                    res[f"ndcg_at_{k}"], res["coverage"], res["diversity"], res["rmse"], res["mae"])

    table = pd.DataFrame(rows)
    segments = pd.DataFrame(segment_rows)
    table.to_csv(reports / "evaluation_results.csv", index=False)
    segments.to_csv(reports / "evaluation_by_segment.csv", index=False)
    md = ["# Model Evaluation Results", "",
          f"Time-based split; relevant = test rating >= {thr}; K = {k}; "
          f"{len(eval_users)} evaluated users ({len(warm)} warm, {len(cold)} cold-start).", "",
          "## All test users", "", to_markdown(table.round(4)), "",
          "## By user segment", "", to_markdown(segments.round(4))]
    (reports / "evaluation_results.md").write_text("\n".join(md), encoding="utf-8")
    plot_comparison(table, k, reports / "model_comparison.png")

    write_json({"k": k, "evaluated_users": len(eval_users), "warm_users": len(warm),
                "cold_users": len(cold), "results": table.round(6).to_dict("records")},
               reports / "evaluation_results.json")
    logger.info("Results written to reports/evaluation_results.md and reports/model_comparison.png")
    print("\n" + table.round(4).to_string(index=False))
    return {"stage": "evaluation", "evaluated_users": len(eval_users)}


if __name__ == "__main__":
    run_evaluation()
