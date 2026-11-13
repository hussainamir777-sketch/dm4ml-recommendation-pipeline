"""Stage 8 - Orchestration with Prefect.

The DAG:  ingest -> validate -> prepare -> transform -> features -> train -> evaluate

Each task receives the upstream task's result, so Prefect records the dependency
graph explicitly; a task only starts after its upstream task has succeeded.
Ingestion retries automatically (network download).

Run once:              python dags/recommendation_flow.py
Run on a schedule:     python dags/recommendation_flow.py --serve    (daily at 02:00)
See runs in the UI:    prefect server start   ->  http://127.0.0.1:4200
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make `src` importable

from prefect import flow, get_run_logger, task  # noqa: E402

from src.evaluation.evaluate import run_evaluation  # noqa: E402
from src.features.feature_store import run_feature_engineering  # noqa: E402
from src.ingestion.ingest import run_ingestion  # noqa: E402
from src.preparation.prepare import run_preparation  # noqa: E402
from src.training.train import run_training  # noqa: E402
from src.transformation.transform import run_transformation  # noqa: E402
from src.utils import load_config, set_seed  # noqa: E402
from src.validation.validate import run_validation  # noqa: E402


@task(name="1-data-ingestion", retries=2, retry_delay_seconds=10)
def ingest_task(cfg: dict) -> dict:
    """Download / generate raw data and save raw tables."""
    return run_ingestion(cfg)


@task(name="2-data-validation")
def validate_task(cfg: dict, upstream: dict) -> dict:
    """Run validation rules; split valid rows from the error dataset."""
    return run_validation(cfg)


@task(name="3-data-preparation")
def prepare_task(cfg: dict, upstream: dict) -> dict:
    """Clean the data and create the time-based train/test split."""
    return run_preparation(cfg)


@task(name="4-data-transformation")
def transform_task(cfg: dict, upstream: dict) -> dict:
    """Build user activity, item popularity and the interaction matrix."""
    return run_transformation(cfg)


@task(name="5-feature-engineering")
def features_task(cfg: dict, upstream: dict) -> dict:
    """Compute, register and document user / item / interaction features."""
    return run_feature_engineering(cfg)


@task(name="6-model-training")
def train_task(cfg: dict, upstream: dict) -> dict:
    """Tune and train all models with MLflow tracking."""
    return run_training(cfg)


@task(name="7-model-evaluation")
def evaluate_task(cfg: dict, upstream: dict) -> dict:
    """Evaluate all models on the test period and write the comparison table."""
    return run_evaluation(cfg)


@flow(name="movielens-recommendation-pipeline", log_prints=True)
def recommendation_pipeline(config_path: str | None = None) -> dict:
    """End-to-end recommendation pipeline with explicit task dependencies."""
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    logger = get_run_logger()
    logger.info("Starting pipeline (seed=%s, source=%s)", cfg["seed"], cfg["ingestion"]["source"])

    ingested = ingest_task(cfg)
    validated = validate_task(cfg, ingested)
    prepared = prepare_task(cfg, validated)
    transformed = transform_task(cfg, prepared)
    features = features_task(cfg, transformed)
    trained = train_task(cfg, features)
    evaluated = evaluate_task(cfg, trained)

    summary = {"ingestion": ingested, "validation": validated, "preparation": prepared,
               "transformation": transformed, "features": features, "training": trained,
               "evaluation": evaluated}
    logger.info("Pipeline finished successfully: %s", list(summary))
    return summary


if __name__ == "__main__":
    if "--serve" in sys.argv:
        # Long-running process that triggers the flow every day at 02:00 (Ctrl+C to stop)
        recommendation_pipeline.serve(name="daily-recommendation-pipeline", cron="0 2 * * *")
    else:
        recommendation_pipeline()
