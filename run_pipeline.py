"""Plain-Python runner for the same pipeline (no Prefect needed).

Usage:
    python run_pipeline.py                     # run every stage in order
    python run_pipeline.py --stages train evaluate   # run selected stages only
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

from src.evaluation.evaluate import run_evaluation
from src.features.feature_store import run_feature_engineering
from src.ingestion.ingest import run_ingestion
from src.preparation.prepare import run_preparation
from src.training.train import run_training
from src.transformation.transform import run_transformation
from src.utils import get_logger, get_path, load_config, set_seed, write_json
from src.validation.validate import run_validation

# Stage order = dependency order: each stage reads the files written by the previous one
STAGES = {
    "ingest": run_ingestion,
    "validate": run_validation,
    "prepare": run_preparation,
    "transform": run_transformation,
    "features": run_feature_engineering,
    "train": run_training,
    "evaluate": run_evaluation,
}


def main() -> None:
    """Run the selected stages in dependency order and write a run summary."""
    parser = argparse.ArgumentParser(description="Run the recommendation pipeline")
    parser.add_argument("--stages", nargs="+", choices=list(STAGES), default=list(STAGES))
    parser.add_argument("--config", default=None, help="path to an alternative config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    logger = get_logger("pipeline", cfg)
    selected = [s for s in STAGES if s in args.stages]          # always keep dependency order
    run = {"started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "stages": {}}

    for name in selected:
        logger.info("========== STAGE: %s ==========", name.upper())
        t0 = time.time()
        result = STAGES[name](cfg)
        run["stages"][name] = {"status": "success", "seconds": round(time.time() - t0, 2), "result": result}

    run["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_json(run, get_path(cfg, "reports") / "pipeline_run_summary.json")
    logger.info("Pipeline finished: %s", {k: v["seconds"] for k, v in run["stages"].items()})


if __name__ == "__main__":
    main()
