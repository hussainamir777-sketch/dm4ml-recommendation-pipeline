"""MLflow experiment-tracking helpers (local SQLite backend + local artifact folder)."""
from __future__ import annotations

import mlflow

from src.utils import PROJECT_ROOT


def setup_mlflow(cfg: dict) -> str:
    """Point MLflow at the project's tracking database and select the experiment."""
    mcfg = cfg["mlflow"]
    uri = mcfg["tracking_uri"]
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = f"sqlite:///{PROJECT_ROOT / uri.replace('sqlite:///', '')}"   # make the DB path absolute
    mlflow.set_tracking_uri(uri)
    name = mcfg["experiment_name"]
    if mlflow.get_experiment_by_name(name) is None:
        artifacts = (PROJECT_ROOT / mcfg.get("artifact_location", "mlruns")).resolve()
        artifacts.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(name, artifact_location=artifacts.as_uri())
    mlflow.set_experiment(name)
    return uri
