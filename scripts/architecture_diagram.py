"""Draw the pipeline architecture diagram -> reports/architecture.png

Usage:  python scripts/architecture_diagram.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "reports" / "architecture.png"

# (x, y, title, detail, colour)
STAGES = [
    (0.5, 6.0, "1. Ingestion", "MovieLens 100K download\n+ documented fault injection\n-> data/raw/*.csv", "#dbe9f6"),
    (4.0, 6.0, "2. Validation", "missing / types / ranges /\ntimestamps / duplicates /\nreferential integrity", "#dbe9f6"),
    (7.5, 6.0, "3. Preparation", "cleaning, imputation,\ndedup, time-based\ntrain/test split", "#dbe9f6"),
    (11.0, 6.0, "4. Transformation", "user activity, item\npopularity, sparse\nuser-item matrix", "#dbe9f6"),
    (11.0, 2.6, "5. Feature store", "user / item / interaction\nfeatures + registry +\ncatalog (Parquet)", "#e3f1df"),
    (7.5, 2.6, "6. Training", "Popularity, MF (SVD,\nvalidation-tuned), Hybrid\n-> models/*.pkl", "#e3f1df"),
    (4.0, 2.6, "7. Evaluation", "P/R/NDCG@10, HitRate,\nCoverage, Diversity,\nRMSE / MAE", "#e3f1df"),
    (0.5, 2.6, "Serving", "recommend.py uses the\nsame feature store and\nsaved models", "#fbe9d6"),
]
SIDE = [  # (x, y, width, text, colour)
    (0.5, 9.0, 3.4, "configs/config.yaml (one config, seed = 42)", "#eeeeee"),
    (4.2, 9.0, 4.6, "data/errors/  +  reports/validation_report.md", "#f8d7d7"),
    (4.0, 0.55, 6.5, "MLflow tracking: params, metrics, models (mlflow.db + mlruns/)", "#efe3f6"),
]
W, H = 3.0, 1.9

fig, ax = plt.subplots(figsize=(15, 8.6))
ax.set_xlim(0, 14.5)
ax.set_ylim(0.0, 10.2)
ax.axis("off")
ax.text(7.25, 10.0, "End-to-end recommendation pipeline (orchestrated by Prefect: dags/recommendation_flow.py)",
        ha="center", va="center", fontsize=14, weight="bold")

for x, y, title, detail, colour in STAGES:
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.05", fc=colour, ec="#555555", lw=1.2))
    ax.text(x + W / 2, y + H - 0.35, title, ha="center", va="center", fontsize=11.5, weight="bold")
    ax.text(x + W / 2, y + 0.75, detail, ha="center", va="center", fontsize=9.2)

for x, y, w, text, colour in SIDE:
    ax.add_patch(FancyBboxPatch((x, y), w, 0.55, boxstyle="round,pad=0.05", fc=colour, ec="#777777", lw=1))
    ax.text(x + w / 2, y + 0.27, text, ha="center", va="center", fontsize=8.8)


def arrow(x1, y1, x2, y2, style="-|>", ls="-"):
    """Draw an arrow between two points."""
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16,
                                 lw=1.5, color="#333333", linestyle=ls))


# main flow
arrow(3.5, 6.95, 4.0, 6.95)
arrow(7.0, 6.95, 7.5, 6.95)
arrow(10.5, 6.95, 11.0, 6.95)
arrow(12.5, 6.0, 12.5, 4.5)
arrow(11.0, 3.55, 10.5, 3.55)
arrow(7.5, 3.55, 7.0, 3.55)
arrow(4.0, 3.55, 3.5, 3.55, ls="--")
# side outputs
arrow(5.5, 7.9, 5.5, 9.0, ls=":")
arrow(9.0, 2.6, 9.0, 1.15, ls=":")
arrow(5.5, 2.6, 5.5, 1.15, ls=":")

ax.text(7.25, 0.1, "Stages 1-7 = Prefect tasks with explicit dependencies;  run_pipeline.py runs the same "
        "stages without Prefect.", ha="center", fontsize=9.5, style="italic")
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved {OUT}")
