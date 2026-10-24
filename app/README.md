# Application extension

Reserved for the M2 application. The team has not chosen a GUI framework. M1 uses the shared `main.py` command routes for reproducible experiments.

The final CLI or GUI must expose dataset analysis, model comparisons, tuning results, clustering, evaluation information, and both prediction tasks. Inference loads saved pipelines instead of retraining on startup. Reuse `src/` modules rather than duplicating model logic in a UI.

