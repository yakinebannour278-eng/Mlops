# ===== 1. Import the libraries =====
import mlflow
from mlflow import MlflowClient

# ===== 2. Connect to the same database as Lab 3 =====
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# ===== 3. Find the best run (highest F1) =====
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    filter_string="metrics.f1 >= 0",      # ignore runs without an f1 metric
    order_by=["metrics.f1 DESC"],
    max_results=1,
)

if runs.empty:
    raise SystemExit("No run with an 'f1' metric found. Check that you are in the folder containing mlflow.db.")

best = runs.iloc[0]
run_id = best["run_id"]
run_name = best.get("tags.mlflow.runName", "n/a")
print("Best run :", run_name, "| id :", run_id[:8], "... | f1 :", round(best["metrics.f1"], 4))

# ===== 4. Register the model of this run in the Registry =====
result = mlflow.register_model(f"runs:/{run_id}/model", "churn-model")
print("Model registered : churn-model | version", result.version)

# ===== 5. Put the alias "champion" on this version =====
client = MlflowClient()
client.set_registered_model_alias("churn-model", "champion", result.version)
print("Alias 'champion' -> version", result.version)