import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# 1. Find the run "trial-2" (most recent one if there are several)
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    filter_string="tags.mlflow.runName = 'trial-2'",
    order_by=["attributes.start_time DESC"],
    max_results=1,
)
if runs.empty:
    raise SystemExit("Run 'trial-2' not found. Check the run name in the MLflow UI.")
run_id = runs.iloc[0]["run_id"]
print("trial-2 | id :", run_id[:8], "... | f1 :", round(runs.iloc[0]["metrics.f1"], 4))

# 2. Register it: creates a NEW VERSION of churn-model
v = mlflow.register_model(f"runs:/{run_id}/model", "churn-model")
print("New version created :", v.version)

# 3. Where does "champion" point today?
current = client.get_model_version_by_alias("churn-model", "champion")
print("Current champion : version", current.version)

# 4. Move the alias to the new version, then ALWAYS put it back
try:
    client.set_registered_model_alias("churn-model", "champion", v.version)
    print("After the move : champion -> version",
          client.get_model_version_by_alias("churn-model", "champion").version)
finally:
    client.set_registered_model_alias("churn-model", "champion", current.version)
    print("After going back : champion -> version",
          client.get_model_version_by_alias("churn-model", "champion").version)