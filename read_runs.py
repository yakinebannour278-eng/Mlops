import mlflow  # we reuse the MLflow module to READ the results

# search_runs returns all the runs of the experiment as a pandas table.
# order_by sorts the rows: "metrics.f1 DESC" = f1 descending (best first)
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"])

# Columns to display: run name, two parameters, two metrics
columns = ["tags.mlflow.runName", "params.n_estimators", "params.max_depth",
           "metrics.accuracy", "metrics.f1"]
print(runs[columns].to_string(index=False))  # index=False: without row numbers

# The first row of the sorted table is the best run
best = runs.iloc[0]
print()
# [:8]: we only display the first 8 characters of the run identifier
print("Best run :", best["tags.mlflow.runName"], "| id :", best["run_id"][:8], "...")