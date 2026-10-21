import mlflow
import mlflow.sklearn  # to reload a scikit-learn model
import pandas as pd

# 1. Find the best run (same logic as read_runs.py)
runs = mlflow.search_runs(experiment_names=["churn-prediction"],
                          order_by=["metrics.f1 DESC"])
run_id = runs.iloc[0]["run_id"]  # identifier of the best run

# 2. Reload the model. "runs:/<id>/model" = the 'model' folder of this run
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
print("Model reloaded :", type(model).__name__)
print("Number of expected variables :", model.n_features_in_)

# 3. Create a new customer (values of your choice)
# The subscription columns are already converted to 0/1 (True/False):
# here: Standard subscription (Premium = False)
customer = pd.DataFrame([{
    "age": 35, "salaire": 2500, "anciennete": 1, "consommation": 10,
    "abonnement_Premium": False, "abonnement_Standard": True,
}])

# 4. Keep only the columns known to the model, in the right order
# (depending on the chosen run, the salary column is used or not)
customer = customer[list(model.feature_names_in_)]

# 5. Predict: 1 = the customer will leave, 0 = stays
print("Will the customer leave? ->", "YES" if model.predict(customer)[0] == 1 else "NO")