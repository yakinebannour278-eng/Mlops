import mlflow
import mlflow.sklearn
import pandas as pd

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 1. Load the champion by its alias
model = mlflow.sklearn.load_model("models:/churn-model@champion")
print("Model loaded :", type(model).__name__)

expected = list(model.feature_names_in_)
print("Expected variables :", expected)

# 2. New customer (Standard subscription)
customer = pd.DataFrame([{
    "age": 35,
    "salaire": 2500,
    "anciennete": 1,
    "consommation": 10,
    "abonnement_Premium": False,
    "abonnement_Standard": True,
}])

# 3. Keep only the model's columns, in the right order
#    (a column missing from the customer is filled with 0 instead of crashing)
customer = customer.reindex(columns=expected, fill_value=0)

# 4. Predict: 1 = leaves, 0 = stays
pred = model.predict(customer)[0]
print("Will the customer leave? ->", "YES" if pred == 1 else "NO")

if hasattr(model, "predict_proba"):
    print("Probability of leaving :", round(model.predict_proba(customer)[0][1], 3))