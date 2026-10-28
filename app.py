from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Churn API")

# Load the champion ONCE, at startup
mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/churn-model@champion")
EXPECTED = list(model.feature_names_in_)


class Customer(BaseModel):
    age: int
    salaire: float
    abonnement: Literal["Basique", "Standard", "Premium"]
    anciennete: float
    consommation: float


@app.get("/health")
def health():
    return {"status": "ok"}
@app.get("/")
def root():
    return {"message": "Churn API is running. Go to /docs to test it."}

@app.post("/predict")
def predict(customer: Customer):
    data = customer.model_dump()

    # One-hot encode "abonnement" by hand (get_dummies on a single row is fragile).
    # "Basique" = the reference category: both columns stay at 0.
    sub = data.pop("abonnement")
    data["abonnement_Standard"] = int(sub == "Standard")
    data["abonnement_Premium"] = int(sub == "Premium")

    # Keep the model's columns, in the right order; missing ones = 0
    row = pd.DataFrame([data]).reindex(columns=EXPECTED, fill_value=0).astype(float)

    result = int(model.predict(row)[0])
    response = {
        "churn": result,
        "message": "The customer will leave" if result == 1 else "The customer stays",
    }
    if hasattr(model, "predict_proba"):
        response["probability_of_leaving"] = round(float(model.predict_proba(row)[0][1]), 3)
    return response