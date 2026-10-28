import requests

URL = "http://127.0.0.1:8000"

# 1. Is the API alive?
print("GET /health ->", requests.get(URL + "/health", timeout=10).json())

# 2. A customer with low usage and low seniority
customer = {"age": 35, "salaire": 2500, "abonnement": "Standard",
            "anciennete": 1, "consommation": 10}
r = requests.post(URL + "/predict", json=customer, timeout=10)
print("POST /predict (customer 1) ->", r.status_code, r.json())

# 3. A loyal customer: high usage, high seniority
customer2 = {"age": 50, "salaire": 4000, "abonnement": "Premium",
             "anciennete": 10, "consommation": 80}
r = requests.post(URL + "/predict", json=customer2, timeout=10)
print("POST /predict (customer 2) ->", r.status_code, r.json())

# 4. A deliberate error: the age is text
bad = dict(customer, age="abc")
r = requests.post(URL + "/predict", json=bad, timeout=10)
print("POST /predict (age='abc') ->", r.status_code)
print("Message :", r.json()["detail"][0]["msg"])