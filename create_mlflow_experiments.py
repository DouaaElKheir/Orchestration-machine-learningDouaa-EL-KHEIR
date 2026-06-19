import requests
import json

MLFLOW_URL = "http://localhost:5001/api/2.0/mlflow"

# Créer expérience 1
exp1_data = {
    "name": "Adult Income Classification v1",
    "tags": [{"key": "project", "value": "adult_income"}, {"key": "version", "value": "1.0"}]
}
response = requests.post(f"{MLFLOW_URL}/experiments/create", json=exp1_data)
print(f"Expérience 1 créée: {response.json().get('experiment_id')}")

# Créer expérience 2
exp2_data = {
    "name": "Adult Income Classification v2", 
    "tags": [{"key": "project", "value": "adult_income"}, {"key": "version", "value": "2.0"}]
}
response = requests.post(f"{MLFLOW_URL}/experiments/create", json=exp2_data)
print(f"Expérience 2 créée: {response.json().get('experiment_id')}")

# Créer expérience 3
exp3_data = {
    "name": "Adult Income Classification v3",
    "tags": [{"key": "project", "value": "adult_income"}, {"key": "version", "value": "3.0"}]
}
response = requests.post(f"{MLFLOW_URL}/experiments/create", json=exp3_data)
print(f"Expérience 3 créée: {response.json().get('experiment_id')}")

print("✅ 3 expériences MLflow créées avec succès!")

