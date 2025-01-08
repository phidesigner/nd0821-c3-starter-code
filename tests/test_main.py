from fastapi.testclient import TestClient
from starter.main import app

# Initialize the TestClient for the FastAPI app
client = TestClient(app)

# 1. Test for the / GET endpoint


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the Census Income Inference API!"}

# 2. Test for the /inference POST endpoint (Prediction 1)


def test_predict_income_1():
    # Sample payload for prediction 1
    payload = {
        "age": 34,
        "workclass": "Private",
        "education": "Bachelors",
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "native-country": "United-States",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "education-num": 13,
        "fnlgt": 77516
    }

    response = client.post("/inference", json=payload)

    assert response.status_code == 200
    # Ensure the returned prediction is one of the expected values (assuming it predicts income as <=50K or >50K)
    assert response.json()['prediction'] in ["<=50K", ">50K"]

# 3. Test for the /inference POST endpoint (Prediction 2)


def test_predict_income_2():
    # Sample payload for prediction 2
    payload = {
        "age": 45,
        "workclass": "Self-emp-not-inc",
        "education": "Bachelors",
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "native-country": "United-States",
        "capital-gain": 10000,
        "capital-loss": 0,
        "hours-per-week": 50,
        "education-num": 13,
        "fnlgt": 83311
    }

    response = client.post("/inference", json=payload)

    assert response.status_code == 200
    # Ensure the returned prediction is one of the expected values (assuming it predicts income as <=50K or >50K)
    assert response.json()['prediction'] in ["<=50K", ">50K"]
