from fastapi.testclient import TestClient
from starter.main import app

# Initialize TestClient
client = TestClient(app)


def test_get_root():
    """Test the GET / endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the Census Income Inference API!"}


def test_post_inference_low_income():
    """Test the POST /inference endpoint with <=50K prediction."""
    payload = {
        "age": 25,
        "workclass": "Private",
        "education": "HS-grad",
        "marital-status": "Never-married",
        "occupation": "Handlers-cleaners",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "native-country": "United-States",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40
    }
    response = client.post("/inference", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
    # Adjust based on expected output
    assert response.json()["prediction"] == "<=50K"


def test_post_inference_high_income():
    """Test the POST /inference endpoint with >50K prediction."""
    payload = {
        "age": 45,
        "workclass": "Private",
        "education": "Bachelors",
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "native-country": "United-States",
        "capital-gain": 10000,
        "capital-loss": 0,
        "hours-per-week": 60
    }
    response = client.post("/inference", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
    # Adjust based on expected output
    assert response.json()["prediction"] == ">50K"
