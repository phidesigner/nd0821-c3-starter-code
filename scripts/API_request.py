import requests

url = "https://humble-robot-r94grr9vv42w5qx-8000.app.github.dev/inference"
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
    "hours-per-week": 40
}

response = requests.post(url, json=payload)
print(response.json())
