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
    "hours-per-week": 40,
    "education-num": 13,
    "fnlgt": 77516
}

response = requests.post(url, json=payload)

# Check status code and response content
print("Status Code:", response.status_code)
print("Response Text:", response.text)

# Try to decode JSON if the response is valid
if response.status_code == 200:
    try:
        print(response.json())
    except ValueError as e:
        print("Error decoding JSON:", e)
else:
    print(f"Request failed with status code {response.status_code}")
