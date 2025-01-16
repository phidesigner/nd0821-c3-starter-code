"""
Script to test the Census Income Inference API
Author: Ivan Diaz
"""

import requests
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Define API URL
base_url = "https://nd0821-c3-starter-code-production.up.railway.app"
inference_url = f"{base_url}/inference"

# Define sample input
sample_input = {
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

# Make a POST request to the /inference endpoint
try:
    logger.info(f"Sending request to {inference_url}")
    response = requests.post(
        inference_url,
        data=json.dumps(sample_input),
        headers={"Content-Type": "application/json"}
    )

    # Log and print response
    logger.info(f"Status code: {response.status_code}")
    logger.info(f"Response: {response.json()}")

    if response.status_code == 200:
        print("Prediction:", response.json())
    else:
        print("Error:", response.json())
except Exception as e:
    logger.error("Error during API request", exc_info=True)
