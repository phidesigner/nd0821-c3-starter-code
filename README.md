# Census Income Inference API

## Objective
The Census Income Inference API is designed to predict whether a person earns `<=50K` or `>50K` based on the Census Income dataset. The API preprocesses input data, runs inference using a pre-trained Random Forest model, and returns the predicted income class.

This project also includes scripts for training the model, preprocessing the data, and testing the API.

---

## Directory Structure
```
├── data
│   ├── census.csv
│   ├── census.csv.dvc
│   ├── census_cleaned.csv
│   ├── census_cleaned.csv.dvc
│   └── data_eda.ipynb
├── environment.yml
├── scripts
│   ├── API_request.py
│   └── __init__.py
├── starter
│   ├── README.md
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-312.pyc
│   │   └── hello.cpython-312.pyc
│   ├── hello.py
│   ├── main.py
│   ├── ml
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-312.pyc
│   │   │   ├── data.cpython-312.pyc
│   │   │   └── model.cpython-312.pyc
│   │   ├── data.py
│   │   └── model.py
│   ├── model
│   │   ├── encoder.pkl
│   │   ├── encoder.pkl.dvc
│   │   ├── lb.pkl
│   │   ├── lb.pkl.dvc
│   │   ├── logs
│   │   │   ├── slice_output.txt
│   │   │   └── train_model.log
│   │   ├── model.pkl
│   │   ├── model.pkl.dvc
│   │   └── model_card.md
│   ├── sanitycheck.py
│   ├── screenshots
│   └── train_model.py
└── tests
    ├── __init__.py
    ├── __pycache__
    │   ├── __init__.cpython-312.pyc
    │   ├── test_hello.cpython-312-pytest-7.4.4.pyc
    │   └── test_model.cpython-312-pytest-7.4.4.pyc
    ├── ml
    ├── test_hello.py
    └── test_model.py
```

---

## Features
- **Train a Model:**
  - `train_model.py` preprocesses the Census Income dataset, trains a Random Forest classifier, evaluates it, and saves artifacts (model, encoder, and label binarizer).

- **Perform Inference:**
  - Use the `/inference` endpoint to predict income classes based on user input.

- **Health Checks:**
  - The `/health` endpoint ensures that the API and its dependencies are loaded correctly.

- **Tests:**
  - Integration tests validate the API endpoints.
  - Unit tests validate core machine learning functions and metrics.

---

## Installation
1. **Clone the Repository:**
   ```bash
   git clone <repository_url>
   cd project
   ```

2. **Set Up Virtual Environment:**
   ```bash
   conda env create -f environment.yml
   conda activate <env_name>
   ```

3. **Verify Installation:**
   ```bash
   python starter/main.py
   ```
   Access the API at `http://127.0.0.1:8000`.

---

## API Endpoints

### Root Endpoint
**`GET /`**
- **Description:** Returns a welcome message.
- **Response:**
  ```json
  {"message": "Welcome to the Census Income Inference API!"}
  ```

### Health Check
**`GET /health`**
- **Description:** Verifies the readiness of the API and its dependencies.
- **Response:**
  ```json
  {"status": "healthy"}
  ```

### Inference
**`POST /inference`**
- **Description:** Predicts income class based on Census data.
- **Request Body Example:**
  ```json
  {
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
  ```
- **Response Example:**
  ```json
  {"prediction": ">50K"}
  ```

---

## Training the Model
Run the following command to train and evaluate the model:
```bash
python starter/train_model.py
```
Artifacts (e.g., `model.pkl`, `encoder.pkl`) will be saved to the `starter/model/` directory.

---

## Testing

### Integration Tests
- Located in `tests/test_model.py`.
- Run using:
  ```bash
  pytest tests/test_model.py
  ```

### Manual Testing
- Use `scripts/API_request.py` to manually test the API.
- Example usage:
  ```bash
  python scripts/API_request.py
  ```

---

## Logging
- Logs are stored in `starter/model/logs/train_model.log`.
- Logs include training metrics, slice metrics, and API request information.

---

## License
This project is licensed under the MIT License.
