# Census Income Inference API
[![Python Package using Conda](https://github.com/phidesigner/nd0821-c3-starter-code/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/phidesigner/nd0821-c3-starter-code/actions/workflows/python-package-conda.yml)

## Objective
The Census Income Inference API predicts whether a person earns <=50K or >50K based on the Census Income dataset. It preprocesses input data, runs inference using a pre-trained Random Forest model, and returns the predicted income class.

The project includes scripts for data preprocessing, model training, API integration, and testing.

---

## Directory Structure
```
├── data
│   ├── census.csv                # Raw dataset
│   ├── census.csv.dvc            # DVC tracking for raw dataset
│   ├── census_cleaned.csv        # Cleaned dataset
│   ├── census_cleaned.csv.dvc    # DVC tracking for cleaned dataset
│   └── data_eda.ipynb            # Exploratory data analysis notebook
├── environment.yml               # Conda environment file
├── starter
│   ├── README.md                 # API documentation
│   ├── __init__.py               # Init file for starter module
│   ├── main.py                   # API application entry point
│   ├── ml
│   │   ├── __init__.py           # Init file for ml module
│   │   ├── data.py               # Data preprocessing utilities
│   │   └── model.py              # Model training and inference utilities
│   ├── model
│   │   ├── encoder.pkl           # OneHotEncoder for categorical features
│   │   ├── encoder.pkl.dvc       # DVC tracking for encoder
│   │   ├── lb.pkl                # Label binarizer for target variable
│   │   ├── lb.pkl.dvc            # DVC tracking for label binarizer
│   │   ├── logs
│   │   │   ├── slice_output.txt  # Model performance on data slices
│   │   │   └── train_model.log   # Logs from training process
│   │   ├── model.pkl             # Trained model
│   │   ├── model.pkl.dvc         # DVC tracking for model
│   │   └── model_card.md         # Documentation for the trained model
│   ├── sanitycheck.py            # Sanity checks for the project
│   ├── screenshots               # Folder for API screenshots
│   └── train_model.py            # Model training script
└── tests
    ├── __init__.py               # Init file for tests module
    ├── test_main.py              # Unit tests for main.py
    └── test_model.py             # Unit tests for ml/model.py
```

---

## Features
- **Model Training:** Automates preprocessing, model training, evaluation, and artifact saving
- **Inference API:** Predicts income class for input data
- **Health Checks:** Validates API readiness and dependencies
- **Testing Framework:** Includes unit and integration tests
- **Github Actions CI:** Automates testing and validation

---

## Installation
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/phidesigner/nd0821-c3-starter-code.git
   cd project
   ```

2. **Set Up Environment:**
   ```bash
   conda env create -f environment.yml
   conda activate API
   ```

3. **Start the API:**
   ```bash
   python -m starter.main
   ```
   Access the API at [http://0.0.0.0:8000](http://0.0.0.0:8000).

---

## API Endpoints

### Root Endpoint
**GET /**
- **Description:** Returns a welcome message.
- **Response:**
  ```json
  {"message": "Welcome to the Census Income Inference API!"}
  ```

### Health Check
**GET /health**
- **Description:** Verifies API readiness.
- **Response:**
  ```json
  {"status": "healthy"}
  ```

### Inference
**POST /inference**
- **Description:** Predicts income class based on input data.
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
      "hours-per-week": 40,
      "education-num": 13,
      "fnlgt": 77516
  }
  ```
- **Response Example:**
  ```json
  {"prediction": ">50K"}
  ```

---

## Model Training
To train the model:
```bash
python starter/train_model.py
```
Artifacts (model.pkl, encoder.pkl, etc.) will be saved to `starter/model/`.

---

## Testing
### Unit and Integration Tests
Run all tests:
```bash
pytest tests/
```

---

## Logging
- Logs are stored in `starter/model/logs/`
- Key logs:
  - `train_model.log` – Training process logs
  - `slice_output.txt` – Slice metrics logs
  - `api_logs.log` – API logs 

---

## GitHub Actions Workflow

This project uses GitHub Actions to automate testing and validation. The workflow is defined in `python-package-conda.yml.yml`.

### Key Steps
1. **Checkout Code:**
   The workflow starts by checking out the code repository.

2. **Configure AWS Credentials:**
   Uses the `aws-actions/configure-aws-credentials` action to set up AWS access using secrets.

3. **Set up Miniconda:**
   Configures the Conda environment with Python 3.12.8 and installs required dependencies using the `environment.yml` file.

4. **Install Dependencies:**
   Installs all necessary dependencies specified in the `environment.yml` file.

5. **Set up DVC and Pull Data:**
   Configures DVC and pulls data tracked by DVC.

6. **Linting:**
   Runs `flake8` for linting to ensure code quality.

7. **Run Tests:**
   Executes tests using `pytest` to validate the application.

### Trigger
- The workflow can be triggered manually using `workflow_dispatch`.

### Workflow File Location
The workflow configuration is stored in `.github/workflows/python-package-conda.yml`.

---

## License
This project is licensed under the MIT License.
