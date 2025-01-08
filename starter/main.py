from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import joblib
from pathlib import Path
import logging
import os
import pandas as pd
from starter.ml.data import process_data
import uvicorn

# Paths for model and data storage
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "starter" / "model"

# Ensure the logs directory exists
LOG_DIR = MODEL_DIR / "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Initialize the FastAPI app
app = FastAPI(
    title="Census Income Inference API",
    description="An API for predicting income level based on Census data",
    version="1.0.0",
)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_DIR / "api_logs.log",
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Globals for model and artifacts
model = None
encoder = None
lb = None


class InferenceRequest(BaseModel):
    age: int
    workclass: str = Field(..., alias="workclass")
    education: str = Field(..., alias="education")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str = Field(..., alias="occupation")
    relationship: str = Field(..., alias="relationship")
    race: str = Field(..., alias="race")
    sex: str = Field(..., alias="sex")
    native_country: str = Field(..., alias="native-country")
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    education_num: int = Field(..., alias="education-num")
    fnlgt: int = Field(..., alias="fnlgt")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )


async def load_artifacts():
    """Load model and artifacts."""
    global model, encoder, lb
try:
    model = joblib.load(MODEL_DIR / "model.pkl")
    logger.info("Model loaded successfully.")
    encoder = joblib.load(MODEL_DIR / "encoder.pkl")
    logger.info("Encoder loaded successfully.")
    lb = joblib.load(MODEL_DIR / "lb.pkl")
    logger.info("LabelBinarizer loaded successfully.")
except Exception as e:
    logger.error("Error loading artifacts: %s", e)
    raise RuntimeError("Failed to load model artifacts")


@app.on_event("startup")
async def startup_event():
    """Run startup tasks."""
    await load_artifacts()


@app.get("/")
def read_root() -> dict:
    """Root endpoint with a welcome message."""
    return {"message": "Welcome to the Census Income Inference API!"}


@app.get("/health", summary="Health Check", description="Check the health of the API and its dependencies")
def health_check() -> dict:
    """Health check endpoint."""
    try:
        if model is None or encoder is None or lb is None:
            raise ValueError("Model artifacts not loaded")
        return {"status": "healthy"}
    except Exception as e:
        logger.error("Health check failed: %s", str(e))
        return {"status": "unhealthy", "error": str(e)}


@app.post(
    "/inference",
    summary="Income Prediction",
    description="Provide Census data to predict income level (<=50K or >50K)",
)
def predict_income(request: InferenceRequest) -> dict:
    """Perform inference on the input data."""
    try:
        # Convert request to DataFrame-compatible format
        input_data = {
            "age": [request.age],
            "workclass": [request.workclass],
            "education": [request.education],
            "marital-status": [request.marital_status],
            "occupation": [request.occupation],
            "relationship": [request.relationship],
            "race": [request.race],
            "sex": [request.sex],
            "native-country": [request.native_country],
            "education-num": [request.education_num],
            "fnlgt": [request.fnlgt],
            "capital-gain": [request.capital_gain],
            "capital-loss": [request.capital_loss],
            "hours-per-week": [request.hours_per_week],
        }
        logger.info("Formatted input data: %s", input_data)

        # Convert to pandas DataFrame
        input_df = pd.DataFrame(input_data)
        logger.info("Input data converted to DataFrame: %s", input_df)

        # Preprocess the input
        categorical_features = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country"
        ]

        X, _, _, _ = process_data(
            input_df,
            categorical_features=categorical_features,
            label=None,
            training=False,
            encoder=encoder,
            lb=lb,
        )
        logger.info("Processed input: %s", X)

        # Perform inference
        prediction = model.predict(X)
        predicted_class = lb.inverse_transform(prediction)[0]
        logger.info("Predicted class: %s", predicted_class)

        return {"prediction": predicted_class}
    except Exception as e:
        logger.error("Error during prediction: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
