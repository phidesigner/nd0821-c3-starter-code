"""
Script to train and evaluate a machine learning model.

This script handles data loading, preprocessing, model training, evaluation,
and saving of artifacts such as the trained model and preprocessing objects.
"""

import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import joblib
from starter.ml.data import process_data
from starter.ml.model import train_model, inference, compute_model_metrics, \
    compute_metrics_on_slices

# Paths for model and data storage
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "starter" / "model"
DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "census_cleaned.csv"
LOG_FILE = BASE_DIR / "starter" / "logs" / "train_model.log"

# Ensure necessary directories exist
MODEL_DIR.mkdir(exist_ok=True)
LOG_FILE.parent.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="w")
    ]
)
logger = logging.getLogger(__name__)


def train_and_evaluate(datafile=DATA_FILE) -> None:
    """
    Main function to train and evaluate the model.
    Handles data preprocessing, model training, evaluation, and artifact
    saving.
    """
    logger.info(f"Loading data from {datafile}")
    data = pd.read_csv(datafile)

    # Split the data
    logger.info("Splitting data into train and test sets...")
    train, test = train_test_split(data, test_size=0.20, random_state=42)

    # Define categorical features
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Process training data
    logger.info("Processing training data...")
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Process test data
    logger.info("Processing test data...")
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary",
        training=False, encoder=encoder, lb=lb
    )

    # Train the model
    logger.info("Training the model...")
    model = train_model(X_train, y_train)

    # Save artifacts
    logger.info(f"Saving model and preprocessing artifacts to {MODEL_DIR}...")
    joblib.dump(model, MODEL_DIR / "model.pkl")
    joblib.dump(encoder, MODEL_DIR / "encoder.pkl")
    joblib.dump(lb, MODEL_DIR / "lb.pkl")

    # Evaluate the model
    logger.info("Running inference and evaluating the model...")
    preds = inference(model, X_test)
    precision, recall, f1 = compute_model_metrics(y_test, preds)
    logger.info(
        f"Overall Metrics -> Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # Compute slice metrics
    logger.info("Computing metrics on slices of the data...")
    slice_output_file = MODEL_DIR / "slice_output.txt"
    with open(slice_output_file, "w") as slice_file:
        for feature in cat_features:
            logger.info(f"Evaluating performance for feature: {feature}")
            performance = compute_metrics_on_slices(
                test, feature, y_test, preds)
            slice_file.write(f"Performance on {
                             feature} slice:\n{performance}\n\n")
    logger.info(f"Slice metrics saved to {slice_output_file}")


if __name__ == "__main__":
    train_and_evaluate()
