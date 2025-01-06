# Script to train machine learning model.
# Add the necessary imports for the starter code.

import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import train_model, inference, compute_model_metrics, compute_metrics_on_slices
from sklearn.model_selection import train_test_split
import joblib

# Add code to load in the data.
data = pd.read_csv("data/census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

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
# Process the training data
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Train the model
print("Training the model...")
model = train_model(X_train, y_train)

# Save the trained model, encoder, and label binarizer
joblib.dump(model, "model/model.pkl")
joblib.dump(encoder, "model/encoder.pkl")
joblib.dump(lb, "model/lb.pkl")
print("Model and preprocessing objects saved.")

# Perform inference on the test set
print("Running inference...")
preds = inference(model, X_test)

# Compute and display model metrics
print("Computing metrics...")
precision, recall, f1 = compute_model_metrics(y_test, preds)
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# Compute metrics on slices of the data
print("Computing metrics on slices of data...")
slice_output_file = "slice_output.txt"
compute_metrics_on_slices(test, "education", y_test,
                          preds, output_file=slice_output_file)
print(f"Slice metrics saved to {slice_output_file}.")
