import pandas as pd
import numpy as np
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics

# Sample DataFrame for testing
data = {
    "age": [25, 30, 35, 40],
    "workclass": ["Private", "Self-emp", "State-gov", "Private"],
    "education": ["Bachelors", "HS-grad", "Masters", "Assoc-acdm"],
    "marital-status": ["Never-married", "Married", "Divorced", "Separated"],
    "occupation": ["Tech-support", "Craft-repair", "Adm-clerical",
                   "Other-service"],
    "relationship": ["Not-in-family", "Husband", "Wife", "Unmarried"],
    "race": ["White", "Black", "Asian", "White"],
    "sex": ["Male", "Female", "Female", "Male"],
    "native-country": ["United-States", "India", "Canada", "United-States"],
    "salary": [">50K", "<=50K", ">50K", "<=50K"]
}
data_df = pd.DataFrame(data)

# Define categorical features
cat_features = [
    "workclass", "education", "marital-status", "occupation", "relationship",
    "race", "sex", "native-country"
]


def test_process_data():
    """Test the process_data function."""
    X, y, encoder, lb = process_data(
        data_df,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    assert X.shape[0] == data_df.shape[0], "Processed data should have the\
         same number of rows as the input."
    assert len(
        y) == data_df.shape[0], "Labels should have the same number of rows\
             as the input."
    assert encoder is not None, "Encoder should be initialized."
    assert lb is not None, "Label binarizer should be initialized."

    # Check the correctness of the transformed feature values
    assert np.array_equal(y, lb.transform(data_df['salary']).flatten(
    )), "Transformed labels do not match expected values."


def test_train_model():
    """Test the train_model function."""
    X, y, _, _ = process_data(
        data_df,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    model = train_model(X, y)
    assert hasattr(model, "predict"), "Model should have a predict method."
    assert hasattr(model, "fit"), "Model should have a fit method."


def test_compute_model_metrics():
    """Test the compute_model_metrics function."""
    y_true = np.array([1, 0, 1, 0])  # True labels
    y_pred = np.array([1, 0, 0, 0])  # Predicted labels

    precision, recall, f1 = compute_model_metrics(y_true, y_pred)

    assert 0 <= precision <= 1, "Precision should be between 0 and 1."
    assert 0 <= recall <= 1, "Recall should be between 0 and 1."
    assert 0 <= f1 <= 1, "F1 score should be between 0 and 1."

    # Test edge cases with extreme precision values
    y_true_edge = np.array([1, 1, 0, 0])
    y_pred_edge = np.array([1, 1, 0, 0])
    precision_edge, recall_edge, f1_edge = compute_model_metrics(
        y_true_edge, y_pred_edge)

    assert precision_edge == 1.0, "Precision should be 1.0 for perfect\
    predictions."
    assert recall_edge == 1.0, "Recall should be 1.0 for perfect predictions."
    assert f1_edge == 1.0, "F1 score should be 1.0 for perfect predictions."
