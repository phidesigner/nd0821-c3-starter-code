from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> BaseEstimator:
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : BaseEstimator
        Trained machine learning model.
    """
    # Using a Random Forest Classifier
    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y: np.ndarray, preds: np.ndarray) -> tuple[float,
                                                                     float,
                                                                     float]:
    """
    Validates the trained machine learning model using precision, recall, 
    and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
        Precision of the model predictions.
    recall : float
        Recall of the model predictions.
    fbeta : float
        F1 score of the model predictions.
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model: BaseEstimator, X: np.ndarray) -> np.ndarray:
    """
    Run model inferences and return the predictions.

    Inputs
    ------
    model : BaseEstimator
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def compute_metrics_on_slices(
    test_df: pd.DataFrame,
    feature: str,
    y: np.ndarray,
    preds: np.ndarray,
    output_file: str = None
) -> pd.DataFrame:
    """
    Computes the model metrics for slices of data based on a specific feature,
    and optionally logs the results to a file.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test data as a DataFrame.
    feature : str
        Feature selected to compute slices.
    y : np.ndarray
        True labels.
    preds : np.ndarray
        Predicted labels.
    output_file : str, optional
        File name to save the metrics. If None, no file is saved.

    Returns
    -------
    pd.DataFrame
        DataFrame with model metrics for each slice of the selected feature.
    """
    slice_options = test_df[feature].unique()
    results = []

    for option in slice_options:
        # Filter rows corresponding to the current slice
        slice_mask = test_df[feature] == option
        slice_y = y[slice_mask]
        slice_preds = preds[slice_mask]

        # Handle edge case for empty slices
        if len(slice_y) > 0:
            precision, recall, fbeta = compute_model_metrics(
                slice_y, slice_preds)
        else:
            precision, recall, fbeta = None, None, None

        # Store results
        results.append({
            "feature": feature,
            "value": option,
            "n_samples": len(slice_y),
            "precision": precision,
            "recall": recall,
            "fbeta": fbeta
        })

    # Convert results to a DataFrame
    perf_df = pd.DataFrame(results)

    # Write to file if specified
    if output_file:
        with open(output_file, "w") as file:
            for _, row in perf_df.iterrows():
                file.write(
                    f"Feature: {row['feature']}, Value: {row['value']} -> "
                    f"Samples: {row['n_samples']}, "
                    f"Precision: {
                        row['precision']:.4f if row['precision'] is not\
                             None else 'NA'}, "
                    f"Recall: {
                        row['recall']:.4f if row['recall'] is not\
                             None else 'NA'}, "
                    f"F1: {row['fbeta']:.4f if row['fbeta'] is not\
                         None else 'NA'}\n"
                )
        print(f"Slice metrics saved to {output_file}.")

    return perf_df
