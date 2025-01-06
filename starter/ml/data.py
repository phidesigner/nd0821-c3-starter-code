import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from typing import Tuple, Optional, List


def process_data(
    X: pd.DataFrame,
    categorical_features: Optional[List[str]] = None,
    label: Optional[str] = None,
    training: bool = True,
    encoder: Optional[OneHotEncoder] = None,
    lb: Optional[LabelBinarizer] = None,
) -> Tuple[np.ndarray, np.ndarray, OneHotEncoder, LabelBinarizer]:
    """
    Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing the features and label. Columns in categorical_features
    categorical_features : Optional[List[str]]
        List containing the names of the categorical features (default=None)
    label : Optional[str]
        Name of the label column in X. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : Optional[OneHotEncoder]
        Pre-trained sklearn OneHotEncoder, only used if training=False.
    lb : Optional[LabelBinarizer]
        Pre-trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.ndarray
        Processed data.
    y : np.ndarray
        Processed labels if labeled=True, otherwise empty np.ndarray.
    encoder : OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """
    categorical_features = categorical_features or []

    # Separate labels and features
    if label is not None:
        y = X[label].values
        X = X.drop(columns=[label])
    else:
        y = np.array([])

    # Split categorical and continuous features
    X_categorical = X[categorical_features]
    X_continuous = X.drop(columns=categorical_features)

    if training:
        # Initialize and fit encoders during training
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y).ravel()
    else:
        # Transform data using pre-trained encoders
        if encoder is None or lb is None:
            raise ValueError(
                "Encoder and LabelBinarizer must be provided in inference mode.")
        X_categorical = encoder.transform(X_categorical)
        if y.size > 0:  # Transform y only if it exists
            y = lb.transform(y).ravel()

    # Concatenate continuous and encoded categorical features
    X = np.concatenate([X_continuous.values, X_categorical], axis=1)
    return X, y, encoder, lb
