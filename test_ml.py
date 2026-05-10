import os

import numpy as np
import pandas as pd

from ml.data import apply_label, process_data
from ml.model import compute_model_metrics, inference, train_model


CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def load_sample_data(rows=200):
    """Load a small sample of census data for testing."""
    data_path = os.path.join("data", "census.csv")
    data = pd.read_csv(data_path)
    data = data.apply(
        lambda x: x.str.strip() if x.dtype == "object" else x
    )
    return data.head(rows)


def test_process_data_returns_expected_shapes():
    """Test that process_data returns aligned feature and label arrays."""
    data = load_sample_data()

    X, y, encoder, lb = process_data(
        data,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )

    assert X.shape[0] == data.shape[0]
    assert y.shape[0] == data.shape[0]
    assert encoder is not None
    assert lb is not None


def test_train_model_returns_predictor():
    """Test that train_model returns a fitted predictor."""
    data = load_sample_data()

    X, y, _, _ = process_data(
        data,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )

    model = train_model(X, y)

    assert hasattr(model, "predict")


def test_inference_returns_predictions():
    """Test that inference returns one prediction per input row."""
    data = load_sample_data()

    X, y, _, _ = process_data(
        data,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )

    model = train_model(X, y)
    preds = inference(model, X)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(y)


def test_compute_model_metrics_returns_valid_scores():
    """Test that model metrics return values between 0 and 1."""
    y = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1


def test_apply_label_returns_salary_text():
    """Test that binary predictions are converted to salary labels."""
    assert apply_label([1]) == ">50K"
    assert apply_label([0]) == "<=50K"
