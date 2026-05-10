import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)


PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_PATH, "data", "census.csv")
MODEL_PATH = os.path.join(PROJECT_PATH, "model", "model.pkl")
ENCODER_PATH = os.path.join(PROJECT_PATH, "model", "encoder.pkl")
LB_PATH = os.path.join(PROJECT_PATH, "model", "lb.pkl")
SLICE_OUTPUT_PATH = os.path.join(PROJECT_PATH, "slice_output.txt")

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


def main():
    """Train model, save artifacts, and calculate slice performance."""
    data = pd.read_csv(DATA_PATH)

    data = data.apply(
        lambda x: x.str.strip() if x.dtype == "object" else x
    )

    train, test = train_test_split(
        data,
        test_size=0.20,
        random_state=42,
    )

    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train)

    save_model(model, MODEL_PATH)
    save_model(encoder, ENCODER_PATH)
    save_model(lb, LB_PATH)

    model = load_model(MODEL_PATH)
    preds = inference(model, X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {fbeta:.4f}")

    with open(SLICE_OUTPUT_PATH, "w") as file:
        file.write("Overall model performance\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"F1: {fbeta:.4f}\n\n")

        for column in CAT_FEATURES:
            for slice_value in sorted(test[column].unique()):
                slice_precision, slice_recall, slice_fbeta = (
                    performance_on_categorical_slice(
                        test,
                        column,
                        slice_value,
                        CAT_FEATURES,
                        "salary",
                        encoder,
                        lb,
                        model,
                    )
                )

                file.write(f"{column}: {slice_value}\n")
                file.write(f"Precision: {slice_precision:.4f}\n")
                file.write(f"Recall: {slice_recall:.4f}\n")
                file.write(f"F1: {slice_fbeta:.4f}\n\n")


if __name__ == "__main__":
    main()
