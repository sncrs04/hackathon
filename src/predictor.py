import os
from typing import Any, Dict, List, Tuple, Union

import joblib
import pandas as pd

from src.data_loader import clean_data


_ARTIFACT: Any = None


def load_model(model_path: str = "models/service_model.joblib") -> Dict[str, Any]:
    global _ARTIFACT
    if _ARTIFACT is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        _ARTIFACT = joblib.load(model_path)
    return _ARTIFACT


def predict_service_outcome(
    input_features: Union[Dict[str, Any], pd.DataFrame],
    model_path: str = "models/service_model.joblib",
) -> Tuple[Any, float]:
    artifact = load_model(model_path)
    model = artifact["model"]
    feature_columns: List[str] = artifact["feature_columns"]

    if isinstance(input_features, dict):
        input_df = pd.DataFrame([input_features])
    else:
        input_df = input_features.copy()

    # Apply the same preprocessing pipeline used during training.
    input_df = clean_data(input_df)

    # Drop the target column if it was accidentally included in the input.
    input_df = input_df.drop(columns=["service_outcome"], errors="ignore")

    # One-hot encode categoricals to match training feature space.
    numeric = input_df.select_dtypes(include="number")
    categorical = input_df.select_dtypes(include=["object", "category"])
    if not categorical.empty:
        categorical = pd.get_dummies(categorical, dummy_na=True, drop_first=True)
    input_processed = pd.concat([numeric, categorical], axis=1).fillna(0)

    # Align columns exactly — add all missing training columns in one concat
    # (avoids DataFrame fragmentation from repeated single-column inserts).
    missing = {col: 0 for col in feature_columns if col not in input_processed.columns}
    if missing:
        input_processed = pd.concat(
            [input_processed, pd.DataFrame(missing, index=input_processed.index)],
            axis=1,
        )
    input_processed = input_processed[feature_columns]

    label_encoder = artifact["label_encoder"]

    predictions_encoded = model.predict(input_processed)
    probabilities = model.predict_proba(input_processed)
    # Confidence for the predicted class (works for binary and multiclass).
    predicted_index = int(predictions_encoded[0])
    confidence = round(float(probabilities[0][predicted_index]) * 100, 2)

    # Decode integer label back to the original class name string.
    prediction = label_encoder.inverse_transform(predictions_encoded)[0]
    return prediction, confidence
