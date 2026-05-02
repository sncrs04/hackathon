import os
from typing import Any, Dict, List, Tuple, Union

import joblib
import pandas as pd


_MODEL: Any = None


def load_model(model_path: str = "models/service_model.joblib") -> Any:
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        _MODEL = joblib.load(model_path)
    return _MODEL


def predict_service_outcome(input_features: Union[Dict[str, Any], pd.DataFrame], model_path: str = "models/service_model.joblib") -> Tuple[Any, float]:
    model = load_model(model_path)

    if isinstance(input_features, dict):
        input_df = pd.DataFrame([input_features])
    else:
        input_df = input_features.copy()

    if hasattr(model, "predict_proba"):
        predictions = model.predict(input_df)
        probabilities = model.predict_proba(input_df)
        positive_prob = float(probabilities[:, 1][0]) * 100
    else:
        predictions = model.predict(input_df)
        positive_prob = 0.0

    return predictions[0], round(positive_prob, 2)
