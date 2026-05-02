import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


class ModelTrainer:
    def __init__(self, model_path: str = "models/service_model.joblib", importance_path: str = "models/feature_importance.png"):
        self.model_path = model_path
        self.importance_path = importance_path
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_columns: List[str] = []

    def _prepare_features(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' is missing from data")

        y = df[target_column]
        X = df.drop(columns=[target_column])

        # Use numeric inputs and one-hot encode categorical fields for XGBoost.
        numeric = X.select_dtypes(include="number")
        categorical = X.select_dtypes(include=["object", "category"])

        if not categorical.empty:
            categorical = pd.get_dummies(categorical, dummy_na=True, drop_first=True)

        X_processed = pd.concat([numeric, categorical], axis=1)
        X_processed = X_processed.fillna(0)

        self.feature_columns = list(X_processed.columns)
        return X_processed, y

    def _save_model(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)
        except Exception as exc:
            raise IOError(f"Failed to save model to {self.model_path}: {exc}") from exc

    def _plot_feature_importance(self) -> None:
        if self.model is None:
            return

        importance = self.model.get_booster().get_score(importance_type="weight")
        importance_df = pd.DataFrame(
            [(feature, score) for feature, score in importance.items()],
            columns=["feature", "importance"],
        ).sort_values(by="importance", ascending=False)

        os.makedirs(os.path.dirname(self.importance_path), exist_ok=True)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df.head(20), x="importance", y="feature", palette="viridis")
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig(self.importance_path, dpi=150)
        plt.close()

    def train(self, df: pd.DataFrame, target_column: str = "service_outcome") -> Dict[str, Any]:
        X, y = self._prepare_features(df, target_column)

        if y.nunique() < 2:
            raise ValueError("Target column must contain at least two classes for classification.")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42,
        )

        class_counts = y_train.value_counts()
        scale_pos_weight = 1.0

        if len(class_counts) == 2:
            minority = class_counts.min()
            majority = class_counts.max()
            if minority > 0:
                scale_pos_weight = float(majority) / float(minority)

        self.model = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            scale_pos_weight=scale_pos_weight,
        )

        self.model.fit(X_train, y_train)
        self._save_model()
        self._plot_feature_importance()

        predictions = self.model.predict(X_test)
        report = classification_report(y_test, predictions)
        print("Classification report:\n", report)

        return {
            "model_path": self.model_path,
            "importance_path": self.importance_path,
            "report": report,
        }
