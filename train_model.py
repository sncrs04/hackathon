import os
from pathlib import Path

import pandas as pd

from src.data_loader import clean_data
from src.train import ModelTrainer


def main() -> None:
    raw_data_path = Path("scramble-data.csv")
    try:
        raw_df = pd.read_csv(raw_data_path, encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Raw data file not found at {raw_data_path}. "
            "Make sure scramble-data.csv is in the project root and rerun."
        )
    except Exception as exc:
        raise IOError(f"Unable to load raw CSV: {exc}") from exc

    cleaned_df = clean_data(raw_df)

    trainer = ModelTrainer()
    os.makedirs("models", exist_ok=True)

    result = trainer.train(cleaned_df)
    print(f"Model saved to: {result['model_path']}")
    print(f"Feature importance plot saved to: {result['importance_path']}")


if __name__ == "__main__":
    main()
