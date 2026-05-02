import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare raw government service records for training or inference.

    Only uses information available at the time a request is submitted,
    preventing any data leakage from post-submission fields.
    """
    df = df.copy()

    # Derive target from status before dropping the source column.
    if "status" in df.columns and "service_outcome" not in df.columns:
        df["service_outcome"] = df["status"]

    # Drop leaky columns (response/finish_time only exist after the outcome is
    # known) and PII/ID columns that add noise without predictive signal.
    drop_cols = [
        "service_request_id", "user_id",
        "first_name", "middle_name", "last_name", "forth_name", "phone_num",
        "status", "response", "finish_time", "nodes",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Convert created_at so we can extract temporal features.
    df["created_at"] = pd.to_datetime(df.get("created_at"), errors="coerce")

    # Hour/day/month reveal patterns like end-of-week backlogs or peak hours.
    df["hour"] = df["created_at"].dt.hour
    df["day_of_week"] = df["created_at"].dt.dayofweek
    df["month"] = df["created_at"].dt.month

    # Citizen age from birth_date enables demographic-informed predictions.
    birth_date = pd.to_datetime(df.get("birth_date"), errors="coerce")
    years = (pd.Timestamp.now() - birth_date).dt.days / 365.25
    df["citizen_age"] = years.round(1)

    # Raw date columns are no longer needed after feature extraction.
    df = df.drop(columns=[c for c in ["created_at", "birth_date"] if c in df.columns])

    # Impute missing values so the model never receives NaN inputs.
    df["hour"] = df["hour"].fillna(-1).astype(int)
    df["day_of_week"] = df["day_of_week"].fillna(-1).astype(int)
    df["month"] = df["month"].fillna(-1).astype(int)
    df["citizen_age"] = df["citizen_age"].fillna(df["citizen_age"].median(skipna=True))

    return df
