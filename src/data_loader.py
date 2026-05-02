import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare raw government service records for training.

    The feature engineering below turns timestamps into actionable signals,
    such as the hour of day and day of week, which can reveal bottlenecks
    in service operations.
    """
    df = df.copy()

    # Convert timestamps so time-based features are reliable.
    df["created_at"] = pd.to_datetime(df.get("created_at"), errors="coerce")
    df["finish_time"] = pd.to_datetime(df.get("finish_time"), errors="coerce")

    # If a request has no finish time, keep it as NaT for now and fill later.
    df["response"] = df.get("response").fillna("not_recorded")

    # Measure processing time in hours, which is directly meaningful for delay analysis.
    df["processing_time"] = (df["finish_time"] - df["created_at"]).dt.total_seconds() / 3600

    # Use hour/day/month to detect operational patterns and service bottlenecks.
    df["hour"] = df["created_at"].dt.hour
    df["day_of_week"] = df["created_at"].dt.dayofweek
    df["month"] = df["created_at"].dt.month

    # Estimate citizen age from birth date to allow demographic-informed predictions.
    birth_date = pd.to_datetime(df.get("birth_date"), errors="coerce")
    years = (pd.Timestamp.now() - birth_date).dt.days / 365.25
    df["citizen_age"] = years.round(1)

    # If finish_time is missing, infer a reasonable processing time from the median.
    median_duration = df["processing_time"].median(skipna=True)
    df.loc[df["finish_time"].isna() & df["created_at"].notna(), "processing_time"] = median_duration

    # Keep a clean numeric output for later modeling; missing age or time features can be imputed.
    df["processing_time"] = df["processing_time"].fillna(median_duration)
    df["citizen_age"] = df["citizen_age"].fillna(df["citizen_age"].median(skipna=True))
    df["hour"] = df["hour"].fillna(-1).astype(int)
    df["day_of_week"] = df["day_of_week"].fillna(-1).astype(int)
    df["month"] = df["month"].fillna(-1).astype(int)

    return df
