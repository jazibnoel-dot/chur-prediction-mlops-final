"""Clean, encode, scale, and balance telecom churn data for model training."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

YES_NO_MAP = {"Yes": 1, "No": 0}
GENDER_MAP = {"Female": 0, "Male": 1}
SERVICE_REPLACEMENTS = {
    "No phone service": "No",
    "No internet service": "No",
}
ONE_HOT_COLUMNS = ["InternetService", "Contract", "PaymentMethod"]
SCALED_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
PREPROCESSOR_PATH = Path("models/preprocessor.pkl")


def _string_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "string", "str"]).columns.tolist()


def _yes_no_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in _string_columns(df)
        if col not in {"Churn", "gender", *ONE_HOT_COLUMNS}
        and set(df[col].dropna().unique()).issubset({"Yes", "No"})
    ]


def preprocess(df: pd.DataFrame, preprocessor_path: str | Path = PREPROCESSOR_PATH) -> pd.DataFrame:
    """Return a cleaned, encoded, scaled, and SMOTE-balanced DataFrame ready for training."""
    df = df.copy()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    for col in _string_columns(df):
        df[col] = df[col].astype(str).str.strip()

    df = df.replace(SERVICE_REPLACEMENTS)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    total_charges_median = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(total_charges_median)

    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["is_long_term_customer"] = (df["tenure"] >= 24).astype(int)
    df["high_monthly_charges"] = (df["MonthlyCharges"] > 70).astype(int)
    df["has_support_services"] = (
        (df["OnlineSecurity"] == "Yes") | (df["TechSupport"] == "Yes")
    ).astype(int)
    df["charge_tenure_ratio"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    yes_no_cols = _yes_no_columns(df)
    for col in yes_no_cols:
        df[col] = df[col].map(YES_NO_MAP)

    if "gender" in df.columns:
        df["gender"] = df["gender"].map(GENDER_MAP)

    one_hot_categories = {
        col: sorted(df[col].dropna().unique().tolist()) for col in ONE_HOT_COLUMNS
    }
    df = pd.get_dummies(df, columns=ONE_HOT_COLUMNS, dtype=int)

    churn = df["Churn"].map(YES_NO_MAP)
    features = df.drop(columns=["Churn"])

    scaler = StandardScaler()
    features[SCALED_COLUMNS] = scaler.fit_transform(features[SCALED_COLUMNS])

    feature_columns = features.columns.tolist()
    x_resampled, y_resampled = SMOTE(random_state=42).fit_resample(features, churn)

    processed = pd.DataFrame(x_resampled, columns=feature_columns)
    processed["Churn"] = y_resampled

    preprocessor_path = Path(preprocessor_path)
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "scaler": scaler,
            "scaled_columns": SCALED_COLUMNS,
            "yes_no_columns": yes_no_cols,
            "one_hot_columns": ONE_HOT_COLUMNS,
            "one_hot_categories": one_hot_categories,
            "feature_columns": feature_columns,
            "total_charges_median": total_charges_median,
            "service_replacements": SERVICE_REPLACEMENTS,
            "yes_no_map": YES_NO_MAP,
            "gender_map": GENDER_MAP,
        },
        preprocessor_path,
    )
    print(f"Saved preprocessor to {preprocessor_path}")
    print(f"Processed shape: {processed.shape}")
    print(f"Churn distribution after SMOTE:\n{processed['Churn'].value_counts()}")

    return processed


def load_preprocessor(preprocessor_path: str | Path = PREPROCESSOR_PATH) -> dict:
    """Load a fitted preprocessor artifact from disk."""
    path = Path(preprocessor_path)
    if not path.exists():
        raise FileNotFoundError(f"Preprocessor not found: {path}")
    return joblib.load(path)


def transform_features(
    df: pd.DataFrame,
    artifact: dict | None = None,
    preprocessor_path: str | Path = PREPROCESSOR_PATH,
) -> pd.DataFrame:
    """Apply saved preprocessing for inference (no SMOTE or target column)."""
    if artifact is None:
        artifact = load_preprocessor(preprocessor_path)

    df = df.copy()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    for col in _string_columns(df):
        df[col] = df[col].astype(str).str.strip()

    df = df.replace(artifact.get("service_replacements", SERVICE_REPLACEMENTS))

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(artifact["total_charges_median"])

    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["is_long_term_customer"] = (df["tenure"] >= 24).astype(int)
    df["high_monthly_charges"] = (df["MonthlyCharges"] > 70).astype(int)
    df["has_support_services"] = (
        (df["OnlineSecurity"] == "Yes") | (df["TechSupport"] == "Yes")
    ).astype(int)
    df["charge_tenure_ratio"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    yes_no_map = artifact.get("yes_no_map", YES_NO_MAP)
    for col in artifact["yes_no_columns"]:
        if col in df.columns:
            df[col] = df[col].map(yes_no_map)

    if "gender" in df.columns:
        df["gender"] = df["gender"].map(artifact.get("gender_map", GENDER_MAP))

    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    for col in artifact["one_hot_columns"]:
        if col not in df.columns:
            categories = artifact["one_hot_categories"][col]
            for category in categories:
                df[f"{col}_{category}"] = 0
            continue
        categories = artifact["one_hot_categories"][col]
        for category in categories:
            dummy_name = f"{col}_{category}"
            df[dummy_name] = (df[col] == category).astype(int)
        df = df.drop(columns=[col])

    scaler = artifact["scaler"]
    scaled_columns = artifact["scaled_columns"]
    df[scaled_columns] = scaler.transform(df[scaled_columns])

    feature_columns = artifact["feature_columns"]
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df[feature_columns]


if __name__ == "__main__":
    import os
    import sys

    from dotenv import load_dotenv

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data_loader import load_data

    load_dotenv()
    data_path = os.getenv("DATA_PATH", "data/raw/data.csv")
    raw_df = load_data(data_path)
    preprocess(raw_df)
