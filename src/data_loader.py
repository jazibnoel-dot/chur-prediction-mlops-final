"""Load and validate telecom churn CSV data."""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def load_data(path: str | Path) -> pd.DataFrame:
    """Read a churn CSV, validate schema, print a summary, and return the DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Expected columns: {', '.join(REQUIRED_COLUMNS)}"
        )

    print(f"Loaded {path.name}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print("\nNull counts per column:")
    for col in df.columns:
        null_count = df[col].isna().sum()
        if null_count > 0:
            print(f"  {col}: {null_count}")
    if df.isna().sum().sum() == 0:
        print("  (none)")
    print("\nData types:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")

    return df


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()
    data_path = os.getenv("DATA_PATH", "data/raw/data.csv")
    load_data(data_path)
