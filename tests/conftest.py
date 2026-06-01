"""Pytest fixtures for API smoke tests."""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("APP_SECRET_KEY", "test-secret-key")
os.environ.setdefault("MODEL_PATH", str(PROJECT_ROOT / "models" / "churn_model.pkl"))
os.environ.setdefault("DATA_PATH", str(PROJECT_ROOT / "data" / "raw" / "data.csv"))
os.environ.setdefault("MONGODB_URI", "mongodb://mock")


def _ensure_model_artifacts() -> None:
    model_path = PROJECT_ROOT / "models" / "churn_model.pkl"
    preprocessor_path = PROJECT_ROOT / "models" / "preprocessor.pkl"
    if model_path.exists() and preprocessor_path.exists():
        return

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import joblib
    import json
    from datetime import datetime, timezone

    from src.data_loader import load_data
    from src.preprocess import preprocess

    raw_df = load_data(os.environ["DATA_PATH"])
    processed_df = preprocess(raw_df)
    x = processed_df.drop(columns=["Churn"])
    y = processed_df["Churn"]
    x_train, _, y_train, _ = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=25, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics_path = PROJECT_ROOT / "models" / "model_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "model_version": "v1.0",
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "xgboost": {"roc_auc": 0.9, "precision": 0.8, "recall": 0.8, "f1": 0.8},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


_ensure_model_artifacts()


@pytest.fixture
def mock_db(monkeypatch):
    async def save_prediction(*_args, **_kwargs):
        return "mock-id"

    async def get_predictions(skip=0, limit=20):
        return [
            {
                "customerID": "7590-VHVEG",
                "churn_probability": 0.12,
                "churn_prediction": False,
                "timestamp": "2026-01-01T00:00:00+00:00",
                "model_version": "v1.0",
            }
        ]

    async def save_metrics(metrics):
        return "metrics-id"

    async def get_latest_metrics():
        return {
            "model_version": "v1.0",
            "xgboost": {
                "roc_auc": 0.93,
                "precision": 0.84,
                "recall": 0.87,
                "f1": 0.85,
            },
            "trained_at": "2026-01-01T00:00:00+00:00",
        }

    async def ping_database():
        return True

    async def close_client():
        return None

    monkeypatch.setattr("src.db.save_prediction", save_prediction)
    monkeypatch.setattr("src.db.get_predictions", get_predictions)
    monkeypatch.setattr("src.db.save_metrics", save_metrics)
    monkeypatch.setattr("src.db.get_latest_metrics", get_latest_metrics)
    monkeypatch.setattr("src.db.ping_database", ping_database)
    monkeypatch.setattr("src.db.close_client", close_client)


@pytest.fixture
def client(mock_db):
    from src.api import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def api_headers():
    return {"X-API-Key": os.environ["APP_SECRET_KEY"]}


@pytest.fixture
def valid_customer():
    return {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "TechSupport": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }
