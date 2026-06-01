"""Train churn prediction models and save evaluation metrics."""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import HfApi
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_data
from src.preprocess import preprocess

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/churn_model.pkl"))
PREPROCESSOR_PATH = Path("models/preprocessor.pkl")
METRICS_PATH = Path("models/model_metrics.json")

XGB_PARAM_DIST = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
}


def _evaluate_model(name: str, model, x_test, y_test) -> dict:
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    print(f"\n{'=' * 60}")
    print(f"{name} Evaluation")
    print(f"{'=' * 60}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
    }


def _upload_to_huggingface(token: str, repo_id: str) -> None:
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")

    for file_path in (MODEL_PATH, PREPROCESSOR_PATH):
        if not file_path.exists():
            raise FileNotFoundError(f"Cannot upload missing file: {file_path}")
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=file_path.name,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"Uploaded {file_path} to {repo_id}")


def main() -> None:
    load_dotenv()

    data_path = os.getenv("DATA_PATH", "data/raw/data.csv")
    model_path = Path(os.getenv("MODEL_PATH", MODEL_PATH))
    metrics_path = METRICS_PATH

    raw_df = load_data(data_path)
    processed_df = preprocess(raw_df)

    x = processed_df.drop(columns=["Churn"])
    y = processed_df["Churn"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"\nTraining set: {len(x_train)} samples")
    print(f"Test set: {len(x_test)} samples")

    rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_model.fit(x_train, y_train)
    rf_metrics = _evaluate_model("Random Forest Baseline", rf_model, x_test, y_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    xgb_base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=XGB_PARAM_DIST,
        n_iter=20,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    print("\nTuning XGBoost hyperparameters...")
    search.fit(x_train, y_train)
    best_model = search.best_estimator_

    print(f"\nBest XGBoost params: {search.best_params_}")
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")

    xgb_metrics = _evaluate_model("XGBoost (Primary)", best_model, x_test, y_test)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"\nSaved best model to {model_path}")

    metrics_payload = {
        "model_version": "v1.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "best_model": "XGBClassifier",
        "best_params": search.best_params_,
        "xgboost": xgb_metrics,
        "random_forest_baseline": rf_metrics,
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics_payload, metrics_file, indent=2)
    print(f"Saved metrics to {metrics_path}")

    hf_token = os.getenv("HF_TOKEN", "")
    hf_repo_id = os.getenv("HF_REPO_ID", "")
    if hf_token and hf_token != "your_huggingface_token_here" and hf_repo_id:
        try:
            _upload_to_huggingface(hf_token, hf_repo_id)
        except Exception as exc:
            print(f"HuggingFace upload skipped due to error: {exc}")
    else:
        print("HuggingFace upload skipped: set HF_TOKEN and HF_REPO_ID in .env to enable.")


if __name__ == "__main__":
    main()
