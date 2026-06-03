"""Continuous retraining pipeline with MongoDB queue and quality gate."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from src.data_loader import load_data
from src.db import (
    add_to_retrain_queue,
    count_pending_retrain_rows,
    get_latest_metrics,
    mark_retrain_queue_done,
    save_metrics,
)
from src.preprocess import preprocess
from src.train import MODEL_PATH, METRICS_PATH, XGB_PARAM_DIST, _evaluate_model, _upload_to_huggingface

load_dotenv()

RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", "100"))
DATA_PATH = Path(os.getenv("DATA_PATH", "data/raw/data.csv"))


def _next_version(current: str | None) -> str:
    if not current or not current.startswith("v"):
        return "v1.1"
    try:
        major, minor = current[1:].split(".", maxsplit=1)
        return f"v{major}.{int(minor) + 1}"
    except ValueError:
        return "v1.1"


async def run_retraining(new_csv_path: str | Path, force: bool = False) -> dict:
    """Queue new data and retrain when the pending threshold is met (or force=True)."""
    new_df = load_data(new_csv_path)
    queue_rows = new_df.to_dict(orient="records")
    queued = 0
    pending = len(queue_rows)

    if os.getenv("MONGODB_URI"):
        try:
            queued = await add_to_retrain_queue(queue_rows)
            pending = await count_pending_retrain_rows()
        except Exception as exc:
            if not force:
                raise RuntimeError(f"Could not update retrain queue: {exc}") from exc

    result = {
        "queued_rows": queued or len(queue_rows),
        "pending_rows": pending,
        "retrain_threshold": RETRAIN_THRESHOLD,
        "retrained": False,
        "message": "",
    }

    if not force and pending < RETRAIN_THRESHOLD:
        result["message"] = (
            f"Queued {queued} row(s). {pending}/{RETRAIN_THRESHOLD} pending "
            "rows required before retraining."
        )
        return result

    base_df = load_data(DATA_PATH) if DATA_PATH.exists() else pd.DataFrame()
    combined = pd.concat([base_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["customerID"], keep="last")

    processed_df = preprocess(combined)
    x = processed_df.drop(columns=["Churn"])
    y = processed_df["Churn"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

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
    )
    search.fit(x_train, y_train)
    best_model = search.best_estimator_
    new_metrics = _evaluate_model("Retrained XGBoost", best_model, x_test, y_test)

    latest = await get_latest_metrics()
    current_auc = None
    if latest and "xgboost" in latest:
        current_auc = latest["xgboost"].get("roc_auc")
    elif latest and "roc_auc" in latest:
        current_auc = latest["roc_auc"]
    elif METRICS_PATH.exists():
        with METRICS_PATH.open(encoding="utf-8") as handle:
            file_metrics = json.load(handle)
        current_auc = file_metrics.get("xgboost", {}).get("roc_auc")

    new_auc = new_metrics["roc_auc"]
    if current_auc is not None and new_auc < current_auc:
        result["message"] = (
            f"Retraining skipped: new ROC-AUC {new_auc:.4f} is below current {current_auc:.4f}."
        )
        result["roc_auc"] = new_auc
        result["accuracy"] = new_metrics.get("accuracy")
        result["precision"] = new_metrics.get("precision")
        result["recall"] = new_metrics.get("recall")
        result["f1"] = new_metrics.get("f1")
        return result

    model_version = _next_version(
        latest.get("model_version") if latest else None
    )
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    metrics_payload = {
        "model_version": model_version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "best_model": "XGBClassifier",
        "best_params": search.best_params_,
        "xgboost": new_metrics,
        "training_samples": len(x_train),
        "test_samples": len(x_test),
    }
    with METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    await save_metrics(metrics_payload)
    await mark_retrain_queue_done()

    hf_token = os.getenv("HF_TOKEN", "")
    hf_repo_id = os.getenv("HF_REPO_ID", "")
    if hf_token and hf_repo_id and hf_token != "your_huggingface_token_here":
        try:
            _upload_to_huggingface(hf_token, hf_repo_id)
        except Exception as exc:
            result["upload_error"] = str(exc)

    result.update(
        {
            "retrained": True,
            "model_version": model_version,
            "accuracy": new_metrics.get("accuracy"),
            "roc_auc": new_auc,
            "precision": new_metrics.get("precision"),
            "recall": new_metrics.get("recall"),
            "f1": new_metrics.get("f1"),
            "message": f"Retraining complete. New model version: {model_version}",
        }
    )
    return result
