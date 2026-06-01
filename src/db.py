"""Async MongoDB access for predictions, metrics, and retraining queue."""

import os
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "telecom_churn")

PREDICTIONS_COLLECTION = "predictions"
METRICS_COLLECTION = "model_metrics"
RETRAIN_QUEUE_COLLECTION = "retrain_queue"

_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    """Return a shared Motor client, creating it on first use."""
    global _client
    if _client is None:
        if not MONGODB_URI:
            raise ValueError("MONGODB_URI is not set in the environment.")
        _client = AsyncIOMotorClient(MONGODB_URI)
    return _client


def get_database() -> AsyncIOMotorDatabase:
    """Return the configured application database."""
    return get_client()[MONGODB_DB_NAME]


async def close_client() -> None:
    """Close the MongoDB client connection."""
    global _client
    if _client is not None:
        _client.close()
        _client = None


async def ping_database() -> bool:
    """Return True if the database responds to a ping."""
    try:
        await get_client().admin.command("ping")
        return True
    except Exception:
        return False


async def save_prediction(
    customer_id: str,
    input_features: dict[str, Any],
    churn_probability: float,
    churn_prediction: bool,
    model_version: str,
) -> str:
    """Insert a prediction record and return its document id."""
    record = {
        "customerID": customer_id,
        "input_features": input_features,
        "churn_probability": churn_probability,
        "churn_prediction": churn_prediction,
        "timestamp": datetime.now(timezone.utc),
        "model_version": model_version,
    }
    result = await get_database()[PREDICTIONS_COLLECTION].insert_one(record)
    return str(result.inserted_id)


async def get_predictions(skip: int = 0, limit: int = 20) -> list[dict[str, Any]]:
    """Return recent predictions with pagination (newest first)."""
    cursor = (
        get_database()[PREDICTIONS_COLLECTION]
        .find({}, {"_id": 0})
        .sort("timestamp", -1)
        .skip(skip)
        .limit(limit)
    )
    return await cursor.to_list(length=limit)


async def save_metrics(metrics: dict[str, Any]) -> str:
    """Insert a model metrics record and return its document id."""
    record = {
        **metrics,
        "timestamp": datetime.now(timezone.utc),
    }
    result = await get_database()[METRICS_COLLECTION].insert_one(record)
    return str(result.inserted_id)


async def get_latest_metrics() -> dict[str, Any] | None:
    """Return the most recent model metrics record, or None if missing."""
    return await get_database()[METRICS_COLLECTION].find_one(
        {},
        {"_id": 0},
        sort=[("timestamp", -1)],
    )


async def add_to_retrain_queue(rows: list[dict[str, Any]]) -> int:
    """Insert new rows into the retrain queue with status pending."""
    if not rows:
        return 0
    documents = [
        {**row, "status": "pending", "timestamp": datetime.now(timezone.utc)}
        for row in rows
    ]
    result = await get_database()[RETRAIN_QUEUE_COLLECTION].insert_many(documents)
    return len(result.inserted_ids)


async def count_pending_retrain_rows() -> int:
    """Count retrain queue documents with status pending."""
    return await get_database()[RETRAIN_QUEUE_COLLECTION].count_documents(
        {"status": "pending"}
    )


async def mark_retrain_queue_done() -> int:
    """Mark all pending retrain queue rows as done."""
    result = await get_database()[RETRAIN_QUEUE_COLLECTION].update_many(
        {"status": "pending"},
        {"$set": {"status": "done", "completed_at": datetime.now(timezone.utc)}},
    )
    return result.modified_count
