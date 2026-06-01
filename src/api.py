"""FastAPI application for telecom churn prediction."""

import io
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from src import db
from src.preprocess import load_preprocessor, transform_features
from src.retrain import run_retraining
from src.schemas import (
    BatchPredictionItem,
    BatchPredictionResponse,
    CustomerInput,
    HealthResponse,
    MetricsResponse,
    PredictionResponse,
)

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/churn_model.pkl"))
METRICS_PATH = Path("models/model_metrics.json")
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", "")
FRONTEND_DIR = PROJECT_ROOT / "frontend"


class AppState:
    model = None
    preprocessor: dict | None = None
    model_version: str = "v1.0"


app_state = AppState()


def _load_model_version() -> str:
    if METRICS_PATH.exists():
        with METRICS_PATH.open(encoding="utf-8") as handle:
            return json.load(handle).get("model_version", "v1.0")
    return "v1.0"


def _load_artifacts() -> None:
    preprocessor_path = Path("models/preprocessor.pkl")
    hf_token = os.getenv("HF_TOKEN", "")
    hf_repo_id = os.getenv("HF_REPO_ID", "")

    if (not MODEL_PATH.exists() or not preprocessor_path.exists()) and hf_token and hf_repo_id:
        if hf_token != "your_huggingface_token_here":
            from src.model_manager import download_model_from_hub

            print("Downloading model artifacts from Hugging Face Hub...")
            download_model_from_hub()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    app_state.model = joblib.load(MODEL_PATH)
    app_state.preprocessor = load_preprocessor()
    app_state.model_version = _load_model_version()


def _predict_dataframe(df: pd.DataFrame) -> list[dict]:
    features = transform_features(df, artifact=app_state.preprocessor)
    probabilities = app_state.model.predict_proba(features)[:, 1]
    predictions = app_state.model.predict(features).astype(bool)

    customer_ids = (
        df["customerID"].astype(str).tolist()
        if "customerID" in df.columns
        else [f"row_{index}" for index in range(len(df))]
    )

    return [
        {
            "customer_id": customer_id,
            "churn_probability": float(probability),
            "churn_prediction": bool(prediction),
        }
        for customer_id, probability, prediction in zip(
            customer_ids, probabilities, predictions
        )
    ]


class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in {"/health", "/"}:
            return await call_next(request)
        if not APP_SECRET_KEY:
            return await call_next(request)

        api_key = request.headers.get("X-API-Key")
        if api_key != APP_SECRET_KEY:
            return JSONResponse(status_code=401, content={"detail": "Invalid API key"})
        return await call_next(request)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_artifacts()
    yield
    await db.close_client()


app = FastAPI(title="Telecom Churn Prediction API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(APIKeyMiddleware)

if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")


@app.get("/", response_class=FileResponse)
async def serve_frontend():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path)


@app.get("/health", response_model=HealthResponse)
async def health():
    db_connected = False
    if os.getenv("MONGODB_URI"):
        try:
            db_connected = await db.ping_database()
        except Exception:
            db_connected = False

    return HealthResponse(
        status="ok",
        model_version=app_state.model_version,
        model_loaded=app_state.model is not None,
        database_connected=db_connected,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerInput, x_api_key: str | None = Header(default=None)):
    _ = x_api_key
    row = pd.DataFrame([customer.model_dump()])
    result = _predict_dataframe(row)[0]
    timestamp = datetime.now(timezone.utc)

    if os.getenv("MONGODB_URI"):
        try:
            await db.save_prediction(
                customer_id=result["customer_id"],
                input_features=customer.model_dump(),
                churn_probability=result["churn_probability"],
                churn_prediction=result["churn_prediction"],
                model_version=app_state.model_version,
            )
        except Exception:
            pass

    return PredictionResponse(
        customer_id=result["customer_id"],
        churn_prediction=result["churn_prediction"],
        churn_probability=result["churn_probability"],
        model_version=app_state.model_version,
        timestamp=timestamp,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    if "customerID" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must include customerID column")

    results = _predict_dataframe(df)
    if os.getenv("MONGODB_URI"):
        for row, prediction in zip(df.to_dict(orient="records"), results):
            try:
                await db.save_prediction(
                    customer_id=prediction["customer_id"],
                    input_features=row,
                    churn_probability=prediction["churn_probability"],
                    churn_prediction=prediction["churn_prediction"],
                    model_version=app_state.model_version,
                )
            except Exception:
                pass

    items = [BatchPredictionItem(**item) for item in results]
    churn_count = sum(1 for item in items if item.churn_prediction)
    return BatchPredictionResponse(
        predictions=items,
        total=len(items),
        churn_count=churn_count,
        model_version=app_state.model_version,
    )


@app.get("/predictions")
async def list_predictions(skip: int = 0, limit: int = 20):
    if not os.getenv("MONGODB_URI"):
        return []
    try:
        return await db.get_predictions(skip=skip, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    if os.getenv("MONGODB_URI"):
        try:
            latest = await db.get_latest_metrics()
            if latest:
                xgb = latest.get("xgboost", latest)
                return MetricsResponse(
                    model_version=latest.get("model_version"),
                    roc_auc=xgb.get("roc_auc"),
                    precision=xgb.get("precision"),
                    recall=xgb.get("recall"),
                    f1=xgb.get("f1"),
                    trained_at=latest.get("trained_at"),
                    source="mongodb",
                )
        except Exception:
            pass

    if METRICS_PATH.exists():
        with METRICS_PATH.open(encoding="utf-8") as handle:
            data = json.load(handle)
        xgb = data.get("xgboost", {})
        return MetricsResponse(
            model_version=data.get("model_version"),
            roc_auc=xgb.get("roc_auc"),
            precision=xgb.get("precision"),
            recall=xgb.get("recall"),
            f1=xgb.get("f1"),
            trained_at=data.get("trained_at"),
            source="file",
        )

    raise HTTPException(status_code=404, detail="No metrics available")


@app.post("/retrain")
async def retrain(file: UploadFile = File(...)):
    upload_dir = PROJECT_ROOT / "data" / "processed"
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / f"retrain_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.csv"
    content = await file.read()
    destination.write_bytes(content)

    try:
        result = await run_retraining(destination)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if result.get("retrained"):
        _load_artifacts()

    return result
