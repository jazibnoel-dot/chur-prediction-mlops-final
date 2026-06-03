"""Pydantic request/response models for the churn prediction API."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    customerID: str
    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: str
    InternetService: Literal["No", "DSL", "Fiber optic"]
    OnlineSecurity: str
    TechSupport: str
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | str


class PredictionResponse(BaseModel):
    customer_id: str
    churn_prediction: bool
    churn_probability: float
    model_version: str
    timestamp: datetime


class BatchPredictionItem(BaseModel):
    customer_id: str
    churn_prediction: bool
    churn_probability: float


class BatchPredictionResponse(BaseModel):
    predictions: list[BatchPredictionItem]
    total: int
    churn_count: int
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_loaded: bool
    database_connected: bool


class MetricsResponse(BaseModel):
    model_version: str | None = None
    accuracy: float | None = None
    roc_auc: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    trained_at: str | None = None
    source: str


class RetrainStatusResponse(BaseModel):
    retrain_threshold: int
    pending_rows: int
    rows_until_retrain: int
    current_metrics: MetricsResponse | None = None


class RetrainResultResponse(BaseModel):
    queued_rows: int
    pending_rows: int
    retrain_threshold: int
    retrained: bool
    message: str
    model_version: str | None = None
    accuracy: float | None = None
    roc_auc: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    upload_error: str | None = None
