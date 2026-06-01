---
title: Telecom Churn Prediction
sdk: docker
app_port: 8000
---

# Telecom Churn Prediction

ML-powered churn prediction API and web app.

**Stack:** Python · FastAPI · Docker · MongoDB Atlas · GitHub · HuggingFace

## Project structure

```
src/                 # Application source code
data/raw/            # Raw CSV datasets
data/processed/      # Processed data (gitignored)
models/              # Trained models and preprocessors
tests/               # Pytest tests
frontend/            # Static web UI
.github/workflows/   # CI/CD pipelines
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your credentials (see development plan).

## Train the model

```bash
python src/train.py
```

## Run the API locally

```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 for the web UI.

## Docker

```bash
docker compose up --build
```

## Tests

```bash
pytest tests/smoke_test.py -v
```

## Required environment variables

| Variable | Description |
|----------|-------------|
| `MONGODB_URI` | MongoDB Atlas connection string |
| `MONGODB_DB_NAME` | Database name (default: `telecom_churn`) |
| `HF_TOKEN` | Hugging Face write token |
| `HF_REPO_ID` | Model repo, e.g. `username/telecom-churn-model` |
| `HF_SPACE_ID` | Space repo, e.g. `username/telecom-churn-app` |
| `APP_SECRET_KEY` | API key for protected endpoints |
| `MODEL_PATH` | Path to `churn_model.pkl` |
| `DATA_PATH` | Path to training CSV |
| `RETRAIN_THRESHOLD` | Pending rows before retrain runs |

## License

MIT
