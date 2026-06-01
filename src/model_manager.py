"""Download and hot-reload models from Hugging Face Hub."""

import os
from pathlib import Path

import joblib
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/churn_model.pkl"))
PREPROCESSOR_PATH = Path("models/preprocessor.pkl")


def download_model_from_hub(
    repo_id: str | None = None,
    token: str | None = None,
    model_path: Path = MODEL_PATH,
    preprocessor_path: Path = PREPROCESSOR_PATH,
) -> tuple[object, dict]:
    """Download model and preprocessor from Hugging Face and return loaded objects."""
    repo_id = repo_id or os.getenv("HF_REPO_ID", "")
    token = token or os.getenv("HF_TOKEN", "")
    if not repo_id:
        raise ValueError("HF_REPO_ID is not set.")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)

    downloaded_model = hf_hub_download(
        repo_id=repo_id,
        filename=model_path.name,
        repo_type="model",
        token=token or None,
        local_dir=str(model_path.parent),
        local_dir_use_symlinks=False,
    )
    downloaded_preprocessor = hf_hub_download(
        repo_id=repo_id,
        filename=preprocessor_path.name,
        repo_type="model",
        token=token or None,
        local_dir=str(preprocessor_path.parent),
        local_dir_use_symlinks=False,
    )

    model = joblib.load(downloaded_model)
    preprocessor = joblib.load(downloaded_preprocessor)
    return model, preprocessor


def reload_model(app_state) -> str:
    """Download latest artifacts from Hub and update FastAPI app state in memory."""
    model, preprocessor = download_model_from_hub()
    app_state.model = model
    app_state.preprocessor = preprocessor
    metrics_path = Path("models/model_metrics.json")
    if metrics_path.exists():
        import json

        with metrics_path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        app_state.model_version = metrics.get("model_version", app_state.model_version)
    return app_state.model_version
