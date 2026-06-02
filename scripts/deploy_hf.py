"""Upload project to Hugging Face Model repo and Space."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)


def main() -> None:
    token = os.getenv("HF_TOKEN", "")
    repo_id = os.getenv("HF_REPO_ID", "")
    space_id = os.getenv("HF_SPACE_ID", "")

    if not token or token == "your_huggingface_token_here":
        print("Set HF_TOKEN in .env before deploying.")
        sys.exit(1)

    api = HfApi(token=token)

    if repo_id:
        api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
        for name in ("churn_model.pkl", "preprocessor.pkl"):
            path = PROJECT_ROOT / "models" / name
            if path.exists():
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=name,
                    repo_id=repo_id,
                    repo_type="model",
                )
                print(f"Uploaded {name} to {repo_id}")

    if space_id:
        api.create_repo(repo_id=space_id, exist_ok=True, repo_type="space", space_sdk="docker")
        api.upload_folder(
            folder_path=str(PROJECT_ROOT),
            repo_id=space_id,
            repo_type="space",
            ignore_patterns=[
                "venv/**",
                ".git/**",
                "__pycache__/**",
                "data/processed/**",
                "*.log",
                ".env",
            ],
        )
        # .gitignore excludes *.pkl, so upload model artifacts explicitly.
        for name in ("churn_model.pkl", "preprocessor.pkl", "model_metrics.json"):
            artifact = PROJECT_ROOT / "models" / name
            if artifact.exists():
                api.upload_file(
                    path_or_fileobj=str(artifact),
                    path_in_repo=f"models/{name}",
                    repo_id=space_id,
                    repo_type="space",
                )
                print(f"Uploaded models/{name} to Space {space_id}")
        print(f"Deployed Space: https://huggingface.co/spaces/{space_id}")
        print(f"Live URL: https://{space_id.replace('/', '-')}.hf.space")


if __name__ == "__main__":
    main()
