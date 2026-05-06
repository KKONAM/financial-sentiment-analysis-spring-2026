from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    news_api_key: str = os.getenv("NEWS_API_KEY", "")
    news_api_base_url: str = os.getenv("NEWS_API_BASE_URL", "https://newsapi.org/v2")
    raw_data_dir: Path = ROOT_DIR / "data" / "raw"
    processed_data_dir: Path = ROOT_DIR / "data" / "processed"
    artifacts_dir: Path = ROOT_DIR / "artifacts"
    finbert_model_name: str = "ProsusAI/finbert"
    sequence_length: int = 5
    forecast_horizon: int = 1
    train_split: float = 0.8

    def ensure_directories(self) -> None:
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()

