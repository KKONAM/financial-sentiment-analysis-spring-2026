from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from config import settings
from data_sources.market_data import download_price_history, save_price_history
from data_sources.newsapi_client import NewsApiClient
from features.build_features import aggregate_daily_sentiment, build_modeling_dataset
from sentiment.finbert import FinBertSentimentAnalyzer
from temporal.gru import GruTrainingResult, train_gru_model
from temporal.lstm import LstmTrainingResult, train_lstm_model


def fetch_news(query: str, from_date: str, to_date: str) -> Path:
    settings.ensure_directories()
    client = NewsApiClient()
    news_frame = client.fetch_everything(query=query, from_date=from_date, to_date=to_date)
    output_path = settings.raw_data_dir / f"{_safe_name(f'news_{query}_{from_date}_{to_date}')}.csv"
    return client.save_news(news_frame, output_path)


def convert_news_json(json_path: Path, output_path: Path | None = None) -> Path:
    settings.ensure_directories()
    frame = pd.read_json(json_path)
    articles = pd.json_normalize(frame["articles"])
    articles = articles.rename(
        columns={
            "source.id": "source_id",
            "source.name": "source_name",
            "publishedAt": "published_at",
            "urlToImage": "url_to_image",
        }
    )
    if output_path is None:
        output_path = settings.raw_data_dir / f"{json_path.stem}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    articles.to_csv(output_path, index=False)
    return output_path


def build_dataset(ticker: str, from_date: str, to_date: str, news_path: Path) -> Path:
    settings.ensure_directories()
    news_frame = pd.read_csv(news_path)
    analyzer = FinBertSentimentAnalyzer()
    scored_news = analyzer.score_news_frame(news_frame, text_column="title")
    daily_sentiment = aggregate_daily_sentiment(scored_news)

    price_frame = download_price_history(ticker=ticker, from_date=from_date, to_date=to_date)
    dataset = build_modeling_dataset(price_frame=price_frame, daily_sentiment_frame=daily_sentiment, forecast_horizon=settings.forecast_horizon)

    save_price_history(price_frame, settings.raw_data_dir / f"prices_{ticker}_{from_date}_{to_date}.csv")
    output_path = settings.processed_data_dir / f"{ticker}_dataset.csv"
    dataset.to_csv(output_path, index=False)
    return output_path


def run_lstm_training(dataset_path: Path) -> LstmTrainingResult:
    return train_lstm_model(pd.read_csv(dataset_path), sequence_length=settings.sequence_length, train_split=settings.train_split)


def run_gru_training(dataset_path: Path) -> GruTrainingResult:
    return train_gru_model(pd.read_csv(dataset_path), sequence_length=settings.sequence_length, train_split=settings.train_split)


def save_torch_artifact(model: torch.nn.Module, filename: str) -> Path:
    settings.ensure_directories()
    output_path = settings.artifacts_dir / filename
    torch.save(model.state_dict(), output_path)
    return output_path


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_").lower()

