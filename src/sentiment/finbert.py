from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import settings

SENTIMENT_FEATURES = [
    "positive_score",
    "negative_score",
    "neutral_score",
    "sentiment_score",
    "sentiment_confidence",
    "article_count",
    "message_count",
]


@dataclass
class SentimentResult:
    label: str
    positive: float
    negative: float
    neutral: float
    sentiment_score: float
    sentiment_confidence: float


class FinBertSentimentExtractor:
    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        self.model_name = model_name or settings.finbert_model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        self.labels = ["positive", "negative", "neutral"]

    def predict_text(self, text: str) -> SentimentResult:
        if not text or not text.strip():
            return SentimentResult(
                label="neutral",
                positive=0.0,
                negative=0.0,
                neutral=1.0,
                sentiment_score=0.0,
                sentiment_confidence=1.0,
            )

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )

        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()

        scores = dict(zip(self.labels, probs, strict=True))

        label = max(scores, key=scores.get)

        return SentimentResult(
            label=label,
            positive=scores["positive"],
            negative=scores["negative"],
            neutral=scores["neutral"],
            sentiment_score=scores["positive"] - scores["negative"],
            sentiment_confidence=max(scores.values()),
        )

    def score_news_frame(
        self,
        news_frame: pd.DataFrame,
        text_column: str = "title",
    ) -> pd.DataFrame:
        rows = []

        for row in news_frame.to_dict(orient="records"):
            text = str(row.get(text_column, ""))
            result = self.predict_text(text)

            row["sentiment_label"] = result.label
            row["positive_score"] = result.positive
            row["negative_score"] = result.negative
            row["neutral_score"] = result.neutral
            row["sentiment_score"] = result.sentiment_score
            row["sentiment_confidence"] = result.sentiment_confidence

            rows.append(row)

        return pd.DataFrame(rows)

    def aggregate_daily_sentiment(
        self,
        scored_news_frame: pd.DataFrame,
        date_column: str | None = None,
        ticker_column: str | None = "ticker",
    ) -> pd.DataFrame:
        if scored_news_frame.empty:
            empty_date_column = date_column or _first_existing_date_column(scored_news_frame) or "date"
            group_columns = [empty_date_column]
            if ticker_column is not None and ticker_column in scored_news_frame.columns:
                group_columns.insert(0, ticker_column)
            return pd.DataFrame(columns=[*group_columns, *SENTIMENT_FEATURES])

        resolved_date_column = _resolve_date_column(scored_news_frame, date_column)
        group_columns = [resolved_date_column]
        if ticker_column is not None and ticker_column in scored_news_frame.columns:
            group_columns.insert(0, ticker_column)

        df = scored_news_frame.copy()
        df[resolved_date_column] = (
            pd.to_datetime(df[resolved_date_column], errors="coerce", utc=True)
            .dt.tz_convert(None)
            .dt.normalize()
        )
        df = df.dropna(subset=[resolved_date_column])

        daily = (
            df.groupby(group_columns)
            .agg(
                positive_score=("positive_score", "mean"),
                negative_score=("negative_score", "mean"),
                neutral_score=("neutral_score", "mean"),
                sentiment_score=("sentiment_score", "mean"),
                sentiment_confidence=("sentiment_confidence", "mean"),
                article_count=("sentiment_score", "count"),
                message_count=("sentiment_score", "count"),
            )
            .reset_index()
        )

        return daily


def _resolve_date_column(frame: pd.DataFrame, preferred: str | None) -> str:
    if preferred is not None:
        if preferred not in frame.columns:
            raise ValueError(f"Expected date column '{preferred}' in scored news data.")
        return preferred

    existing_date_column = _first_existing_date_column(frame)
    if existing_date_column is not None:
        return existing_date_column

    raise ValueError("Could not find a date column in scored news data.")


def _first_existing_date_column(frame: pd.DataFrame) -> str | None:
    for candidate in ["date", "Date", "published_at", "publishedAt"]:
        if candidate in frame.columns:
            return candidate
    return None
