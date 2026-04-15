from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import settings


@dataclass
class SentimentResult:
    label: str
    positive: float
    negative: float
    neutral: float
    score: float


class FinBertSentimentAnalyzer:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.finbert_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()
        self.labels = ["positive", "negative", "neutral"]

    def predict_text(self, text: str) -> SentimentResult:
        if not text or not text.strip():
            return SentimentResult("neutral", 0.0, 0.0, 1.0, 0.0)

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        with torch.no_grad():
            logits = self.model(**encoded).logits
            probabilities = torch.softmax(logits, dim=1).squeeze(0).tolist()

        scores = dict(zip(self.labels, probabilities, strict=True))
        label = max(scores, key=scores.get)
        return SentimentResult(
            label=label,
            positive=scores["positive"],
            negative=scores["negative"],
            neutral=scores["neutral"],
            score=scores["positive"] - scores["negative"],
        )

    def score_news_frame(self, news_frame: pd.DataFrame, text_column: str = "title") -> pd.DataFrame:
        scored_rows = []
        for row in news_frame.to_dict(orient="records"):
            result = self.predict_text(str(row.get(text_column, "")))
            row["sentiment_label"] = result.label
            row["positive_score"] = result.positive
            row["negative_score"] = result.negative
            row["neutral_score"] = result.neutral
            row["sentiment_score"] = result.score
            scored_rows.append(row)
        return pd.DataFrame(scored_rows)

