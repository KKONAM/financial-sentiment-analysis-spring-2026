from __future__ import annotations

import numpy as np
import pandas as pd


SENTIMENT_FEATURES = [
    "positive_score",
    "negative_score",
    "neutral_score",
    "sentiment_score",
    "sentiment_confidence",
    "article_count",
]


def filter_frame_by_date_range(
    df: pd.DataFrame,
    date_column: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    if date_column not in df.columns:
        raise ValueError(f"Expected date column '{date_column}' in data.")

    start_ts = pd.to_datetime(start_date).normalize()
    end_ts = pd.to_datetime(end_date).normalize()
    if start_ts > end_ts:
        raise ValueError("start_date must be before or equal to end_date.")

    filtered = df.copy()
    filtered[date_column] = pd.to_datetime(filtered[date_column], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    return filtered[(filtered[date_column] >= start_ts) & (filtered[date_column] <= end_ts)]


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Could not find any of these columns: {', '.join(candidates)}.")


def fill_missing_sentiment(frame: pd.DataFrame) -> pd.DataFrame:
    filled = frame.copy()
    fill_values = {
        "positive_score": 0.0,
        "negative_score": 0.0,
        "neutral_score": 1.0,
        "sentiment_score": 0.0,
        "sentiment_confidence": 1.0,
        "article_count": 0,
    }

    for column, value in fill_values.items():
        if column not in filled.columns:
            filled[column] = value
        else:
            filled[column] = filled[column].fillna(value)

    return filled


def resolve_column(
    df: pd.DataFrame,
    preferred: str | None,
    candidates: list[str],
    column_kind: str,
) -> str:
    if preferred is not None:
        if preferred not in df.columns:
            raise ValueError(f"Expected {column_kind} column '{preferred}' in news data.")
        return preferred

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise ValueError(f"Could not find a {column_kind} column. Tried: {', '.join(candidates)}.")


def build_finbert_text(df: pd.DataFrame, text_column: str) -> pd.Series:
    text = df[text_column].fillna("").astype(str)

    if text_column == "title" and "description" in df.columns:
        description = df["description"].fillna("").astype(str)
        text = (text + ". " + description).str.strip()

    return text


def create_sequences(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    sequence_length: int,
    include_current_row: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    missing_columns = [column for column in [*feature_columns, target_column] if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns for sequence creation: {missing_columns}")

    sequence_features = []
    sequence_targets = []

    if {"ticker", "Date"}.issubset(frame.columns):
        groups = (group.sort_values("Date") for _, group in frame.groupby("ticker"))
    else:
        groups = [frame]

    for group in groups:
        minimum_rows = sequence_length if include_current_row else sequence_length + 1
        if len(group) < minimum_rows:
            continue

        features = group[feature_columns].to_numpy(dtype=np.float32)
        targets = group[target_column].to_numpy(dtype=np.float32)

        start_end_idx = sequence_length - 1 if include_current_row else sequence_length
        for end_idx in range(start_end_idx, len(group)):
            if include_current_row:
                start_idx = end_idx - sequence_length + 1
                sequence_features.append(features[start_idx : end_idx + 1])
            else:
                start_idx = end_idx - sequence_length
                sequence_features.append(features[start_idx:end_idx])
            sequence_targets.append(targets[end_idx])

    if not sequence_features:
        raise ValueError("Not enough rows to create sequences.")

    return np.array(sequence_features), np.array(sequence_targets)
