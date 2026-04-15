from __future__ import annotations

import numpy as np
import pandas as pd


MARKET_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "return_1d",
    "return_3d",
    "return_5d",
    "volatility_5d",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bollinger_mid_20",
    "bollinger_upper_20",
    "bollinger_lower_20",
    "bollinger_width_20",
]

SENTIMENT_FEATURES = [
    "article_count",
    "sentiment_mean",
    "sentiment_std",
    "positive_mean",
    "negative_mean",
    "neutral_mean",
    "headline_length_mean",
]


def aggregate_daily_sentiment(news_frame: pd.DataFrame) -> pd.DataFrame:
    if news_frame.empty:
        raise ValueError("News frame is empty; cannot build sentiment features.")

    frame = news_frame.copy()
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True)
    frame["date"] = frame["published_at"].dt.tz_convert(None).dt.normalize()
    frame["headline_length"] = frame["title"].fillna("").str.len()

    aggregated = (
        frame.groupby("date", as_index=False)
        .agg(
            article_count=("title", "count"),
            sentiment_mean=("sentiment_score", "mean"),
            sentiment_std=("sentiment_score", "std"),
            positive_mean=("positive_score", "mean"),
            negative_mean=("negative_score", "mean"),
            neutral_mean=("neutral_score", "mean"),
            headline_length_mean=("headline_length", "mean"),
        )
        .fillna(0.0)
    )
    return aggregated.sort_values("date").reset_index(drop=True)


def build_modeling_dataset(
    price_frame: pd.DataFrame,
    daily_sentiment_frame: pd.DataFrame,
    forecast_horizon: int = 1,
) -> pd.DataFrame:
    prices = price_frame.copy().sort_values("date").reset_index(drop=True)
    prices["return_1d"] = prices["close"].pct_change()
    prices["return_3d"] = prices["close"].pct_change(3)
    prices["return_5d"] = prices["close"].pct_change(5)
    prices["volatility_5d"] = prices["return_1d"].rolling(5).std()
    prices["rsi_14"] = _compute_rsi(prices["close"], window=14)

    ema_12 = prices["close"].ewm(span=12, adjust=False).mean()
    ema_26 = prices["close"].ewm(span=26, adjust=False).mean()
    prices["macd"] = ema_12 - ema_26
    prices["macd_signal"] = prices["macd"].ewm(span=9, adjust=False).mean()
    prices["macd_hist"] = prices["macd"] - prices["macd_signal"]

    rolling_mean_20 = prices["close"].rolling(20).mean()
    rolling_std_20 = prices["close"].rolling(20).std()
    prices["bollinger_mid_20"] = rolling_mean_20
    prices["bollinger_upper_20"] = rolling_mean_20 + 2.0 * rolling_std_20
    prices["bollinger_lower_20"] = rolling_mean_20 - 2.0 * rolling_std_20
    prices["bollinger_width_20"] = (
        prices["bollinger_upper_20"] - prices["bollinger_lower_20"]
    ) / prices["bollinger_mid_20"]

    prices["target_return"] = prices["close"].shift(-forecast_horizon) / prices["close"] - 1.0
    prices["target_direction"] = (prices["target_return"] > 0).astype(int)

    merged = prices.merge(daily_sentiment_frame, on="date", how="left")
    for column in SENTIMENT_FEATURES:
        if column in merged.columns:
            merged[column] = merged[column].fillna(0.0)

    return merged.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def create_sequences(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    features = frame[feature_columns].to_numpy(dtype=np.float32)
    targets = frame[target_column].to_numpy(dtype=np.float32)

    if len(frame) <= sequence_length:
        raise ValueError("Not enough rows to create sequences.")

    sequence_features = []
    sequence_targets = []
    for end_idx in range(sequence_length, len(frame)):
        start_idx = end_idx - sequence_length
        sequence_features.append(features[start_idx:end_idx])
        sequence_targets.append(targets[end_idx])

    return np.array(sequence_features), np.array(sequence_targets)


def _compute_rsi(close_series: pd.Series, window: int = 14) -> pd.Series:
    delta = close_series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    relative_strength = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))
    rsi = rsi.mask((avg_loss == 0.0) & (avg_gain > 0.0), 100.0)
    rsi = rsi.mask((avg_gain == 0.0) & (avg_loss > 0.0), 0.0)
    rsi = rsi.mask((avg_gain == 0.0) & (avg_loss == 0.0), 50.0)
    return rsi
