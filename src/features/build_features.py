from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from features.utils import (
        SENTIMENT_FEATURES,
        build_finbert_text,
        create_sequences,
        fill_missing_sentiment,
        filter_frame_by_date_range,
        first_existing_column,
        resolve_column,
    )
    from sentiment.finbert import FinBertSentimentExtractor
    from technical_indicators.indicators import (
        MARKET_FEATURES,
        build_market_indicators,
        build_price_feature_frame,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {"features", "sentiment", "technical_indicators"}:
        raise
    from src.features.utils import (
        SENTIMENT_FEATURES,
        build_finbert_text,
        create_sequences,
        fill_missing_sentiment,
        filter_frame_by_date_range,
        first_existing_column,
        resolve_column,
    )
    from src.sentiment.finbert import FinBertSentimentExtractor
    from src.technical_indicators.indicators import (
        MARKET_FEATURES,
        build_market_indicators,
        build_price_feature_frame,
    )


DATE_COLUMN_CANDIDATES = ["date", "Date", "created_at", "createdAt", "published_at", "publishedAt"]
TEXT_COLUMN_CANDIDATES = ["title", "headline", "body", "description", "content"]
STOCKTWITS_DATE_COLUMN_CANDIDATES = [
    "created_at",
    "date",
    "Date",
    "createdAt",
    "published_at",
    "publishedAt",
]
BULLISH_LABEL_PATTERN = "'Bullish'|\"Bullish\""
BEARISH_LABEL_PATTERN = "'Bearish'|\"Bearish\""


def build_daily_sentiment(
    news_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    text_column: str = "title",
    date_column: str = "date",
    ticker_column: str = "ticker",
) -> pd.DataFrame:
    if news_df.empty:
        return _empty_daily_sentiment_frame()

    filtered_news = filter_frame_by_date_range(
        news_df,
        date_column=date_column,
        start_date=start_date,
        end_date=end_date,
    )
    if filtered_news.empty:
        return _empty_daily_sentiment_frame()

    extractor = FinBertSentimentExtractor()
    scored_news = extractor.score_news_frame(
        news_frame=filtered_news,
        text_column=text_column,
    )
    daily_sentiment = extractor.aggregate_daily_sentiment(
        scored_news_frame=scored_news,
        date_column=date_column,
        ticker_column=ticker_column,
    )

    daily_sentiment = daily_sentiment.rename(columns={date_column: "Date"})
    daily_sentiment["Date"] = pd.to_datetime(daily_sentiment["Date"])
    return daily_sentiment


def build_modeling_dataset(
    price_frame: pd.DataFrame,
    daily_sentiment_frame: pd.DataFrame,
    forecast_horizon: int = 1,
) -> pd.DataFrame:
    prices = build_price_feature_frame(price_frame=price_frame, forecast_horizon=forecast_horizon)
    sentiment = _normalize_sentiment_dates(daily_sentiment_frame)
    merged = prices.merge(sentiment, on="Date", how="left")
    filled = fill_missing_sentiment(merged)
    finite_values = filled.replace([np.inf, -np.inf], np.nan)
    required_columns = MARKET_FEATURES + ["future_return", "label"]
    complete_rows = finite_values.dropna(subset=required_columns)

    return complete_rows.reset_index(drop=True)


def build_daily_sentiment_from_csv(
    csv_path: str | Path,
    start_date: str,
    end_date: str,
    ticker: str | None = None,
    text_column: str | None = None,
    date_column: str | None = None,
    ticker_column: str = "ticker",
) -> pd.DataFrame:
    news_df = pd.read_csv(csv_path, low_memory=False)

    if _is_stocktwits_raw_frame(news_df):
        return _build_daily_sentiment_from_stocktwits_labels(
            stocktwits_df=news_df,
            start_date=start_date,
            end_date=end_date,
            ticker=ticker,
            date_column=date_column,
            ticker_column=ticker_column,
        )

    prepared_news = prepare_news_for_sentiment(
        news_df=news_df,
        ticker=ticker,
        text_column=text_column,
        date_column=date_column,
        ticker_column=ticker_column,
        start_date=start_date,
        end_date=end_date,
    )

    return build_daily_sentiment(
        news_df=prepared_news,
        start_date=start_date,
        end_date=end_date,
        text_column="finbert_text",
        date_column="date",
        ticker_column="ticker",
    )


def build_daily_sentiment_from_local_csv(
    symbol: str,
    start_date: str,
    end_date: str,
    text_column: str | None = None,
    date_column: str | None = None,
    ticker_column: str = "ticker",
) -> pd.DataFrame:
    csv_path = _find_local_sentiment_csv(symbol)
    return build_daily_sentiment_from_csv(
        csv_path=csv_path,
        start_date=start_date,
        end_date=end_date,
        ticker=symbol,
        text_column=text_column,
        date_column=date_column,
        ticker_column=ticker_column,
    )


def prepare_news_for_sentiment(
    news_df: pd.DataFrame,
    ticker: str | None = None,
    text_column: str | None = None,
    date_column: str | None = None,
    ticker_column: str = "ticker",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    if news_df.empty:
        return pd.DataFrame(columns=["ticker", "date", "finbert_text"])

    frame = news_df.copy()
    resolved_date_column, resolved_text_column = _resolve_news_columns(
        frame=frame,
        date_column=date_column,
        text_column=text_column,
    )

    frame = _assign_ticker(
        frame=frame,
        ticker=ticker,
        ticker_column=ticker_column,
        data_name="News data",
    )
    frame["date"] = _to_normalized_datetime(frame[resolved_date_column])
    frame["finbert_text"] = build_finbert_text(frame, resolved_text_column)
    frame = frame.dropna(subset=["ticker", "date"])

    if start_date is not None and end_date is not None:
        frame = filter_frame_by_date_range(
            frame,
            date_column="date",
            start_date=start_date,
            end_date=end_date,
        )

    frame = frame[frame["finbert_text"].str.strip().astype(bool)]
    return frame[["ticker", "date", "finbert_text"]].reset_index(drop=True)


def combine_indicators_and_sentiment(
    symbols,
    start_date: str,
    end_date: str,
    news_df: pd.DataFrame | None = None,
    news_csv_path: str | Path | None = None,
    train_end: str = "2021-04-30",
    forecast_horizon: int = 1,
    text_column: str = "title",
    news_date_column: str = "date",
    news_ticker_column: str = "ticker",
) -> pd.DataFrame:
    symbols = list(symbols)

    indicator_df = build_market_indicators(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        train_end=train_end,
        forecast_horizon=forecast_horizon,
    )
    sentiment_df = _build_sentiment_for_merge(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        news_df=news_df,
        news_csv_path=news_csv_path,
        text_column=text_column,
        news_date_column=news_date_column,
        news_ticker_column=news_ticker_column,
    )

    combined = indicator_df.merge(sentiment_df, on=["ticker", "Date"], how="left")
    return fill_missing_sentiment(combined).sort_values(["ticker", "Date"]).reset_index(drop=True)


def build_lstm_sequences(
    df: pd.DataFrame,
    sequence_length: int = 30,
    target_column: str = "label",
    include_current_row: bool = False,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    feature_cols = MARKET_FEATURES + SENTIMENT_FEATURES
    required_columns = [*feature_cols, target_column, "ticker", "Date"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns for LSTM sequence creation: {missing_columns}")

    features, labels = create_sequences(
        frame=df,
        feature_columns=feature_cols,
        target_column=target_column,
        sequence_length=sequence_length,
        include_current_row=include_current_row,
    )
    metadata = _build_sequence_metadata(
        df,
        sequence_length=sequence_length,
        include_current_row=include_current_row,
    )
    return features, labels, metadata


def _build_sentiment_for_merge(
    symbols: list[str],
    start_date: str,
    end_date: str,
    news_df: pd.DataFrame | None,
    news_csv_path: str | Path | None,
    text_column: str,
    news_date_column: str,
    news_ticker_column: str,
) -> pd.DataFrame:
    csv_text_column, csv_date_column = _resolve_csv_column_overrides(
        text_column=text_column,
        news_date_column=news_date_column,
    )

    if news_csv_path is not None:
        return build_daily_sentiment_from_csv(
            csv_path=news_csv_path,
            start_date=start_date,
            end_date=end_date,
            ticker=symbols[0] if len(symbols) == 1 else None,
            text_column=csv_text_column,
            date_column=csv_date_column,
            ticker_column=news_ticker_column,
        )

    if news_df is None:
        daily_sentiment_frames = _build_local_daily_sentiment_frames(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            text_column=csv_text_column,
            date_column=csv_date_column,
            ticker_column=news_ticker_column,
        )
        return pd.concat(daily_sentiment_frames, ignore_index=True)

    return build_daily_sentiment(
        news_df=news_df,
        start_date=start_date,
        end_date=end_date,
        text_column=text_column,
        date_column=news_date_column,
        ticker_column=news_ticker_column,
    )


def _is_stocktwits_raw_frame(frame: pd.DataFrame) -> bool:
    return {"created_at", "entities"}.issubset(frame.columns)


def _resolve_csv_column_overrides(
    text_column: str,
    news_date_column: str,
) -> tuple[str | None, str | None]:
    csv_text_column = None if text_column == "title" else text_column
    csv_date_column = None if news_date_column == "date" else news_date_column
    return csv_text_column, csv_date_column


def _build_local_daily_sentiment_frames(
    symbols: list[str],
    start_date: str,
    end_date: str,
    text_column: str | None,
    date_column: str | None,
    ticker_column: str,
) -> list[pd.DataFrame]:
    daily_sentiment_frames = []

    for symbol in symbols:
        daily_sentiment = build_daily_sentiment_from_local_csv(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            text_column=text_column,
            date_column=date_column,
            ticker_column=ticker_column,
        )
        daily_sentiment_frames.append(daily_sentiment)

    return daily_sentiment_frames


def _build_daily_sentiment_from_stocktwits_labels(
    stocktwits_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    ticker: str | None,
    date_column: str | None,
    ticker_column: str,
) -> pd.DataFrame:
    frame = stocktwits_df.copy()
    resolved_date_column = resolve_column(
        frame,
        preferred=date_column,
        candidates=STOCKTWITS_DATE_COLUMN_CANDIDATES,
        column_kind="date",
    )

    frame["Date"] = _to_normalized_datetime(frame[resolved_date_column])
    frame = frame.dropna(subset=["Date"])
    frame = filter_frame_by_date_range(
        frame,
        date_column="Date",
        start_date=start_date,
        end_date=end_date,
    )

    frame = _assign_ticker(
        frame=frame,
        ticker=ticker,
        ticker_column=ticker_column,
        data_name="StockTwits data",
    )
    scored_frame = _add_stocktwits_label_scores(frame)
    daily = _aggregate_stocktwits_daily_scores(scored_frame)
    daily["sentiment_confidence"] = _max_sentiment_confidence(daily)

    return daily[["ticker", "Date", *SENTIMENT_FEATURES]]


def _resolve_news_columns(
    frame: pd.DataFrame,
    date_column: str | None,
    text_column: str | None,
) -> tuple[str, str]:
    resolved_date_column = resolve_column(
        frame,
        preferred=date_column,
        candidates=DATE_COLUMN_CANDIDATES,
        column_kind="date",
    )
    resolved_text_column = resolve_column(
        frame,
        preferred=text_column,
        candidates=TEXT_COLUMN_CANDIDATES,
        column_kind="text",
    )
    return resolved_date_column, resolved_text_column


def _assign_ticker(
    frame: pd.DataFrame,
    ticker: str | None,
    ticker_column: str,
    data_name: str,
) -> pd.DataFrame:
    frame = frame.copy()

    if ticker_column in frame.columns:
        frame["ticker"] = frame[ticker_column]
        return frame

    if ticker is not None:
        frame["ticker"] = ticker
        return frame

    raise ValueError(
        f"{data_name} must include a '{ticker_column}' column or receive a ticker argument "
        "so sentiment can be merged with indicator features."
    )


def _to_normalized_datetime(values: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(values, errors="coerce", utc=True)
    timestamps = timestamps.dt.tz_convert(None)
    return timestamps.dt.normalize()


def _add_stocktwits_label_scores(frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    entities = scored["entities"].fillna("").astype(str)
    bullish = entities.str.contains(BULLISH_LABEL_PATTERN, regex=True)
    bearish = entities.str.contains(BEARISH_LABEL_PATTERN, regex=True)

    scored["is_labeled"] = bullish | bearish
    scored["positive_score"] = bullish.astype(float)
    scored["negative_score"] = bearish.astype(float)
    scored["neutral_score"] = 0.0
    scored["sentiment_score"] = scored["positive_score"] - scored["negative_score"]
    return scored


def _aggregate_stocktwits_daily_scores(frame: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["ticker", "Date"]
    message_counts = (
        frame.groupby(group_columns)
        .size()
        .rename("message_count")
        .reset_index()
    )
    labeled = frame[frame["is_labeled"]]

    if labeled.empty:
        daily = message_counts
        daily["positive_score"] = np.nan
        daily["negative_score"] = np.nan
        daily["neutral_score"] = np.nan
        daily["sentiment_score"] = np.nan
        daily["article_count"] = 0
        return daily

    daily_scores = (
        labeled.groupby(group_columns)
        .agg(
            positive_score=("positive_score", "mean"),
            negative_score=("negative_score", "mean"),
            neutral_score=("neutral_score", "mean"),
            sentiment_score=("sentiment_score", "mean"),
            article_count=("sentiment_score", "count"),
        )
        .reset_index()
    )
    daily = message_counts.merge(daily_scores, on=group_columns, how="left")
    daily["article_count"] = daily["article_count"].fillna(0).astype(int)
    return daily


def _max_sentiment_confidence(frame: pd.DataFrame) -> pd.Series:
    score_columns = ["positive_score", "negative_score", "neutral_score"]
    return frame[score_columns].max(axis=1)


def _normalize_sentiment_dates(daily_sentiment_frame: pd.DataFrame) -> pd.DataFrame:
    sentiment = daily_sentiment_frame.copy()

    if "Date" not in sentiment.columns:
        date_column = first_existing_column(sentiment, ["date", "published_at", "publishedAt"])
        sentiment = sentiment.rename(columns={date_column: "Date"})

    if sentiment.empty:
        return sentiment

    sentiment["Date"] = pd.to_datetime(sentiment["Date"]).dt.normalize()
    return sentiment


def _build_sequence_metadata(
    df: pd.DataFrame,
    sequence_length: int,
    include_current_row: bool,
) -> pd.DataFrame:
    rows = []
    for ticker, ticker_df in df.groupby("ticker"):
        ticker_df = ticker_df.sort_values("Date").reset_index(drop=True)
        start_idx = sequence_length - 1 if include_current_row else sequence_length
        for idx in range(start_idx, len(ticker_df)):
            rows.append({"ticker": ticker, "Date": ticker_df.loc[idx, "Date"]})

    if not rows:
        raise ValueError("Not enough rows to create LSTM sequences.")
    return pd.DataFrame(rows)


def _empty_daily_sentiment_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["ticker", "Date", *SENTIMENT_FEATURES])


def _find_local_sentiment_csv(symbol: str) -> Path:
    normalized_symbol = "FB" if symbol.upper() == "META" else symbol.upper()
    data_dir = Path(__file__).resolve().parents[2] / "data" / "Sentiment_Year_Data"

    csv_path = next(
        (path for path in data_dir.rglob("*.csv") if path.name.lower().startswith(normalized_symbol.lower())),
        None,
    )
    if csv_path is None:
        raise FileNotFoundError(f"No local sentiment CSV found for {normalized_symbol} under {data_dir}.")

    return csv_path
