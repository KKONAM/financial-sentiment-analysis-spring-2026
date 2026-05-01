from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


MARKET_FEATURES = ["returns", "rsi", "macd", "bbp", "momentum", "volume"]


def build_market_indicators(
    symbols,
    start_date: str,
    end_date: str,
    train_end: str = "2021-12-31",
    forecast_horizon: int = 1,
) -> pd.DataFrame:
    symbols = list(symbols)
    if not symbols:
        raise ValueError("At least one symbol is required.")

    dates = _build_business_date_range(start_date=start_date, end_date=end_date)
    frames = [
        _build_symbol_indicator_frame(
            symbol=symbol,
            dates=dates,
            forecast_horizon=forecast_horizon,
        )
        for symbol in symbols
    ]
    full_frame = pd.concat(frames, ignore_index=True)

    full_frame["Date"] = pd.to_datetime(full_frame["Date"])
    full_frame = full_frame.dropna(subset=MARKET_FEATURES + ["future_return", "label", "future_return_5d"])

    return _scale_market_features_by_ticker(full_frame, train_end=train_end)


def build_price_feature_frame(price_frame: pd.DataFrame, forecast_horizon: int = 1) -> pd.DataFrame:
    prices = price_frame.copy()
    prices.columns = [str(column).lower() for column in prices.columns]
    required_columns = {"date", "close", "volume"}
    missing_columns = required_columns - set(prices.columns)
    if missing_columns:
        raise ValueError(f"Missing required price columns: {sorted(missing_columns)}")

    prices = prices.sort_values("date").reset_index(drop=True)
    prices["Date"] = pd.to_datetime(prices["date"]).dt.normalize()

    close = prices["close"].ffill().bfill()
    prices["returns"] = close.pct_change()
    prices["rsi"] = compute_rsi(close)
    prices["macd"] = compute_macd_signal(close)
    prices["bbp"] = compute_bollinger_band_percent(close)
    prices["momentum"] = close / close.shift(14) - 1.0

    prices["future_return"] = close.shift(-forecast_horizon) / close - 1.0
    prices["label"] = (prices["future_return"] > 0).astype(int)
    prices["target_direction"] = prices["label"]

    return prices


def load_local_price_data(symbol: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    normalized_symbol = "META" if symbol.upper() == "FB" else symbol.upper()
    data_dir = Path(__file__).resolve().parents[2] / "data" / "Stock_Movement_Data"

    csv_path = next(
        (path for path in data_dir.rglob("*.csv") if path.name.lower().startswith(normalized_symbol.lower())),
        None,
    )
    if csv_path is None:
        csv_path = _download_local_price_data(
            symbol=normalized_symbol,
            start_date=dates.min(),
            end_date=dates.max(),
            data_dir=data_dir,
        )

    price_frame = pd.read_csv(
        csv_path,
        parse_dates=["Date"],
        dtype={
            "Open": "float64",
            "High": "float64",
            "Low": "float64",
            "Close": "float64",
            "Volume": "float64",
        },
    )
    price_frame = price_frame.set_index("Date").sort_index()
    price_frame = price_frame.loc[dates.min():dates.max()]
    if price_frame.empty:
        raise ValueError(
            f"No local price rows found for {normalized_symbol} between "
            f"{dates.min().date()} and {dates.max().date()}."
        )
    price_frame.index.name = "Date"
    return price_frame.ffill().bfill()


def _download_local_price_data(
    symbol: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    data_dir: Path,
) -> Path:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise FileNotFoundError(
            f"No local price CSV found for {symbol} under {data_dir}, and yfinance is not installed "
            "to download it."
        ) from exc

    data_dir.mkdir(parents=True, exist_ok=True)
    download_end = end_date + pd.Timedelta(days=1)
    price_frame = yf.download(
        symbol,
        start=start_date.strftime("%Y-%m-%d"),
        end=download_end.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )
    if price_frame.empty:
        raise FileNotFoundError(
            f"No local price CSV found for {symbol} under {data_dir}, and yfinance returned no data."
        )

    if isinstance(price_frame.columns, pd.MultiIndex):
        price_frame.columns = price_frame.columns.get_level_values(0)

    price_frame = price_frame.reset_index()
    output_path = data_dir / f"{symbol.lower()}_us_d.csv"
    price_frame[["Date", "Open", "High", "Low", "Close", "Volume"]].to_csv(output_path, index=False)
    return output_path


def compute_rsi(close_series: pd.Series, window: int = 14) -> pd.Series:
    delta = close_series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.rolling(window=window, min_periods=window).mean()
    avg_loss = losses.rolling(window=window, min_periods=window).mean()
    relative_strength = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))
    rsi = rsi.mask((avg_loss == 0.0) & (avg_gain > 0.0), 100.0)
    rsi = rsi.mask((avg_gain == 0.0) & (avg_loss > 0.0), 0.0)
    rsi = rsi.mask((avg_gain == 0.0) & (avg_loss == 0.0), 50.0)
    return rsi


def compute_macd_signal(close_series: pd.Series, window: int = 14) -> pd.Series:
    macd = close_series.ewm(span=12, adjust=False).mean() - close_series.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_norm = (macd - macd.rolling(window).mean()) / macd.rolling(window).std()
    signal_norm = (signal - signal.rolling(window).mean()) / signal.rolling(window).std()
    return macd_norm - signal_norm


def compute_bollinger_band_percent(close_series: pd.Series, window: int = 14) -> pd.Series:
    rolling_mean = close_series.rolling(window=window, min_periods=window).mean()
    rolling_std = close_series.rolling(window=window, min_periods=window).std()
    upper_band = rolling_mean + (rolling_std * 2.0)
    lower_band = rolling_mean - (rolling_std * 2.0)
    return (close_series - lower_band) / (upper_band - lower_band)


def _build_business_date_range(start_date: str, end_date: str) -> pd.DatetimeIndex:
    start_ts = pd.to_datetime(start_date).normalize()
    end_ts = pd.to_datetime(end_date).normalize()
    if start_ts > end_ts:
        raise ValueError("start_date must be before or equal to end_date.")

    dates = pd.bdate_range(start=start_ts, end=end_ts)
    if dates.empty:
        raise ValueError("Date range does not contain any business days.")
    return dates


def _build_symbol_indicator_frame(
    symbol: str,
    dates: pd.DatetimeIndex,
    forecast_horizon: int,
) -> pd.DataFrame:
    price_frame = load_local_price_data(symbol=symbol, dates=dates)
    trading_dates = price_frame.index
    close = price_frame["Close"].ffill().bfill()
    volume = price_frame["Volume"].ffill().bfill()

    frame = pd.DataFrame(
        {
            "Date": trading_dates,
            "ticker": symbol,
            "returns": close.pct_change().to_numpy(),
            "rsi": compute_rsi(close).to_numpy(),
            "macd": compute_macd_signal(close).to_numpy(),
            "bbp": compute_bollinger_band_percent(close).to_numpy(),
            "momentum": (close / close.shift(14) - 1.0).to_numpy(),
            "volume": volume.to_numpy(),
        }
    )
    frame["future_return"] = close.shift(-1).to_numpy() / close.to_numpy() - 1.0
    frame["future_return_5d"] = close.shift(-forecast_horizon).to_numpy() / close.to_numpy() - 1.0
    frame["label"] = (frame["future_return"] > 0).astype(int)
    frame["target_direction"] = frame["label"]
    return frame


def _scale_market_features_by_ticker(frame: pd.DataFrame, train_end: str) -> pd.DataFrame:
    train_end_ts = pd.to_datetime(train_end)
    scaled_frames = []

    for ticker, ticker_frame in frame.groupby("ticker"):
        ticker_frame = ticker_frame.copy()
        train_mask = ticker_frame["Date"] <= train_end_ts
        if not train_mask.any():
            raise ValueError(f"No training rows found for {ticker} on or before train_end={train_end_ts.date()}.")

        scaler = StandardScaler()
        scaler.fit(ticker_frame.loc[train_mask, MARKET_FEATURES])
        ticker_frame[MARKET_FEATURES] = scaler.transform(ticker_frame[MARKET_FEATURES])
        scaled_frames.append(ticker_frame)

    return pd.concat(scaled_frames, ignore_index=True)
