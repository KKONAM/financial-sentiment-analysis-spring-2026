from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MARKET_FEATURES = ["returns", "rsi", "macd", "bbp", "momentum", "volume"]
INDICATOR_WARMUP_DAYS = 120


def build_market_indicators(
    symbols,
    start_date: str,
    end_date: str,
    train_end: str = "2021-04-30",
    forecast_horizon: int = 1,
) -> pd.DataFrame:
    symbols = list(symbols)
    if not symbols:
        raise ValueError("At least one symbol is required.")

    output_dates = _build_business_date_range(start_date=start_date, end_date=end_date)
    indicator_dates = _build_indicator_date_range(
        start_date=start_date,
        end_date=end_date,
        warmup_days=INDICATOR_WARMUP_DAYS,
    )
    frames = [
        _build_symbol_indicator_frame(
            symbol=symbol,
            dates=indicator_dates,
            forecast_horizon=forecast_horizon,
        )
        for symbol in symbols
    ]
    full_frame = pd.concat(frames, ignore_index=True)

    full_frame["Date"] = pd.to_datetime(full_frame["Date"])
    full_frame = full_frame[
        (full_frame["Date"] >= output_dates.min()) & (full_frame["Date"] <= output_dates.max())
    ]
    full_frame = full_frame.dropna(subset=MARKET_FEATURES + ["future_return", "label", "future_return_5d"])

    return full_frame.sort_values(["ticker", "Date"]).reset_index(drop=True)


def build_price_feature_frame(price_frame: pd.DataFrame, forecast_horizon: int = 1) -> pd.DataFrame:
    prices = price_frame.copy()
    prices.columns = [str(column).lower() for column in prices.columns]
    required_columns = {"date", "close", "volume"}
    missing_columns = required_columns - set(prices.columns)
    if missing_columns:
        raise ValueError(f"Missing required price columns: {sorted(missing_columns)}")

    prices = prices.sort_values("date").reset_index(drop=True)
    prices["Date"] = pd.to_datetime(prices["date"]).dt.normalize()

    close = prices["close"].ffill()
    volume = prices["volume"].ffill()
    prices["returns"] = close.pct_change()
    prices["rsi"] = compute_rsi(close)
    prices["macd"] = compute_macd_signal(close)
    prices["bbp"] = compute_bollinger_band_percent(close)
    prices["momentum"] = close / close.shift(14) - 1.0
    prices["volume"] = volume

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
    return price_frame.ffill()


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
    if "Adj Close" in price_frame.columns:
        price_frame["Close"] = price_frame["Adj Close"]
    price_frame[["Date", "Open", "High", "Low", "Close", "Volume"]].to_csv(output_path, index=False)
    return output_path


def compute_rsi(close_series: pd.Series, window: int = 14) -> pd.Series:
    delta = close_series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = losses.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    relative_strength = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))
    rsi = rsi.mask((avg_loss == 0.0) & (avg_gain > 0.0), 100.0)
    rsi = rsi.mask((avg_gain == 0.0) & (avg_loss > 0.0), 0.0)
    rsi = rsi.mask((avg_gain == 0.0) & (avg_loss == 0.0), 50.0)
    return rsi


def compute_macd_signal(close_series: pd.Series, window: int = 14) -> pd.Series:
    macd = close_series.ewm(span=12, adjust=False, min_periods=12).mean() - close_series.ewm(
        span=26,
        adjust=False,
        min_periods=26,
    ).mean()
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    histogram = macd - signal
    return histogram / close_series.replace(0.0, np.nan)


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


def _build_indicator_date_range(start_date: str, end_date: str, warmup_days: int) -> pd.DatetimeIndex:
    start_ts = pd.to_datetime(start_date).normalize() - pd.tseries.offsets.BDay(warmup_days)
    end_ts = pd.to_datetime(end_date).normalize()
    return _build_business_date_range(start_date=start_ts.strftime("%Y-%m-%d"), end_date=end_ts.strftime("%Y-%m-%d"))


def _build_symbol_indicator_frame(
    symbol: str,
    dates: pd.DatetimeIndex,
    forecast_horizon: int,
) -> pd.DataFrame:
    price_frame = load_local_price_data(symbol=symbol, dates=dates)
    trading_dates = price_frame.index
    close = price_frame["Close"].ffill()
    volume = price_frame["Volume"].ffill()

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
    frame["future_return"] = close.shift(-forecast_horizon).to_numpy() / close.to_numpy() - 1.0
    frame["future_return_5d"] = close.shift(-forecast_horizon).to_numpy() / close.to_numpy() - 1.0
    frame["label"] = (frame["future_return"] > 0).astype(int)
    frame["target_direction"] = frame["label"]
    return frame

