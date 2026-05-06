from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


def download_price_history(ticker: str, from_date: str, to_date: str) -> pd.DataFrame:
    frame = yf.download(ticker, start=from_date, end=to_date, auto_adjust=True, progress=False)
    if frame.empty:
        raise ValueError(f"No market data returned for ticker {ticker}.")

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    frame = frame.reset_index()
    frame.columns = [str(col).lower().replace(" ", "_") for col in frame.columns]
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    return frame


def save_price_history(price_frame: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    price_frame.to_csv(output_path, index=False)
    return output_path

