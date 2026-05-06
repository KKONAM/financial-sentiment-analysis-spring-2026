import os
import pandas as pd

def get_data(
    symbol,
    dates=pd.bdate_range(start='2020-01-01', end='2022-02-28')
):
    symbols_available = ["AAPL", "AMZN", "FB", "TSLA", "META", "NVDA"]

    if symbol in ["META", "FB"]:
        symbol = "META"

    if symbol not in symbols_available:
        raise ValueError(
            f"Symbol {symbol} is not available. Please choose from {symbols_available}"
        )

    root_dir = '../data/Stock_Movement_Data'
    csv_file = None

    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith('.csv') and fn.lower().startswith(symbol.lower()):
                csv_file = os.path.join(dirpath, fn)
                break
        if csv_file is not None:
            break

    if csv_file is None:
        raise FileNotFoundError(
            f"No CSV file found under {root_dir} for symbol: {symbol}"
        )

    df = pd.read_csv(
        csv_file,
        parse_dates=['Date'],
        dtype={
            'Open': 'float64',
            'High': 'float64',
            'Low': 'float64',
            'Close': 'float64',
            'Volume': 'int64'
        }
    )

    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)

    df = df.loc[dates.min():dates.max()]
    df = df.reindex(dates)

    return df[['High']], df[['Low']], df[['Close']], df[['Volume']]
