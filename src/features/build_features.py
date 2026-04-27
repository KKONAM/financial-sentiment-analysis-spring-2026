from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import indicators as ind

def build_indicators(
    symbols,
    dates,
    train_end="2021-12-31"
):
    feature_cols = ["returns", "rsi", "macd", "bbp", "momentum", "volume"]

    all_dfs = []

    for symbol in symbols:
        ind = ind.indicators(
            symbol=symbol,
            dates=dates
        )

        df = pd.DataFrame(index=dates)

        df["ticker"] = symbol
        df["returns"] = ind.get_returns().iloc[:, 0]
        df["rsi"] = ind.get_rsi_indicator().iloc[:, 0]
        df["macd"] = ind.get_macd_indicator().iloc[:, 0]
        df["bbp"] = ind.get_bollinger_bands_indicator().iloc[:, 0]
        df["momentum"] = ind.get_momentum_indicator().iloc[:, 0]
        df["volume"] = ind.volume.iloc[:, 0]

        # next-day label
        df["future_return"] = df["returns"].shift(-1)
        df["label"] = (df["future_return"] > 0).astype(int)

        df = df.reset_index().rename(columns={"index": "aDate"})

        all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)

    # remove NaNs from indicators + final shifted label
    full_df = full_df.dropna(subset=feature_cols + ["label"])

    scaled_dfs = []

    for ticker, df_ticker in full_df.groupby("ticker"):
        df_ticker = df_ticker.copy()

        train_mask = df_ticker["Date"] <= pd.to_datetime(train_end)

        scaler = StandardScaler()
        scaler.fit(df_ticker.loc[train_mask, feature_cols])

        df_ticker[feature_cols] = scaler.transform(df_ticker[feature_cols])

        scaled_dfs.append(df_ticker)

    full_scaled_df = pd.concat(scaled_dfs, ignore_index=True)

    return full_scaled_df