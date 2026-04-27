import pandas as pd
from features.build_features import build_indicators



def main():
    technical_indicators_df = build_indicators(
        symbols=["AMZN", "META", "NVDA"],
        dates=pd.bdate_range(start='2019-01-01', end='2021  -12-31'),
        train_end="2021-12-31"
    )

    sentiment_df = [pd.DataFrame()]  # Replace with actual sentiment data
    final_feature_df = technical_indicators_df.merge(
        sentiment_df,
        on=["Date", "ticker"],
        how="left"
    )


if __name__ == "__main__":
    main()