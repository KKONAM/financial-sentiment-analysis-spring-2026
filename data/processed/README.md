# Processed Sentiment Data

This directory contains sentiment-analyzed data ready for model training.

## Files

### 1. `stocktwits_daily_sentiment_kaggle_roberta.csv`
**Training Data (2020-2022)**

- **Source**: Kaggle StockTwits dataset
- **Sentiment Model**: RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`)
- **Date Range**: 2020-01-01 to 2022-12-31 (755 trading days)
- **Tickers**: AAPL, AMZN, NVDA, TSLA
- **Size**: ~127 KB

**Columns**:
- `date`: Trading date
- `ticker`: Stock ticker symbol
- `message_count`: Number of StockTwits messages that day
- `sentiment_mean`: Average sentiment score (-1 to +1)
- `sentiment_std`: Standard deviation of sentiment scores
- `sentiment_sum`: Sum of all sentiment scores
- `bullish_count`: Number of bullish messages
- `bearish_count`: Number of bearish messages
- `neutral_count`: Number of neutral messages

**Usage**: Use this for training your LSTM/GRU models (split into train/val 80/20)

### 2. `stocktwits_api_daily_sentiment_2026_full.csv`
**Test Data (2026)**

- **Source**: StockTwits API (collected in 2026)
- **Sentiment Model**: Same RoBERTa model
- **Date Range**: 2026-01-01 to 2026-05-02 (122 trading days)
- **Tickers**: AAPL, AMZN, NVDA, TSLA
- **Size**: ~25 KB

**Columns**: Same as training data

**Usage**: Use this for final out-of-sample testing ONLY (do not use for validation!)

### 3. `combined_sentiment_2026.csv`
**Combined StockTwits + NewsAPI (2026)**

- **Source**: StockTwits API + NewsAPI
- **Sentiment Models**: 
  - StockTwits: RoBERTa
  - NewsAPI: RoBERTa on headlines
- **Date Range**: 2026-01-01 to 2026-05-02
- **Tickers**: AAPL, AMZN, NVDA, TSLA
- **Size**: ~33 KB

**Additional Columns**:
- `news_count`: Number of news articles
- `news_sentiment_mean`: Average news sentiment
- `news_sentiment_std`: Standard deviation of news sentiment
- `combined_sentiment`: Weighted average (60% StockTwits + 40% NewsAPI)

**Note**: NewsAPI coverage is limited (only 39 days have news data)

**Usage**: Optional - for experiments with multi-source sentiment

## Data Processing Pipeline

1. **Raw StockTwits messages** → RoBERTa sentiment scoring → Daily aggregation
2. **Raw news headlines** → RoBERTa sentiment scoring → Daily aggregation
3. **Combine sources** → Weighted average (60/40 split)

## Important Notes for Training

### ⚠️ Proper Train/Val/Test Split

**CRITICAL**: Do NOT use test data for model selection!

**Correct approach**:
```python
# 1. Load training data (2020-2022)
train_df = pd.read_csv('stocktwits_daily_sentiment_kaggle_roberta.csv')

# 2. Split into train (80%) and validation (20%)
from sklearn.model_selection import train_test_split
train_data, val_data = train_test_split(
    train_df, test_size=0.2, random_state=42, shuffle=False  # No shuffle for time series!
)

# 3. Train model and select best based on VALIDATION performance
for epoch in range(epochs):
    train_model(train_data)
    val_acc = evaluate(val_data)
    if val_acc > best_val_acc:
        save_model()  # ✓ Save based on validation

# 4. Load test data ONLY for final evaluation
test_df = pd.read_csv('stocktwits_api_daily_sentiment_2026_full.csv')
final_test_acc = evaluate(test_df)  # ✓ Use test data only once
```

**WRONG approach** (data leakage):
```python
# ❌ DO NOT DO THIS
for epoch in range(epochs):
    train_model(train_data)
    test_acc = evaluate(test_data)  # ❌ Using test data during training!
    if test_acc > best_test_acc:
        save_model()  # ❌ Selecting model based on test performance
```

### Data Statistics

**Training Data (2020-2022)**:
- Total records: ~755 per ticker
- After 80/20 split: ~604 train, ~151 validation
- After sequence creation (5-day windows): ~415 train, ~104 validation

**Test Data (2026)**:
- Total records: ~122 per ticker
- After sequence creation: ~53-58 test samples per ticker

### Features Available

From sentiment data, you can create:
1. **Sentiment features** (14 features):
   - message_count, sentiment_mean, sentiment_std, sentiment_sum
   - bullish_count, bearish_count, neutral_count
   - Rolling averages (3-day, 7-day)
   - Sentiment momentum (change from previous day)

2. **Technical indicators** (17 features):
   - You'll need to merge with price data from Yahoo Finance
   - RSI, MACD, Bollinger Bands, returns, volatility, etc.

3. **Combined** (31 features total):
   - 14 sentiment + 17 technical

## Example Usage

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load training data
train_df = pd.read_csv('data/processed/stocktwits_daily_sentiment_kaggle_roberta.csv')
train_df['date'] = pd.to_datetime(train_df['date'])

# Filter for specific ticker
ticker = 'AAPL'
ticker_df = train_df[train_df['ticker'] == ticker].sort_values('date')

# Create features and target
# (You'll need to add price data and create sequences here)

# Split into train/validation (80/20)
train_data, val_data = train_test_split(
    ticker_df, test_size=0.2, random_state=42, shuffle=False
)

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")

# Train your LSTM/GRU model
# ...

# Load test data for final evaluation
test_df = pd.read_csv('data/processed/stocktwits_api_daily_sentiment_2026_full.csv')
test_df['date'] = pd.to_datetime(test_df['date'])
test_ticker_df = test_df[test_df['ticker'] == ticker]

print(f"Test samples (2026): {len(test_ticker_df)}")
```

## Questions?

If you need help with:
- Merging sentiment data with price data
- Creating sequence windows
- Feature engineering
- Model architecture

Check the training scripts in `scripts/` directory:
- `train_transformer_model.py` - Shows complete pipeline
- `train_lstm_with_combined_sentiment.py` - LSTM example

## Citation

If you use this data, please cite:
- StockTwits dataset: Kaggle StockTwits dataset
- Sentiment model: Cardiff NLP RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`)
- NewsAPI: NewsAPI.org
