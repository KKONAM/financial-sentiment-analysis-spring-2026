# Sentiment Data Ready for Your LSTM/GRU Models

Hey! I've pushed all the sentiment-processed data to the repo so you can use it with your LSTM/GRU models.

## 📦 What's Available

All data is in: `data/processed/`

### 1. Training Data (2020-2022)
**File**: `stocktwits_daily_sentiment_kaggle_roberta.csv`
- 755 trading days of StockTwits sentiment
- Processed with RoBERTa sentiment model
- Tickers: AAPL, AMZN, NVDA, TSLA
- **Use this for training** (split 80/20 into train/val)

### 2. Test Data (2026)
**File**: `stocktwits_api_daily_sentiment_2026_full.csv`
- 122 trading days of fresh 2026 data
- Same RoBERTa processing
- **Use ONLY for final testing** (not for validation!)

### 3. Combined Sentiment (Optional)
**File**: `combined_sentiment_2026.csv`
- StockTwits + NewsAPI combined
- Weighted 60/40 split
- Limited coverage (39 days with news)

## ⚠️ CRITICAL: Avoid Data Leakage!

I discovered both our training scripts had data leakage - we were using test data to select the best model during training. This inflated results by up to 20 percentage points!

**Correct approach**:
```python
# 1. Split 2020-2022 data into train (80%) and validation (20%)
train_data, val_data = train_test_split(
    df_2020_2022, test_size=0.2, random_state=42, shuffle=False
)

# 2. Train and select best model based on VALIDATION
for epoch in range(epochs):
    train(train_data)
    val_acc = evaluate(val_data)
    if val_acc > best_val_acc:  # ✓ Use validation!
        save_model()

# 3. Test on 2026 data ONLY ONCE at the end
test_acc = evaluate(test_data_2026)  # ✓ Final evaluation only
```

**WRONG** (what we were doing):
```python
# ❌ DO NOT DO THIS
for epoch in range(epochs):
    train(train_data)
    test_acc = evaluate(test_data_2026)  # ❌ Using test data during training!
    if test_acc > best_test_acc:
        save_model()  # ❌ Selecting based on test performance
```

## 📊 Data Format

Each CSV has these columns:
- `date`: Trading date
- `ticker`: Stock symbol
- `message_count`: Number of messages
- `sentiment_mean`: Average sentiment (-1 to +1)
- `sentiment_std`: Standard deviation
- `sentiment_sum`: Sum of sentiments
- `bullish_count`, `bearish_count`, `neutral_count`

## 🚀 Quick Start

```python
import pandas as pd

# Load training data
train_df = pd.read_csv('data/processed/stocktwits_daily_sentiment_kaggle_roberta.csv')
train_df['date'] = pd.to_datetime(train_df['date'])

# Filter for your ticker
ticker_df = train_df[train_df['ticker'] == 'AAPL']

# Split train/val (80/20)
from sklearn.model_selection import train_test_split
train, val = train_test_split(
    ticker_df, test_size=0.2, random_state=42, shuffle=False
)

print(f"Train: {len(train)} days")  # ~604 days
print(f"Val: {len(val)} days")      # ~151 days

# After creating sequences (5-day windows):
# Train: ~415 samples
# Val: ~104 samples
```

## 📈 Expected Results (After Fixing Data Leakage)

Our corrected results:
- **LSTM**: 44.87% average (AAPL: 48.28%, NVDA: 41.51%, TSLA: 44.83%)
- **Transformer**: 47.17% average (AAPL: 48.28%, NVDA: 41.51%, TSLA: 51.72%)

Both models struggle with:
- Class imbalance (predicting mostly one class)
- Poor generalization to 2026 (-12% to -15% degradation)
- Limited training data (~415 samples)

## 💡 Tips

1. **Address class imbalance**: Use class weights, focal loss, or SMOTE
2. **Feature engineering**: Combine sentiment with technical indicators (RSI, MACD, etc.)
3. **Sequence length**: We used 5-day windows, you might try 3, 7, or 10
4. **Regularization**: Dropout, early stopping on validation loss
5. **Try GRU**: Often performs similarly to LSTM with fewer parameters

## 📚 Documentation

- Full README: `data/processed/README.md`
- Example training script: `scripts/train_transformer_model.py`
- Data leakage fix details: `DATA_LEAKAGE_FIX.md`
- Corrected results: `FINAL_CORRECTED_RESULTS.md`

## 🔗 GitHub

Everything is pushed to:
https://github.com/KKONAM/financial-sentiment-analysis-spring-2026/tree/transformer-model-2026

Branch: `transformer-model-2026`

## Questions?

Let me know if you need:
- Help merging sentiment with price data
- Feature engineering ideas
- Model architecture suggestions
- Debugging data leakage in your code

Good luck with your LSTM/GRU experiments! 🚀
