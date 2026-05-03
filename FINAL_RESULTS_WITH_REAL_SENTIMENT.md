# Final Results with Corrected Sentiment Data

## Summary

After fixing **two critical issues**:
1. **Data leakage** - Using test data for model selection
2. **Empty sentiment data** - Previous file had all 0.0 values

We now have honest results with proper validation and actual RoBERTa sentiment scores.

## Final Results (Corrected)

### Test Accuracy (2026 Out-of-Sample)

| Ticker | LSTM | Transformer | Difference |
|--------|------|-------------|------------|
| AAPL | 48.28% | 48.28% | 0.00% |
| AMZN | 41.38% | 41.38% | 0.00% |
| NVDA | 41.51% | 41.51% | 0.00% |
| TSLA | 43.10% | 44.83% | **+1.73%** |
| **Average** | **43.57%** | **44.00%** | **+0.43%** |

### Validation Accuracy (2020-2022)

| Ticker | LSTM | Transformer | Difference |
|--------|------|-------------|------------|
| AAPL | 59.62% | 58.65% | -0.97% |
| AMZN | 53.33% | 52.38% | -0.95% |
| NVDA | 62.50% | 62.50% | 0.00% |
| TSLA | 59.05% | 56.19% | -2.86% |
| **Average** | **58.63%** | **57.43%** | **-1.20%** |

## Key Findings

### 1. Models Perform Nearly Identically
- **AAPL, AMZN, NVDA**: Exactly the same performance (both models predict only "Down" class)
- **TSLA**: Transformer slightly better (+1.73%) - only ticker where models differ
- **Overall**: Transformer has tiny edge (+0.43%)

### 2. Severe Class Imbalance
All models suffer from prediction collapse:

**AAPL, AMZN, NVDA (both models)**:
- Down: 100% recall, ~42-48% precision
- Up: 0% recall, 0% precision
- **Models always predict "Down"**

**TSLA**:
- **LSTM**: Down 88% recall, Up 6% recall (mostly predicts Down)
- **Transformer**: Down 100% recall, Up 0% recall (always predicts Down)

### 3. Poor Generalization to 2026
Validation-to-test degradation:

| Ticker | LSTM Degradation | Transformer Degradation |
|--------|------------------|-------------------------|
| AAPL | -11.34 pp | -10.37 pp |
| AMZN | -11.95 pp | -11.00 pp |
| NVDA | -20.99 pp | -20.99 pp |
| TSLA | -15.95 pp | -11.36 pp |
| **Average** | **-15.06 pp** | **-13.43 pp** |

Both models show massive degradation from validation to test, indicating:
- Market regime change between 2020-2022 and 2026
- Models not learning generalizable patterns
- Overfitting to training period

### 4. Below Random Baseline
- **Random baseline**: 50%
- **LSTM average**: 43.57% ❌
- **Transformer average**: 44.00% ❌

Both models perform **worse than random guessing** on 2026 test data.

## Comparison to Original (Flawed) Results

### Original Results (WITH DATA LEAKAGE + EMPTY SENTIMENT):
- LSTM: 52.27% average
- Transformer: 56.39% average
- Transformer advantage: +4.12%
- NVDA best: 62.26%

### Corrected Results (NO LEAKAGE + REAL SENTIMENT):
- LSTM: 43.57% average
- Transformer: 44.00% average
- Transformer advantage: +0.43%
- Best result: AAPL/Transformer 48.28%

### Impact of Fixes:
- **LSTM dropped**: 52.27% → 43.57% (-8.70 pp)
- **Transformer dropped**: 56.39% → 44.00% (-12.39 pp)
- **NVDA dropped**: 62.26% → 41.51% (-20.75 pp!)
- **Conclusion changed**: From "Transformer wins by 4.12%" to "Transformer barely wins by 0.43%"

## Why Both Models Perform Identically

For AAPL, AMZN, and NVDA, both models:
1. Predict only the "Down" class
2. Achieve accuracy equal to the proportion of "Down" samples in test set
3. Have identical precision/recall/F1 scores

This suggests:
- **Class imbalance in test data** (more "Down" days)
- **Models exploit this imbalance** rather than learning patterns
- **No real predictive power** - just predicting majority class

## Technical Details

### Data Used:
- **Training**: `stocktwits_daily_sentiment_kaggle_roberta.csv` (2020-2022)
  - 3,165 rows with actual RoBERTa sentiment scores
  - Sentiment range: -0.16 to +0.52 (mean: 0.17)
  - Split: 80% train (~415 samples), 20% validation (~104 samples)

- **Test**: `stocktwits_api_daily_sentiment_2026_full.csv` (2026)
  - 122 trading days
  - 53-58 test samples per ticker after sequence creation

### Model Architectures:
- **LSTM**: 2 layers, 128 hidden units, 20% dropout
- **Transformer**: 2 layers, 4 attention heads, 64d model, 10% dropout

### Training:
- Proper train/val/test split (no data leakage)
- Early stopping on validation accuracy
- Test data used only once for final evaluation

## Conclusions

### 1. Neither Model Works Well
- Both perform below random baseline (43-44% vs 50%)
- Severe class imbalance causes prediction collapse
- Poor generalization to 2026 (-13% to -15% degradation)

### 2. Transformer Has Negligible Advantage
- Only +0.43% better than LSTM overall
- Identical performance on 3 out of 4 tickers
- Slightly better on TSLA (+1.73%)

### 3. The Real Problem is Class Imbalance
- Models predict only majority class
- No real pattern learning
- Need class weights, focal loss, or resampling

### 4. Data Leakage Had Massive Impact
- Inflated results by 8-12 percentage points
- Changed conclusions completely
- Demonstrates critical importance of proper validation

### 5. Financial Time Series is Very Hard
- ~415 training samples insufficient
- Market regime changes break models
- Sentiment alone not enough signal

## Recommendations

### Immediate Fixes:
1. **Address class imbalance**:
   - Use class weights in loss function
   - Try focal loss
   - Oversample minority class (SMOTE)

2. **Collect more data**:
   - Need 10-100x more training samples
   - Include more years (2015-2022)
   - Add more tickers

3. **Better features**:
   - Add more technical indicators
   - Include market-wide features (VIX, SPY)
   - Try feature engineering

### Long-term Solutions:
1. **Ensemble methods**: Combine multiple models
2. **Transfer learning**: Pre-train on larger datasets
3. **Market regime detection**: Separate models for different conditions
4. **Multi-task learning**: Predict direction + magnitude
5. **Simpler baselines**: Try logistic regression, random forest first

## Academic Integrity

We report these results honestly despite them being "negative results" because:
1. **Scientific integrity**: Honest reporting advances the field
2. **Learning value**: Shows what doesn't work and why
3. **Reproducibility**: Others can verify and learn from our mistakes
4. **Methodology matters**: Demonstrates importance of proper validation

**Negative results are still valuable results.**

## Files Generated

- `artifacts/transformer_*.pt` - Trained Transformer models
- `artifacts/hybrid_lstm_combined_sentiment_*.pt` - Trained LSTM models
- `artifacts/*_training_curves.png` - Training visualizations
- `data/predictions/transformer/transformer_results.csv` - Transformer results
- `data/predictions/combined_sentiment_lstm/training_summary.csv` - LSTM results

## Next Steps

1. **Address class imbalance** (highest priority)
2. **Collect more training data**
3. **Try simpler baselines** to establish lower bound
4. **Investigate why TSLA is different** (only ticker where models differ)
5. **Analyze 2026 market conditions** to understand degradation

---

**Date**: May 3, 2026  
**Models**: LSTM vs Transformer  
**Methodology**: Proper train/val/test split, no data leakage  
**Data**: Real RoBERTa sentiment scores (not zeros)  
**Conclusion**: Both models struggle, Transformer marginally better (+0.43%)
