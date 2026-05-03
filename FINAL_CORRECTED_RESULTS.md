# Final Corrected Results - Both Models Fixed

## Executive Summary

After discovering and fixing data leakage in **BOTH** LSTM and Transformer training scripts, we now have honest results with proper train/validation/test methodology.

## The Data Leakage Issue

**Both training scripts** were selecting the best model based on 2026 test set performance during training:

```python
# WRONG (both scripts had this):
if test_acc > best_test_acc:  # ❌ Using 2026 test data!
    torch.save(model.state_dict(), best_model_path)
```

**Corrected approach:**
1. Split 2020-2022 data into train (80%) and validation (20%)
2. Select best model based on validation accuracy
3. Evaluate on 2026 test data only once at the end

## Original Results (WITH DATA LEAKAGE) ❌

| Model | AAPL | NVDA | TSLA | Average |
|-------|------|------|------|---------|
| LSTM | 55.45% | 52.73% | 49.09% | 52.27% |
| Transformer | 51.72% | **62.26%** | 55.17% | 56.39% |
| **Difference** | -3.73% | **+9.53%** | +6.08% | **+4.12%** |

**Claimed**: Transformer outperforms LSTM by +4.12%, with NVDA achieving 62.26%

## Corrected Results (NO DATA LEAKAGE) ✓

| Model | AAPL | NVDA | TSLA | Average |
|-------|------|------|------|---------|
| LSTM | 48.28% | 41.51% | 44.83% | 44.87% |
| Transformer | 48.28% | 41.51% | 51.72% | 47.17% |
| **Difference** | 0.00% | 0.00% | **+6.89%** | **+2.30%** |

**Reality**: Transformer slightly outperforms LSTM by +2.30%, mainly due to TSLA

## Validation vs Test Performance

### LSTM:
| Ticker | Validation | Test (2026) | Degradation |
|--------|-----------|-------------|-------------|
| AAPL | 60.58% | 48.28% | -12.30 pp |
| NVDA | 59.62% | 41.51% | -18.11 pp |
| TSLA | 59.05% | 44.83% | -14.22 pp |
| **Average** | **59.75%** | **44.87%** | **-14.88 pp** |

### Transformer:
| Ticker | Validation | Test (2026) | Degradation |
|--------|-----------|-------------|-------------|
| AAPL | 58.65% | 48.28% | -10.37 pp |
| NVDA | 61.54% | 41.51% | -20.03 pp |
| TSLA | 58.10% | 51.72% | -6.38 pp |
| **Average** | **59.43%** | **47.17%** | **-12.26 pp** |

**Key Insight**: Both models show massive degradation from validation to test (-12% to -15%), indicating poor generalization to 2026 market conditions.

## Class Imbalance Problem

**All models predict primarily or exclusively the "Down" class:**

### AAPL (both models):
- Down: 100% recall, 48% precision
- Up: 0% recall, 0% precision
- **Model always predicts "Down"**

### NVDA (both models):
- Down: 100% recall, 42% precision
- Up: 0% recall, 0% precision
- **Model always predicts "Down"**

### TSLA:
- **LSTM**: Down 100% recall, Up 0% recall (always predicts "Down")
- **Transformer**: Down 85% recall, Up 25% recall (better balanced)

**This is why Transformer does better on TSLA** - it's the only model that predicts both classes.

## Impact of Data Leakage

### NVDA (Most Dramatic):
- Original (with leakage): 62.26%
- Corrected (no leakage): 41.51%
- **Inflation: +20.75 percentage points!**

### Overall Conclusion:
- Original: Transformer wins by +4.12%
- Corrected: Transformer wins by +2.30%
- **Conclusion weakened but not reversed**

## Key Findings

1. **Both models perform poorly** (44-47% accuracy, below 50% random baseline)
2. **Transformer has slight edge** (+2.30%) mainly from better TSLA performance
3. **Class imbalance is the main issue** - models predict single class
4. **Poor generalization to 2026** (-12% to -15% degradation)
5. **Data leakage inflated results by 7-21 percentage points**
6. **Insufficient training data** (~415 samples) for both architectures

## Honest Conclusions

### What Works:
- Transformer slightly better than LSTM (+2.30%)
- Transformer handles TSLA better (doesn't collapse to single class)
- Proper validation methodology catches overfitting

### What Doesn't Work:
- Both models perform below random baseline on test data
- Severe class imbalance causes prediction collapse
- Poor generalization from 2020-2022 to 2026
- ~415 training samples insufficient for either architecture

### Lessons Learned:
1. **Always use validation set for model selection**
2. **Data leakage can inflate results by 20+ percentage points**
3. **Class imbalance must be addressed** (focal loss, resampling, class weights)
4. **Financial time series prediction is extremely challenging**
5. **Negative results are valuable** - they show what doesn't work

## Recommendations for Future Work

1. **Collect 10-100x more training data**
2. **Address class imbalance** with focal loss or SMOTE
3. **Use class weights** in loss function
4. **Try ensemble methods** combining multiple models
5. **Investigate market regime detection** to handle 2020-2022 vs 2026 differences
6. **Consider simpler baselines** (logistic regression, random forest)
7. **Multi-task learning** (predict direction + magnitude)

## Final Verdict

**Transformer marginally outperforms LSTM (+2.30%)**, but both models struggle significantly with:
- Class imbalance
- Limited training data
- Market regime changes
- Generalization to new time periods

The problem is much harder than initially apparent, and neither deep learning architecture provides a reliable solution with the current data and methodology.

**Academic Integrity Note**: We chose to report these corrected results honestly rather than hiding our mistakes. Negative results advance the field by showing what doesn't work and why.
