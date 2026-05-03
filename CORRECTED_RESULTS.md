# Corrected Results After Fixing Data Leakage

## Summary

After fixing the data leakage issue (using validation set instead of test set for model selection), the results changed **dramatically**:

## Original Results (WITH DATA LEAKAGE) ❌

| Model | AAPL | NVDA | TSLA | Average |
|-------|------|------|------|---------|
| LSTM | 55.45% | 52.73% | 49.09% | 52.27% |
| **Transformer** | 51.72% | **62.26%** | 55.17% | **56.39%** |
| **Improvement** | -3.73% | +9.53% | +6.08% | **+4.12%** |

**Claimed**: Transformer outperforms LSTM by +4.12%

## Corrected Results (NO DATA LEAKAGE) ✓

| Model | AAPL | NVDA | TSLA | Average |
|-------|------|------|------|---------|
| LSTM | 55.45% | 52.73% | 49.09% | 52.27% |
| **Transformer** | 48.28% | 41.51% | 51.72% | **47.17%** |
| **Improvement** | -7.17% | -11.22% | +2.63% | **-5.10%** |

**Reality**: Transformer is WORSE than LSTM by -5.10%

## Key Insights

### 1. Data Leakage Impact
- **NVDA**: Dropped from 62.26% to 41.51% (-20.75 percentage points!)
- **AAPL**: Dropped from 51.72% to 48.28% (-3.44 pp)
- **TSLA**: Dropped from 55.17% to 51.72% (-3.45 pp)

The original "best result" of 62.26% for NVDA was completely due to data leakage.

### 2. Class Imbalance Issues
All models show severe class imbalance:
- **AAPL**: Predicts only "Down" (100% recall for Down, 0% for Up)
- **NVDA**: Predicts only "Down" (100% recall for Down, 0% for Up)
- **TSLA**: Better balanced (85% recall Down, 25% recall Up)

### 3. Validation vs Test Performance

| Ticker | Validation Acc | Test Acc (2026) | Degradation |
|--------|----------------|-----------------|-------------|
| AAPL | 58.65% | 48.28% | -10.37 pp |
| NVDA | 61.54% | 41.51% | -20.03 pp |
| TSLA | 58.10% | 51.72% | -6.38 pp |

Significant degradation from validation to test suggests:
- Market regime change between 2020-2022 and 2026
- Models not generalizing well to new market conditions

### 4. Why Transformer Underperforms

Possible reasons:
1. **Insufficient data**: ~415 training samples may be too few for Transformer
2. **Overfitting**: Transformer has more parameters, easier to overfit
3. **Temporal structure**: LSTM's sequential processing may be better for time series
4. **Hyperparameters**: May need more tuning for Transformer

## Training Details

### Data Split (Corrected):
- **Training**: 2020-2022 data, 80% split (~415 samples)
- **Validation**: 2020-2022 data, 20% split (~104 samples)
- **Test**: 2026 data (53-58 samples depending on ticker)

### Model Selection:
- ✓ Best model selected based on **validation accuracy**
- ✓ Test data only used for final evaluation
- ✓ No data leakage

## Implications for Paper

### What to Report:
1. **Be honest**: Transformer does NOT outperform LSTM
2. **Explain the fix**: Original results had data leakage
3. **Focus on lessons learned**:
   - Importance of proper validation
   - Class imbalance is a major issue
   - Market regime change affects generalization
4. **Emphasize methodology**: Proper train/val/test split is critical

### Revised Key Findings:
1. ~~Transformer outperforms LSTM by +4.12%~~ → **LSTM outperforms Transformer by +5.10%**
2. ~~NVDA achieves 62.26% accuracy~~ → **Best result is AAPL with LSTM at 55.45%**
3. **Class imbalance causes prediction collapse** (new finding)
4. **Significant validation-to-test degradation** (-6% to -20%)
5. **Data leakage can inflate results by 20+ percentage points**

## Academic Integrity

This correction demonstrates:
- ✓ Proper scientific methodology
- ✓ Honest reporting of negative results
- ✓ Learning from mistakes
- ✓ Importance of validation in ML research

**Negative results are still valuable results** - they teach us what doesn't work and why.
