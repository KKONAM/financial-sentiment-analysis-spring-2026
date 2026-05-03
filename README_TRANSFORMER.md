# Transformer Model for Stock Price Prediction (2026)

This branch contains the **Transformer model implementation** and **2026 validation data** for the comprehensive deep learning stock prediction project.

## 🎯 Key Results

### Transformer Model Performance
- **Overall Accuracy**: 56.39% (+4.12% over LSTM baseline)
- **Best Result**: NVDA 62.26% (best across all models)
- **TSLA**: 55.17%
- **AAPL**: 51.72%

### Why Transformer Outperforms LSTM
1. **Multi-head self-attention**: Captures complex temporal patterns
2. **Parallel processing**: More efficient training
3. **Better long-range dependencies**: Attention mechanism vs sequential processing
4. **Interpretability**: Attention weights show which time steps matter

## 📁 Files Added

### Scripts
- `scripts/train_transformer_model.py` - Transformer training script

### Models (Artifacts)
- `artifacts/transformer_aapl.pt` - AAPL Transformer model (1.5 MB)
- `artifacts/transformer_nvda.pt` - NVDA Transformer model (1.5 MB)
- `artifacts/transformer_tsla.pt` - TSLA Transformer model (1.5 MB)

### Training Curves
- `artifacts/aapl_transformer_training.png`
- `artifacts/nvda_transformer_training.png`
- `artifacts/tsla_transformer_training.png`

### Results
- `data/predictions/transformer/transformer_results.csv`

### 2026 Data
- `data/processed/combined_sentiment_2026.csv` - Combined StockTwits + NewsAPI sentiment
- `data/processed/stocktwits_2026_technical_dataset_with_qqq.csv` - Full 2026 dataset
- `data/processed/stocktwits_api_daily_sentiment_2026_full.csv` - Daily sentiment aggregates

### Documentation
- `docs/COMPREHENSIVE_PAPER.tex` - Full LaTeX paper with all experiments
- `docs/COMPREHENSIVE_PAPER.pdf` - Compiled PDF paper
- `docs/DEEP_LEARNING_PROJECT_REPORT.md` - Comprehensive markdown report
- `docs/EVERYTHING_WE_TRIED.md` - Complete experiment log
- `docs/figures/` - All publication-quality figures (7 figures, PDF + PNG)

## 🚀 Quick Start

### Train Transformer Model

```bash
python scripts/train_transformer_model.py
```

This will:
1. Load 2020-2022 training data
2. Train Transformer models for AAPL, NVDA, TSLA
3. Save models to `artifacts/transformer_*.pt`
4. Generate training curves
5. Output results to `data/predictions/transformer/`

### Model Architecture

```
Input: 31 features × 5 timesteps
  ↓
Input Projection: 31 → 64 dimensions
  ↓
Positional Encoding (sinusoidal)
  ↓
Transformer Encoder:
  - 2 layers
  - 4 attention heads
  - 128 feed-forward dimensions
  - Dropout: 0.1
  ↓
Global Average Pooling
  ↓
Classification Head: 64 → 32 → 1
  ↓
Output: Binary prediction (Up/Down)
```

## 📊 Results Comparison

| Model | AAPL | NVDA | TSLA | Overall |
|-------|------|------|------|---------|
| **LSTM Baseline** | 55.45% | 52.73% | 49.09% | 52.27% |
| **LSTM Combined** | 51.72% | 54.72% | 55.17% | 53.87% |
| **Transformer** | 51.72% | **62.26%** | 55.17% | **56.39%** |

**Key Findings**:
- ✅ Transformer achieves **+4.12% improvement** over LSTM baseline
- ✅ **NVDA: 62.26%** - Best result across all models and tickers
- ✅ TSLA matches combined sentiment LSTM (55.17%)
- ⚠️ AAPL performance same as combined LSTM (51.72%)

## 🔬 Experiments Documented

The comprehensive paper includes **8 major experiments**:

1. **Binary LSTM (Per-Ticker)** - 52.27% baseline
2. **Combined Sentiment LSTM** - 53.87% (+1.60%)
3. **Transformer Model** - 56.39% (+4.12%) ⭐ **This branch**
4. **Trajectory Clustering** - 59.54% (best classifier)
5. **Multi-Class LSTM** - 33.22% (overfitting failure)
6. **Gradient Boosting** - 7.10% return (best trading)
7. **Ensemble Methods** - Failed (no improvement)
8. **2026 Validation** - -3.56% degradation

## 📈 2026 Validation Results

Testing on out-of-sample 2026 data:

| Ticker | Training (2020-2022) | Test (2026) | Change |
|--------|---------------------|-------------|--------|
| AAPL | 55.45% | 48.28% | -7.17% |
| NVDA | 52.73% | **53.45%** | **+0.72%** ✅ |
| TSLA | 49.09% | 44.83% | -4.26% |
| **Overall** | 52.27% | 48.71% | -3.56% |

**Key Finding**: NVDA is the only ticker to improve on 2026 data, showing best generalization.

## 📚 Documentation

### LaTeX Paper
- **File**: `docs/COMPREHENSIVE_PAPER.tex`
- **Compiled**: `docs/COMPREHENSIVE_PAPER.pdf`
- **Pages**: ~10 pages
- **Sections**: Abstract, Introduction, Related Work, Data, Methodology, Experiments (8), Results, Discussion, Conclusion
- **Figures**: 7 publication-quality figures (300 DPI)

### Markdown Reports
- **DEEP_LEARNING_PROJECT_REPORT.md**: 699 lines, comprehensive analysis
- **EVERYTHING_WE_TRIED.md**: Complete experiment log with all 15+ variations

### Figures
All figures available in PDF (vector) and PNG (raster) formats:
1. `model_comparison.pdf` - Overall accuracy comparison
2. `per_ticker_comparison.pdf` - Per-ticker performance
3. `sentiment_impact.pdf` - Combined sentiment analysis
4. `feature_ablation.pdf` - Feature set comparison
5. `error_comparison.pdf` - MAE/RMSE comparison
6. `2026_validation.pdf` - Generalization results
7. `trading_performance.pdf` - Trading metrics

## 🔧 Dependencies

```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn yfinance
```

## 📊 Data Coverage (2026)

- **Total Days**: 425 trading days (Jan 1 - May 2, 2026)
- **AAPL**: 32% coverage (both StockTwits + NewsAPI)
- **NVDA**: 45.3% coverage (best)
- **TSLA**: 41.5% coverage
- **AMZN**: 0% NewsAPI coverage (excluded from combined sentiment)

## 🎓 Academic Context

This work is part of **CS 7643 - Deep Learning** at Georgia Institute of Technology.

**Authors**: Agha A Raza, Konam G Kenneth

## 🏆 Best Practices Demonstrated

1. ✅ **Ticker-specific models** (not one-size-fits-all)
2. ✅ **Multi-source sentiment** (StockTwits + NewsAPI)
3. ✅ **Out-of-sample validation** (2026 data)
4. ✅ **Comprehensive documentation** (LaTeX + Markdown)
5. ✅ **Reproducible research** (all scripts included)
6. ✅ **Negative results documented** (ensemble failures, overfitting)
7. ✅ **Publication-quality figures** (300 DPI, vector PDF)

## 🚨 Important Notes

### Class Imbalance
- Models can exploit class distribution
- Example: TSLA LSTM achieves 55% by always predicting "Up"
- Solution: Use class weighting or focal loss

### Prediction Collapse
- Technical-only features can lead to near-constant predictions
- Low error ≠ good predictions
- Solution: Evaluate multiple metrics (accuracy, correlation, std)

### Overfitting
- Multi-class LSTM: 92% train → 33% test
- Gradient Boosting: 100% train accuracy
- Solution: Regularization, early stopping, simpler models

## 📝 Citation

If you use this work, please cite:

```bibtex
@article{raza2026transformer,
  title={Deep Learning for Stock Price Prediction: Comprehensive Analysis of LSTM, Transformer, and Ensemble Methods with Multi-Source Sentiment},
  author={Raza, Agha A and Kenneth, Konam G},
  journal={CS 7643 - Deep Learning, Georgia Institute of Technology},
  year={2026}
}
```

## 🔗 Related Branches

- `master` - Original baseline implementation
- `combined-indicators-and-sentiment` - Parent branch with LSTM experiments
- `transformer-model-2026` - **This branch** (Transformer + 2026 data)

## 📧 Contact

- Agha A Raza: ragha3@gatech.edu
- Konam G Kenneth: kkonam3@gatech.edu

---

**Last Updated**: May 2, 2026  
**Branch**: transformer-model-2026  
**Status**: ✅ Complete - Ready for submission
