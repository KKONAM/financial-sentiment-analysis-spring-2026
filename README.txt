Financial Sentiment Analysis for Stock Return Prediction

Current Paper Protocol
The final CS 7643 paper uses a return-only five-trading-day prediction task on
AAPL, AMZN, META, NVDA, and TSLA from 2020-01-01 through 2022-02-28. Models are
trained on pooled ticker sequences with purged chronological train, validation,
and test splits. Directional accuracy is computed only after regression by
thresholding predicted returns.

Final Sentiment Pipeline
The final experiment uses daily StockTwits aggregates from a FinBERT model
fine-tuned on explicit StockTwits Bullish/Bearish labels.

Earlier out-of-box FinBERT scoring was a poor fit for this domain: 80.77% of
row-level predictions were neutral, and neutral was the largest daily aggregate
on every ticker-day. Those outputs are kept only as an unsuccessful iteration.

Fine-tuned FinBERT text-label performance:
- Validation accuracy: 0.8198
- Validation AUC: 0.9051
- Test accuracy: 0.8082
- Test AUC: 0.8968

Final Sequence Model Results

| Model | Feature set | MAE | RMSE | Dir. Acc. | Corr. |
| --- | --- | ---: | ---: | ---: | ---: |
| LSTM | Combined | 0.0480 | 0.0665 | 0.3563 | -0.0127 |
| LSTM | Sentiment only | 0.0479 | 0.0649 | 0.3933 | 0.2091 |
| LSTM | Technical only | 0.0505 | 0.0698 | 0.3950 | -0.0267 |
| GRU | Combined | 0.0476 | 0.0655 | 0.3395 | 0.0838 |
| GRU | Sentiment only | 0.0476 | 0.0649 | 0.3765 | 0.1451 |
| GRU | Technical only | 0.0500 | 0.0681 | 0.3681 | -0.0513 |
| Transformer | Combined | 0.0503 | 0.0684 | 0.3950 | 0.0059 |
| Transformer | Sentiment only | 0.0470 | 0.0647 | 0.3782 | 0.1112 |
| Transformer | Technical only | 0.0497 | 0.0681 | 0.3681 | -0.0703 |

Important caveat:
No neural model beats the always-down directional baseline of 0.5025, and the
zero-return baseline remains difficult to beat on raw MAE/RMSE. The honest
conclusion is that fine-tuned StockTwits sentiment improves return-ranking
signal, especially correlation, but it does not produce a deployable trading
classifier.

Useful Files
- `financial-paper.tex`
- `data/fine_tune_finbert.ipynb`
- `data/processed/stocktwits_finetuned_finbert_daily_sentiment.csv`
- `data/processed/combined_features_AAPL_AMZN_META_NVDA_TSLA_2020-01-01_to_2022-02-28_5d_target_v6_purged_warmtech_finetuned_finbert_sentiment.csv`
- `reports/final/tuned_v6_architecture_feature_grid_summary.csv`
- `scripts/run_v6_tuning_grid.py`

Run Final Tuning Grid
```bash
python scripts/run_v6_tuning_grid.py --trials 60 --max-parallel 3 --torch-threads 2 --force
```

Historical Note
Older binary next-day, RoBERTa, NewsAPI, and 2026-validation experiments remain
useful as development history, but they are not the final paper protocol.
