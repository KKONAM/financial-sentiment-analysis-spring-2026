# Financial Sentiment Analysis Spring 2026

This cloned project has been upgraded with a runnable Python pipeline for:

- collecting financial news from NewsAPI,
- scoring headlines with FinBERT,
- downloading market data from Yahoo Finance,
- building daily financial and sentiment features,
- training LSTM and GRU sequence models.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
cp .env.example .env
```

## Structure

```text
financial-sentiment-analysis-spring-2026/
├── pyproject.toml
├── .env.example
├── src/
│   ├── cli.py
│   ├── config.py
│   ├── pipeline.py
│   ├── data_sources/
│   ├── features/
│   ├── sentiment/
│   └── temporal/
└── notebooks/
```

## Example commands

```bash
financial-sentiment-2026 fetch-news --query "Apple OR Microsoft OR Nvidia" --from-date 2026-04-10 --to-date 2026-04-14
financial-sentiment-2026 convert-news-json --json-path data/raw/news.json
financial-sentiment-2026 build-dataset --ticker AAPL --from-date 2026-04-08 --to-date 2026-04-14
financial-sentiment-2026 train-lstm --dataset data/processed/AAPL_dataset.csv
financial-sentiment-2026 train-gru --dataset data/processed/AAPL_dataset.csv
```

## Notes

- Market features include returns, volatility, RSI, MACD, and Bollinger Bands.
- Sentiment features include FinBERT label probabilities and aggregated daily scores.
- The original repository files were mostly placeholders, so the working implementation added here mirrors the stronger pipeline we built in the parent project.

