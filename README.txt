Financial Sentiment Analysis for Stock Movement Prediction (2026)

Overview
This project explores whether textual sentiment derived from financial news and social media
provides predictive value for stock price movements beyond traditional financial indicators.

We formulate stock prediction as a sequence learning problem, where both market signals and
sentiment evolve over time. Using deep learning, we evaluate whether modern NLP models can extract
meaningful signals from text and improve prediction performance when combined with financial data.


Research Objective
The central question of this study is:
Does sentiment extracted from financial text improve stock movement prediction beyond traditional financial features?

To answer this, we perform a controlled comparison between:
1. Financial-only model
2. Sentiment-only model
3. Combined financial + sentiment model


Methodology
1. Sentiment Modeling
Financial text (news + social media) is processed using a fine-tuned transformer (RoBERTa).

Fine-tuned on:
- Financial PhraseBank
- Stock-related social datasets (e.g., StockTwits)

The model produces:
- sentiment scores (positive / neutral / negative), or
- dense embeddings representing semantic meaning


2. Feature Engineering
For each time step (daily):
Financial Features:
- returns
- trading volume
- technical indicators:
  - RSI
  - MACD
  - Bollinger Bands

Sentiment Features:
- aggregated daily sentiment from:
  - financial news
  - social media posts


3. Temporal Modeling
Sequential dependencies are modeled using:
- LSTM / GRU (primary architecture)
- (optional) Temporal Transformer (extension)

The model learns relationships between:
- historical price behavior
- evolving sentiment signals


4. Experimental Setup
We evaluate three configurations:
Model 1: Random Forest (financial baseline)
Model 2: LSTM/GRU Financial-only
Model 3: LSTM/GRU Sentiment-only
Model 4: LSTM/GRU Multimodal (financial + sentiment)
All models share the same temporal architecture to ensure a fair comparison.

Data Sources
- Financial PhraseBank
- Financial news datasets (Kaggle)
- StockTwits sentiment data
- Historical price data from Yahoo Finance


Target Assets
The study focuses on a small set of high-volume stocks:
- AAPL (Apple)
- TSLA (Tesla)
- Additional tickers to be included
Using multiple assets helps improve generalization and reduces overfitting.

Prediction Task
Binary classification:
1 → price increases (next day)
0 → price decreases (next day)


Evaluation Metrics
- Accuracy
- F1 Score
- Precision / Recall

Optional:
- Backtesting performance (extension)


Expected Contributions
- Empirical evaluation of sentiment vs financial signals
- A reproducible pipeline for multimodal financial prediction
- Insights into the interaction between:
  - market behavior
  - textual sentiment


Tech Stack
- PyTorch
- HuggingFace Transformers
- Pandas / NumPy


Project Structure (Planned)

financial-sentiment-analysis-2026/
│
├── data/
├── model_checkpoints/
├── notebooks/
├── src/
│   ├── sentiment/
│   ├── features/
│   ├── temporal/
│
├── experiments/
└── README.txt


Project Status
Work in progress
- Data collection
- Sentiment model fine-tuning
- Baseline model implementation
- Experimental evaluation


Future Work
- Incorporating Reddit-based sentiment
- Temporal attention mechanisms
- Cross-asset generalization
- Longer prediction horizons


Long-Term Goal
This project is designed as a foundation for a research publication investigating the role of language models in financial prediction.