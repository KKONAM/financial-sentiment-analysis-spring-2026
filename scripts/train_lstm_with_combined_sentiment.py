#!/usr/bin/env python3
"""
Train LSTM Models with Combined StockTwits + NewsAPI Sentiment

This script trains per-ticker LSTM models using:
- Technical indicators (price, volume, RSI, MACD, etc.)
- Combined sentiment from StockTwits + NewsAPI
- QQQ market context

The combined sentiment should provide richer signals than StockTwits alone.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from financial_news_sentiment.data_sources.market_data import download_price_history
from financial_news_sentiment.features.builder import build_modeling_dataset
from financial_news_sentiment.models.hybrid import HYBRID_FEATURES, HybridLSTM

# Configuration
TICKERS = ["AAPL", "AMZN", "NVDA", "TSLA"]
TRAIN_START = "2020-01-01"
TRAIN_END = "2022-12-31"
TEST_START = "2026-01-01"
TEST_END = "2026-05-02"
SEQUENCE_LENGTH = 5
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10


def prepare_lstm_data(ticker: str, sentiment_df: pd.DataFrame, 
                     start_date: str, end_date: str, sequence_length: int = 5):
    """Prepare LSTM training data for a single ticker."""
    
    # Download price data
    prices_df = download_price_history(ticker, start_date, end_date)
    
    # Filter sentiment for this ticker
    ticker_sentiment = sentiment_df[sentiment_df['ticker'] == ticker].copy()
    
    # Build modeling dataset
    dataset = build_modeling_dataset(
        price_frame=prices_df,
        daily_sentiment_frame=ticker_sentiment,
        forecast_horizon=1,
    )
    
    if len(dataset) < sequence_length + 1:
        return None, None, None
    
    # Create sequences
    X_sequences = []
    y_labels = []
    
    for i in range(len(dataset) - sequence_length):
        sequence = dataset.iloc[i:i+sequence_length][HYBRID_FEATURES].values
        label = dataset.iloc[i+sequence_length]['target_direction']
        
        X_sequences.append(sequence)
        y_labels.append(label)
    
    X = np.array(X_sequences)
    y = np.array(y_labels)
    
    return X, y, dataset


def train_lstm_model(X_train, y_train, X_test, y_test, ticker: str, output_dir: Path):
    """Train LSTM model for a single ticker with proper train/validation split.
    
    Args:
        X_train: Training data from 2020-2022 (will be split into train/val)
        y_train: Training labels from 2020-2022 (will be split into train/val)
        X_test: Test data from 2026 (only used for final evaluation)
        y_test: Test labels from 2026 (only used for final evaluation)
    """
    
    print(f"\n{'='*60}")
    print(f"Training LSTM for {ticker}")
    print(f"{'='*60}")
    
    # Split training data into train (80%) and validation (20%)
    from sklearn.model_selection import train_test_split
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, shuffle=False  # No shuffle to preserve time order
    )
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_split)
    y_train_t = torch.FloatTensor(y_train_split)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model
    input_size = X_train.shape[2]
    model = HybridLSTM(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )
    
    criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for raw logits
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Model: {input_size} features -> {HIDDEN_SIZE} hidden -> 1 output")
    print(f"Training samples: {len(X_train_split)}, Validation samples: {len(X_val)}, Test samples (2026): {len(X_test)}")
    
    # Training loop with early stopping
    best_val_acc = 0
    patience_counter = 0
    train_losses = []
    val_accs = []
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Evaluate on train and validation
        model.eval()
        with torch.no_grad():
            train_outputs = torch.sigmoid(model(X_train_t))  # Apply sigmoid for probabilities
            train_preds = (train_outputs > 0.5).float()
            train_acc = (train_preds == y_train_t).float().mean().item()
            
            val_outputs = torch.sigmoid(model(X_val_t))  # Apply sigmoid for probabilities
            val_preds = (val_outputs > 0.5).float()
            val_acc = (val_preds == y_val_t).float().mean().item()
            val_accs.append(val_acc)
        
        # Early stopping based on VALIDATION accuracy (not test!)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model based on validation performance
            best_model_path = output_dir / f'hybrid_lstm_combined_sentiment_{ticker.lower()}.pt'
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} - "
                  f"Loss: {avg_train_loss:.4f} - "
                  f"Train Acc: {train_acc:.4f} - "
                  f"Val Acc: {val_acc:.4f} - "
                  f"Best Val: {best_val_acc:.4f}")
        
        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"\n✓ Training complete. Best validation accuracy: {best_val_acc:.4f}")
    
    # Load best model for final evaluation on 2026 test data
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    with torch.no_grad():
        test_outputs = torch.sigmoid(model(X_test_t))  # Apply sigmoid for probabilities
        test_preds = (test_outputs > 0.5).float().numpy()
        test_probs = test_outputs.numpy()
    
    # Calculate test accuracy
    test_acc = (test_preds.flatten() == y_test).mean()
    print(f"\n2026 Out-of-Sample Test Accuracy: {test_acc:.4f}")
    
    # Classification report
    print(f"\nClassification Report (2026 Test Data):")
    print(classification_report(y_test, test_preds, target_names=['Down', 'Up']))
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title(f'{ticker} - Training Loss')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(val_accs)
    ax2.axhline(y=0.5, color='r', linestyle='--', label='Random Baseline')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title(f'{ticker} - Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / f'{ticker.lower()}_training_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return test_acc, test_preds, test_probs, y_test


def main():
    print("="*60)
    print("Train LSTM with Combined Sentiment")
    print("="*60)
    
    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = Path("data/predictions/combined_sentiment_lstm")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load combined sentiment
    sentiment_file = Path("data/processed/combined_sentiment_2026.csv")
    if not sentiment_file.exists():
        print(f"Error: Combined sentiment file not found: {sentiment_file}")
        print("Run: python scripts/combine_stocktwits_newsapi_sentiment.py")
        return
    
    sentiment_df = pd.read_csv(sentiment_file)
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    
    print(f"\n✓ Loaded combined sentiment: {len(sentiment_df)} records")
    print(f"  Date range: {sentiment_df['date'].min().date()} to {sentiment_df['date'].max().date()}")
    print(f"  Tickers: {sorted(sentiment_df['ticker'].unique())}")
    
    # For training, we need historical sentiment (2020-2022)
    # Since we only have 2026 sentiment, we'll use the old StockTwits data for training
    # and test on 2026 with combined sentiment
    
    print(f"\n⚠ Note: Training on 2020-2022 StockTwits data (historical)")
    print(f"  Testing on 2026 combined sentiment data")
    
    train_sentiment_file = Path("data/processed/stocktwits_daily_sentiment_kaggle_roberta.csv")
    if not train_sentiment_file.exists():
        print(f"Error: Training sentiment file not found: {train_sentiment_file}")
        return
    
    train_sentiment_df = pd.read_csv(train_sentiment_file)
    train_sentiment_df['date'] = pd.to_datetime(train_sentiment_df['date'])
    
    # Train models
    all_results = []
    
    for ticker in TICKERS:
        print(f"\n{'='*60}")
        print(f"Processing {ticker}")
        print(f"{'='*60}")
        
        # Prepare training data (2020-2022)
        X_train, y_train, train_dataset = prepare_lstm_data(
            ticker, train_sentiment_df, TRAIN_START, TRAIN_END, SEQUENCE_LENGTH
        )
        
        if X_train is None:
            print(f"⚠ Not enough training data for {ticker}")
            continue
        
        # Prepare test data (2026 with combined sentiment)
        X_test, y_test, test_dataset = prepare_lstm_data(
            ticker, sentiment_df, TEST_START, TEST_END, SEQUENCE_LENGTH
        )
        
        if X_test is None:
            print(f"⚠ Not enough test data for {ticker}")
            continue
        
        # Normalize features
        scaler = StandardScaler()
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        scaler.fit(X_train_flat)
        
        X_train = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        # Train model
        test_acc, test_preds, test_probs, y_test_actual = train_lstm_model(
            X_train, y_train, X_test, y_test, ticker, output_dir
        )
        
        all_results.append({
            'ticker': ticker,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'test_accuracy': test_acc,
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")
    
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))
    
    # Save summary
    summary_path = results_dir / 'training_summary.csv'
    results_df.to_csv(summary_path, index=False)
    
    print(f"\n✓ Saved summary to: {summary_path}")
    print(f"✓ Models saved to: {output_dir}/")
    print(f"✓ Training curves saved to: {output_dir}/")
    
    # Compare to original models
    print(f"\n{'='*60}")
    print("Comparison to Original Models (StockTwits only)")
    print(f"{'='*60}")
    print(f"\nOriginal LSTM (StockTwits only):")
    print(f"  AAPL: 55.45%")
    print(f"  AMZN: 51.82%")
    print(f"  NVDA: 52.73%")
    print(f"  TSLA: 49.09%")
    print(f"  Overall: 52.27%")
    
    print(f"\nNew LSTM (Combined Sentiment):")
    for _, row in results_df.iterrows():
        print(f"  {row['ticker']}: {row['test_accuracy']:.2%}")
    
    avg_acc = results_df['test_accuracy'].mean()
    print(f"  Overall: {avg_acc:.2%}")
    
    improvement = avg_acc - 0.5227
    print(f"\nImprovement: {improvement:+.2%}")


if __name__ == "__main__":
    main()
