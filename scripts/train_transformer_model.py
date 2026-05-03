#!/usr/bin/env python3
"""
Transformer Model for Stock Price Prediction

Uses multi-head self-attention instead of LSTM to capture temporal patterns.

Architecture:
- Positional encoding for time series
- Multi-head self-attention
- Feed-forward layers
- Binary classification (up/down)

Advantages over LSTM:
- Parallel processing (faster training)
- Better long-range dependencies
- Attention weights show which days are important
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
import seaborn as sns
import math

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from financial_news_sentiment.data_sources.market_data import download_price_history
from financial_news_sentiment.features.builder import build_modeling_dataset
from financial_news_sentiment.models.hybrid import HYBRID_FEATURES

# Configuration
TICKERS = ["AAPL", "AMZN", "NVDA", "TSLA"]
TRAIN_START = "2020-01-01"
TRAIN_END = "2022-12-31"
TEST_START = "2026-01-01"
TEST_END = "2026-05-02"
SEQUENCE_LENGTH = 5
D_MODEL = 64  # Embedding dimension
N_HEADS = 4   # Number of attention heads
N_LAYERS = 2  # Number of transformer layers
D_FF = 128    # Feed-forward dimension
DROPOUT = 0.2
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10


class PositionalEncoding(nn.Module):
    """
    Positional encoding for time series.
    Adds position information to the input embeddings.
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Transformer model for stock price prediction.
    
    Architecture:
    1. Input projection: features -> d_model
    2. Positional encoding
    3. Transformer encoder layers
    4. Global average pooling
    5. Classification head
    """
    def __init__(self, input_size: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, d_ff: int = 128, dropout: float = 0.2):
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, input_size)
        Returns:
            Tensor of shape (batch_size,) with logits
        """
        # Project input to d_model
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        # Note: pos_encoder expects (seq_len, batch, d_model)
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)
        
        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Global average pooling over sequence
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Classification
        logits = self.classifier(x).squeeze(-1)  # (batch,)
        
        return logits


def prepare_data(ticker: str, sentiment_df: pd.DataFrame, 
                start_date: str, end_date: str):
    """Prepare data for Transformer."""
    
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
    
    if len(dataset) < SEQUENCE_LENGTH + 1:
        return None, None
    
    # Create sequences
    X_sequences = []
    y_labels = []
    
    for i in range(len(dataset) - SEQUENCE_LENGTH):
        sequence = dataset.iloc[i:i+SEQUENCE_LENGTH][HYBRID_FEATURES].values
        label = dataset.iloc[i+SEQUENCE_LENGTH]['target_direction']
        
        X_sequences.append(sequence)
        y_labels.append(label)
    
    X = np.array(X_sequences)
    y = np.array(y_labels)
    
    return X, y


def train_transformer(X_train, y_train, X_test, y_test, ticker: str, output_dir: Path):
    """Train Transformer model with proper train/validation split.
    
    Args:
        X_train: Training data from 2020-2022 (will be split into train/val)
        y_train: Training labels from 2020-2022 (will be split into train/val)
        X_test: Test data from 2026 (only used for final evaluation)
        y_test: Test labels from 2026 (only used for final evaluation)
    """
    
    print(f"\n{'='*60}")
    print(f"Training Transformer for {ticker}")
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
    model = TransformerModel(
        input_size=input_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
    )
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Model: {input_size} features -> {D_MODEL}d -> {N_HEADS} heads -> {N_LAYERS} layers")
    print(f"Training samples: {len(X_train_split)}, Validation samples: {len(X_val)}, Test samples (2026): {len(X_test)}")
    
    # Training loop
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
            train_outputs = torch.sigmoid(model(X_train_t))
            train_preds = (train_outputs > 0.5).float()
            train_acc = (train_preds == y_train_t).float().mean().item()
            
            val_outputs = torch.sigmoid(model(X_val_t))
            val_preds = (val_outputs > 0.5).float()
            val_acc = (val_preds == y_val_t).float().mean().item()
            val_accs.append(val_acc)
        
        # Early stopping based on VALIDATION accuracy (not test!)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model based on validation performance
            best_model_path = output_dir / f'transformer_{ticker.lower()}.pt'
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
        test_outputs = torch.sigmoid(model(X_test_t))
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
    plot_path = output_dir / f'{ticker.lower()}_transformer_training.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return test_acc, test_preds, test_probs, y_test


def main():
    print("="*60)
    print("Transformer Model for Stock Price Prediction")
    print("="*60)
    
    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = Path("data/predictions/transformer")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load sentiment data
    train_sentiment_file = Path("data/processed/stocktwits_daily_sentiment_kaggle_roberta.csv")
    test_sentiment_file = Path("data/processed/stocktwits_api_daily_sentiment_2026_full.csv")
    
    if not train_sentiment_file.exists() or not test_sentiment_file.exists():
        print("Error: Sentiment files not found")
        return
    
    train_sentiment_df = pd.read_csv(train_sentiment_file)
    train_sentiment_df['date'] = pd.to_datetime(train_sentiment_df['date'])
    
    test_sentiment_df = pd.read_csv(test_sentiment_file)
    test_sentiment_df['date'] = pd.to_datetime(test_sentiment_df['date'])
    
    # Train models
    all_results = []
    
    for ticker in TICKERS:
        print(f"\n{'='*60}")
        print(f"Processing {ticker}")
        print(f"{'='*60}")
        
        # Prepare training data
        X_train, y_train = prepare_data(
            ticker, train_sentiment_df, TRAIN_START, TRAIN_END
        )
        
        if X_train is None:
            print(f"⚠ Not enough training data for {ticker}")
            continue
        
        # Prepare test data
        X_test, y_test = prepare_data(
            ticker, test_sentiment_df, TEST_START, TEST_END
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
        test_acc, test_preds, test_probs, y_test_actual = train_transformer(
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
    summary_path = results_dir / 'transformer_results.csv'
    results_df.to_csv(summary_path, index=False)
    
    print(f"\n✓ Saved summary to: {summary_path}")
    print(f"✓ Models saved to: {output_dir}/transformer_*.pt")
    print(f"✓ Training curves saved to: {output_dir}/*_transformer_training.png")
    
    # Compare to LSTM
    print(f"\n{'='*60}")
    print("Comparison to LSTM")
    print(f"{'='*60}")
    print(f"\nLSTM (2020-2022 training):")
    print(f"  AAPL: 55.45%")
    print(f"  AMZN: 51.82%")
    print(f"  NVDA: 52.73%")
    print(f"  TSLA: 49.09%")
    print(f"  Overall: 52.27%")
    
    print(f"\nTransformer (2020-2022 training):")
    for _, row in results_df.iterrows():
        print(f"  {row['ticker']}: {row['test_accuracy']:.2%}")
    
    avg_acc = results_df['test_accuracy'].mean()
    print(f"  Overall: {avg_acc:.2%}")
    
    improvement = avg_acc - 0.5227
    print(f"\nImprovement over LSTM: {improvement:+.2%}")
    
    if improvement > 0:
        print("✅ Transformer outperforms LSTM!")
    else:
        print("❌ LSTM still better than Transformer")


if __name__ == "__main__":
    main()
