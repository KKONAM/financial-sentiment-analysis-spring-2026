from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from features.build_features import MARKET_FEATURES, SENTIMENT_FEATURES, create_sequences


LSTM_FEATURES = MARKET_FEATURES + SENTIMENT_FEATURES


class HybridLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        classifier_hidden: int = 16,
        bidirectional: bool = False,
        lstm_dropout: float = 0.0,
        fc_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        recurrent_dropout = lstm_dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
        )
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(classifier_hidden, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(inputs)
        return self.classifier(output[:, -1, :]).squeeze(-1)


@dataclass
class LstmTrainingResult:
    accuracy: float
    report: str
    model: HybridLSTM


def train_lstm_model(
    dataset: pd.DataFrame,
    sequence_length: int = 5,
    train_split: float = 0.8,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    batch_size: int = 16,
) -> LstmTrainingResult:
    scaled = dataset.copy()
    scaled[LSTM_FEATURES] = StandardScaler().fit_transform(scaled[LSTM_FEATURES])
    features, targets = create_sequences(scaled, LSTM_FEATURES, "target_direction", sequence_length)
    split_idx = int(len(features) * train_split)

    x_train = torch.tensor(features[:split_idx], dtype=torch.float32)
    y_train = torch.tensor(targets[:split_idx], dtype=torch.float32)
    x_test = torch.tensor(features[split_idx:], dtype=torch.float32)
    y_test = torch.tensor(targets[split_idx:], dtype=torch.float32)

    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=False)
    model = HybridLSTM(input_size=len(LSTM_FEATURES))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        for batch_features, batch_targets in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_features), batch_targets)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(x_test)).cpu().numpy()
    predictions = (probabilities >= 0.5).astype(np.int64)
    accuracy = accuracy_score(y_test.cpu().numpy(), predictions)
    report = classification_report(y_test.cpu().numpy(), predictions, digits=4)
    return LstmTrainingResult(accuracy=accuracy, report=report, model=model)
