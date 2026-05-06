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
from .multitask import MultitaskOutput


GRU_FEATURES = MARKET_FEATURES + SENTIMENT_FEATURES


class HybridGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        classifier_hidden: int = 16,
        head_layers: int = 1,
        bidirectional: bool = False,
        gru_dropout: float = 0.0,
        fc_dropout: float = 0.0,
        activation_name: str = "relu",
        pooling_name: str = "last",
        use_layer_norm: bool = False,
        gru_bias: bool = True,
    ) -> None:
        super().__init__()
        recurrent_dropout = gru_dropout if num_layers > 1 else 0.0
        self.pooling_name = pooling_name
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
            bias=gru_bias,
        )
        gru_output_size = hidden_size * (2 if bidirectional else 1)
        pooled_size = gru_output_size * 2 if pooling_name == "last_mean" else gru_output_size
        self.layer_norm = nn.LayerNorm(pooled_size) if use_layer_norm else nn.Identity()
        self.classifier = _build_regressor(
            input_size=pooled_size,
            hidden_size=classifier_hidden,
            head_layers=head_layers,
            activation_name=activation_name,
            dropout=fc_dropout,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(inputs)
        pooled = _pool_sequence(output, self.pooling_name)
        pooled = self.layer_norm(pooled)
        return self.classifier(pooled).squeeze(-1)


class HybridGRUMultitask(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        classifier_hidden: int = 16,
        head_layers: int = 1,
        bidirectional: bool = False,
        gru_dropout: float = 0.0,
        fc_dropout: float = 0.0,
        activation_name: str = "relu",
        pooling_name: str = "last",
        use_layer_norm: bool = False,
        gru_bias: bool = True,
        direction_classes: int = 3,
    ) -> None:
        super().__init__()
        recurrent_dropout = gru_dropout if num_layers > 1 else 0.0
        self.pooling_name = pooling_name
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
            bias=gru_bias,
        )
        gru_output_size = hidden_size * (2 if bidirectional else 1)
        pooled_size = gru_output_size * 2 if pooling_name == "last_mean" else gru_output_size
        self.layer_norm = nn.LayerNorm(pooled_size) if use_layer_norm else nn.Identity()
        self.shared_head = _build_hidden_head(
            input_size=pooled_size,
            hidden_size=classifier_hidden,
            head_layers=head_layers,
            activation_name=activation_name,
            dropout=fc_dropout,
        )
        self.return_head = nn.Linear(classifier_hidden, 1)
        self.direction_head = nn.Linear(classifier_hidden, direction_classes)

    def forward(self, inputs: torch.Tensor) -> MultitaskOutput:
        output, _ = self.gru(inputs)
        pooled = _pool_sequence(output, self.pooling_name)
        pooled = self.layer_norm(pooled)
        features = self.shared_head(pooled)
        return MultitaskOutput(
            return_prediction=self.return_head(features).squeeze(-1),
            direction_logits=self.direction_head(features),
        )


def _build_regressor(
    input_size: int,
    hidden_size: int,
    head_layers: int,
    activation_name: str,
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_size = input_size
    for _ in range(head_layers):
        layers.append(nn.Linear(current_size, hidden_size))
        layers.append(_build_activation(activation_name))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        current_size = hidden_size
    layers.append(nn.Linear(current_size, 1))
    return nn.Sequential(*layers)


def _build_hidden_head(
    input_size: int,
    hidden_size: int,
    head_layers: int,
    activation_name: str,
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_size = input_size
    for _ in range(head_layers):
        layers.append(nn.Linear(current_size, hidden_size))
        layers.append(_build_activation(activation_name))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        current_size = hidden_size
    return nn.Sequential(*layers)


def _build_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def _pool_sequence(output: torch.Tensor, pooling_name: str) -> torch.Tensor:
    if pooling_name == "last":
        return output[:, -1, :]
    if pooling_name == "mean":
        return output.mean(dim=1)
    if pooling_name == "max":
        return output.max(dim=1).values
    if pooling_name == "last_mean":
        return torch.cat([output[:, -1, :], output.mean(dim=1)], dim=1)
    raise ValueError(f"Unsupported pooling: {pooling_name}")


@dataclass
class GruTrainingResult:
    accuracy: float
    report: str
    model: HybridGRU


def train_gru_model(
    dataset: pd.DataFrame,
    sequence_length: int = 5,
    train_split: float = 0.8,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    batch_size: int = 16,
) -> GruTrainingResult:
    scaled = dataset.copy()
    scaled[GRU_FEATURES] = StandardScaler().fit_transform(scaled[GRU_FEATURES])
    features, targets = create_sequences(scaled, GRU_FEATURES, "target_direction", sequence_length)
    split_idx = int(len(features) * train_split)

    x_train = torch.tensor(features[:split_idx], dtype=torch.float32)
    y_train = torch.tensor(targets[:split_idx], dtype=torch.float32)
    x_test = torch.tensor(features[split_idx:], dtype=torch.float32)
    y_test = torch.tensor(targets[split_idx:], dtype=torch.float32)

    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=False)
    model = HybridGRU(input_size=len(GRU_FEATURES))
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
    return GruTrainingResult(accuracy=accuracy, report=report, model=model)
