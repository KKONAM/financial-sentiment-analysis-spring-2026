from __future__ import annotations

import math

import torch
from torch import nn

from features.build_features import MARKET_FEATURES, SENTIMENT_FEATURES
from .multitask import MultitaskOutput


TRANSFORMER_FEATURES = MARKET_FEATURES + SENTIMENT_FEATURES


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.size(1) > self.encoding.size(1):
            raise ValueError(
                f"Sequence length {inputs.size(1)} exceeds positional encoding limit "
                f"{self.encoding.size(1)}."
            )
        positioned = inputs + self.encoding[:, : inputs.size(1), :]
        return self.dropout(positioned)


class HybridTransformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        classifier_hidden: int = 32,
        head_layers: int = 1,
        transformer_dropout: float = 0.1,
        fc_dropout: float = 0.0,
        activation_name: str = "gelu",
        pooling_name: str = "mean",
        use_layer_norm: bool = True,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.pooling_name = pooling_name
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            dropout=transformer_dropout,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
            activation=activation_name,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )
        pooled_size = d_model * 2 if pooling_name == "last_mean" else d_model
        self.layer_norm = nn.LayerNorm(pooled_size) if use_layer_norm else nn.Identity()
        self.regressor = _build_regressor(
            input_size=pooled_size,
            hidden_size=classifier_hidden,
            head_layers=head_layers,
            activation_name=activation_name,
            dropout=fc_dropout,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.input_projection(inputs)
        encoded = self.positional_encoding(encoded)
        encoded = self.transformer_encoder(encoded)
        pooled = _pool_sequence(encoded, self.pooling_name)
        pooled = self.layer_norm(pooled)
        return self.regressor(pooled).squeeze(-1)


class HybridTransformerMultitask(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        classifier_hidden: int = 32,
        head_layers: int = 1,
        transformer_dropout: float = 0.1,
        fc_dropout: float = 0.0,
        activation_name: str = "gelu",
        pooling_name: str = "mean",
        use_layer_norm: bool = True,
        direction_classes: int = 3,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.pooling_name = pooling_name
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            dropout=transformer_dropout,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
            activation=activation_name,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )
        pooled_size = d_model * 2 if pooling_name == "last_mean" else d_model
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
        encoded = self.input_projection(inputs)
        encoded = self.positional_encoding(encoded)
        encoded = self.transformer_encoder(encoded)
        pooled = _pool_sequence(encoded, self.pooling_name)
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
    layers = list(_hidden_layers(input_size, hidden_size, head_layers, activation_name, dropout))
    layers.append(nn.Linear(hidden_size, 1))
    return nn.Sequential(*layers)


def _build_hidden_head(
    input_size: int,
    hidden_size: int,
    head_layers: int,
    activation_name: str,
    dropout: float,
) -> nn.Sequential:
    return nn.Sequential(*_hidden_layers(input_size, hidden_size, head_layers, activation_name, dropout))


def _hidden_layers(
    input_size: int,
    hidden_size: int,
    head_layers: int,
    activation_name: str,
    dropout: float,
) -> list[nn.Module]:
    layers: list[nn.Module] = []
    current_size = input_size
    for _ in range(head_layers):
        layers.append(nn.Linear(current_size, hidden_size))
        layers.append(_build_activation(activation_name))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        current_size = hidden_size
    return layers


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
