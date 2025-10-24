"""
Transformer-based Regime Detector

Classifies market regimes: trending_up, trending_down, ranging, volatile
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logger import logger


class RegimeDetector(nn.Module):
    """
    Transformer model for regime detection.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.2,
        num_regimes: int = 4
    ):
        """
        Args:
            input_size: Number of input features
            d_model: Transformer embedding dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
            num_regimes: Number of regime classes (4: up, down, ranging, volatile)
        """
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.num_regimes = num_regimes

        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=2000)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, num_regimes)
        )

        # Volatility regression head (auxiliary task)
        self.volatility_regressor = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 4, 1)
        )

        logger.info(f"RegimeDetector initialized: {input_size} features → {d_model} d_model → {num_regimes} regimes")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, input_size)

        Returns:
            regime_logits: (batch, num_regimes)
            volatility: (batch, 1)
        """
        # Project to d_model
        x = self.input_proj(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer(x)  # (batch, seq_len, d_model)

        # Use last timestep for classification
        x_last = x[:, -1, :]  # (batch, d_model)

        # Regime classification
        regime_logits = self.classifier(x_last)  # (batch, num_regimes)

        # Volatility prediction
        volatility = self.volatility_regressor(x_last)  # (batch, 1)

        return regime_logits, volatility


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# Import numpy for positional encoding
import numpy as np


if __name__ == '__main__':
    # Test regime detector
    batch_size = 32
    seq_len = 1000
    input_size = 150

    model = RegimeDetector(input_size=input_size, d_model=256, nhead=8, num_layers=4)

    # Random input
    x = torch.randn(batch_size, seq_len, input_size)

    # Forward pass
    regime_logits, volatility = model(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Regime logits shape: {regime_logits.shape}")
    print(f"Volatility shape: {volatility.shape}")

    # Check regime predictions
    regime_preds = torch.argmax(regime_logits, dim=1)
    print(f"\nRegime predictions: {regime_preds[:10]}")
    print(f"Volatility predictions: {volatility[:10, 0]}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
