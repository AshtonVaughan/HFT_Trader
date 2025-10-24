"""
Specialized Prediction Models

LSTM, GRU, and CNN-LSTM models for different market conditions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logger import logger


class LSTMPredictor(nn.Module):
    """
    LSTM model for trend-following predictions.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )

        # Prediction heads
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2)  # Binary: up/down
        )

        self.magnitude_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)  # Regression: predicted return
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()  # Confidence 0-1
        )

        logger.info(f"LSTMPredictor initialized: {input_size} features → {hidden_size}x{num_layers} LSTM")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_size)

        Returns:
            Dict with keys:
                direction_logits: (batch, 2) - up/down classification
                magnitude: (batch, 1) - predicted return
                confidence: (batch, 1) - prediction confidence
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state
        last_hidden = h_n[-1]  # (batch, hidden_size)

        # Predictions
        direction_logits = self.direction_head(last_hidden)
        magnitude = self.magnitude_head(last_hidden)
        confidence = self.confidence_head(last_hidden)

        return {
            'direction_logits': direction_logits,
            'magnitude': magnitude,
            'confidence': confidence
        }


class GRUPredictor(nn.Module):
    """
    GRU model for mean-reversion predictions.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )

        # Prediction heads
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2)
        )

        self.magnitude_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )

        logger.info(f"GRUPredictor initialized: {input_size} features → {hidden_size}x{num_layers} GRU")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # GRU
        gru_out, h_n = self.gru(x)

        # Use last hidden state
        last_hidden = h_n[-1]

        # Predictions
        direction_logits = self.direction_head(last_hidden)
        magnitude = self.magnitude_head(last_hidden)
        confidence = self.confidence_head(last_hidden)

        return {
            'direction_logits': direction_logits,
            'magnitude': magnitude,
            'confidence': confidence
        }


class CNNLSTMPredictor(nn.Module):
    """
    CNN-LSTM hybrid for volatility breakout predictions.

    CNN extracts local patterns, LSTM captures temporal dependencies.
    """

    def __init__(
        self,
        input_size: int,
        cnn_channels: list = [64, 128, 256],
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        self.input_size = input_size

        # 1D CNN for feature extraction
        self.conv_layers = nn.ModuleList()

        in_channels = 1  # Start with 1 channel
        for out_channels in cnn_channels:
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(dropout)
            ))
            in_channels = out_channels

        # Calculate size after convolutions
        self.cnn_output_size = cnn_channels[-1]

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout if lstm_num_layers > 1 else 0,
            batch_first=True
        )

        # Prediction heads
        self.direction_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size // 2, 2)
        )

        self.magnitude_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size // 2, 1)
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size // 4, 1),
            nn.Sigmoid()
        )

        logger.info(f"CNNLSTMPredictor initialized: {input_size} features → CNN → {lstm_hidden_size}x{lstm_num_layers} LSTM")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_size)
        """
        batch_size, seq_len, input_size = x.shape

        # Reshape for CNN: (batch, channels=1, seq_len * input_size)
        # Treat entire sequence as 1D signal
        x = x.reshape(batch_size, 1, seq_len * input_size)

        # Apply CNN layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Reshape for LSTM: (batch, new_seq_len, cnn_output_size)
        x = x.transpose(1, 2)  # (batch, seq_len_reduced, channels)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state
        last_hidden = h_n[-1]

        # Predictions
        direction_logits = self.direction_head(last_hidden)
        magnitude = self.magnitude_head(last_hidden)
        confidence = self.confidence_head(last_hidden)

        return {
            'direction_logits': direction_logits,
            'magnitude': magnitude,
            'confidence': confidence
        }


if __name__ == '__main__':
    # Test models
    batch_size = 32
    seq_len = 1000
    input_size = 150

    print("="*60)
    print("Testing LSTM Predictor")
    print("="*60)
    lstm_model = LSTMPredictor(input_size=input_size, hidden_size=256, num_layers=3)
    x = torch.randn(batch_size, seq_len, input_size)
    outputs = lstm_model(x)
    print(f"Direction logits: {outputs['direction_logits'].shape}")
    print(f"Magnitude: {outputs['magnitude'].shape}")
    print(f"Confidence: {outputs['confidence'].shape}")
    print(f"Parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")

    print("\n" + "="*60)
    print("Testing GRU Predictor")
    print("="*60)
    gru_model = GRUPredictor(input_size=input_size, hidden_size=256, num_layers=3)
    outputs = gru_model(x)
    print(f"Direction logits: {outputs['direction_logits'].shape}")
    print(f"Magnitude: {outputs['magnitude'].shape}")
    print(f"Confidence: {outputs['confidence'].shape}")
    print(f"Parameters: {sum(p.numel() for p in gru_model.parameters()):,}")

    print("\n" + "="*60)
    print("Testing CNN-LSTM Predictor")
    print("="*60)
    cnn_lstm_model = CNNLSTMPredictor(input_size=input_size, cnn_channels=[64, 128], lstm_hidden_size=256)
    outputs = cnn_lstm_model(x)
    print(f"Direction logits: {outputs['direction_logits'].shape}")
    print(f"Magnitude: {outputs['magnitude'].shape}")
    print(f"Confidence: {outputs['confidence'].shape}")
    print(f"Parameters: {sum(p.numel() for p in cnn_lstm_model.parameters()):,}")
