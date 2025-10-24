"""
Meta-Learning Layer

Combines predictions from all specialized models using attention mechanism.
Learns which model to trust in which regime.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logger import logger


class AttentionMetaLearner(nn.Module):
    """
    Attention-based meta-learner for combining multiple model predictions.
    """

    def __init__(
        self,
        num_models: int = 3,  # LSTM, GRU, CNN-LSTM
        embedding_dim: int = 128,
        num_regimes: int = 4,
        dropout: float = 0.2
    ):
        """
        Args:
            num_models: Number of specialized models
            embedding_dim: Embedding dimension for attention
            num_regimes: Number of market regimes
            dropout: Dropout rate
        """
        super().__init__()

        self.num_models = num_models
        self.embedding_dim = embedding_dim

        # Prediction embedding (per model)
        # Input: [direction_logit_0, direction_logit_1, magnitude, confidence] = 4 values per model
        self.pred_embedding = nn.Linear(4, embedding_dim)

        # Regime embedding
        self.regime_embedding = nn.Embedding(num_regimes, embedding_dim)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # Output heads
        self.direction_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # Up/down
        )

        self.magnitude_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Predicted return
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Overall confidence
        )

        logger.info(f"AttentionMetaLearner initialized: {num_models} models → attention → final prediction")

    def forward(
        self,
        model_predictions: List[Dict[str, torch.Tensor]],
        regime: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Combine predictions from multiple models.

        Args:
            model_predictions: List of prediction dicts from each model
                Each dict has keys: direction_logits, magnitude, confidence
            regime: (batch,) regime IDs

        Returns:
            Dict with keys:
                direction_logits: (batch, 2) direction logits
                magnitude: (batch, 1) predicted return
                confidence: (batch, 1) overall confidence
                attention_weights: (batch, num_models) model weights
        """
        batch_size = regime.shape[0]

        # Embed each model's prediction
        pred_embeddings = []
        for pred_dict in model_predictions:
            # Extract tensors from dictionary
            direction_logits = pred_dict['direction_logits']
            magnitude = pred_dict['magnitude']
            confidence = pred_dict['confidence']

            # Concatenate predictions: [dir_0, dir_1, magnitude, confidence]
            pred_vec = torch.cat([
                direction_logits,  # (batch, 2)
                magnitude,  # (batch, 1)
                confidence  # (batch, 1)
            ], dim=1)  # (batch, 4)

            # Embed
            embedded = self.pred_embedding(pred_vec)  # (batch, embedding_dim)
            pred_embeddings.append(embedded)

        # Stack: (batch, num_models, embedding_dim)
        pred_embeddings = torch.stack(pred_embeddings, dim=1)

        # Embed regime
        regime_emb = self.regime_embedding(regime)  # (batch, embedding_dim)
        regime_emb = regime_emb.unsqueeze(1)  # (batch, 1, embedding_dim)

        # Use regime as query, predictions as keys/values
        # Attention: which model to trust given the regime
        attn_output, attn_weights = self.attention(
            query=regime_emb,  # (batch, 1, embedding_dim)
            key=pred_embeddings,  # (batch, num_models, embedding_dim)
            value=pred_embeddings,  # (batch, num_models, embedding_dim)
            need_weights=True
        )

        # attn_output: (batch, 1, embedding_dim)
        # attn_weights: (batch, 1, num_models)

        attn_output = attn_output.squeeze(1)  # (batch, embedding_dim)
        attn_weights = attn_weights.squeeze(1)  # (batch, num_models)

        # Final predictions
        final_direction = self.direction_head(attn_output)
        final_magnitude = self.magnitude_head(attn_output)
        final_confidence = self.confidence_head(attn_output)

        return {
            'direction_logits': final_direction,
            'magnitude': final_magnitude,
            'confidence': final_confidence,
            'attention_weights': attn_weights
        }


class EnsemblePredictor(nn.Module):
    """
    Full ensemble: Regime Detector + Specialized Models + Meta-Learner.
    """

    def __init__(
        self,
        regime_detector,
        specialized_models: List[nn.Module],
        meta_learner
    ):
        super().__init__()

        self.regime_detector = regime_detector
        self.specialized_models = nn.ModuleList(specialized_models)
        self.meta_learner = meta_learner

        logger.info(f"EnsemblePredictor initialized with {len(specialized_models)} specialized models")

    def forward(self, x: torch.Tensor, regime: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through ensemble.

        Args:
            x: (batch, seq_len, input_size)
            regime: (batch,) optional ground-truth regimes (if None, predict from regime_detector)

        Returns:
            Dictionary with all predictions
        """
        # 1. Detect regime (if not provided)
        if regime is None:
            regime_logits, volatility = self.regime_detector(x)
            regime = torch.argmax(regime_logits, dim=1)  # (batch,)
        else:
            regime_logits, volatility = self.regime_detector(x)

        # 2. Get predictions from all specialized models
        model_predictions = []
        for model in self.specialized_models:
            pred_dict = model(x)  # Returns dict
            model_predictions.append(pred_dict)

        # 3. Meta-learning: combine predictions
        meta_output = self.meta_learner(model_predictions, regime)

        return {
            'regime_logits': regime_logits,
            'regime': regime,
            'volatility': volatility,
            'direction_logits': meta_output['direction_logits'],
            'magnitude': meta_output['magnitude'],
            'confidence': meta_output['confidence'],
            'attention_weights': meta_output['attention_weights'],
            'model_predictions': model_predictions
        }


if __name__ == '__main__':
    # Test meta-learner
    from models.predictors.specialized_models import LSTMPredictor, GRUPredictor, CNNLSTMPredictor
    from models.regime_detector.transformer_detector import RegimeDetector

    batch_size = 16
    seq_len = 1000
    input_size = 150

    # Create models
    regime_detector = RegimeDetector(input_size=input_size, d_model=128, nhead=4, num_layers=2)
    lstm = LSTMPredictor(input_size=input_size, hidden_size=128, num_layers=2)
    gru = GRUPredictor(input_size=input_size, hidden_size=128, num_layers=2)
    cnn_lstm = CNNLSTMPredictor(input_size=input_size, cnn_channels=[32, 64], lstm_hidden_size=128)

    specialized_models = [lstm, gru, cnn_lstm]

    meta_learner = AttentionMetaLearner(num_models=3, embedding_dim=128)

    # Create ensemble
    ensemble = EnsemblePredictor(regime_detector, specialized_models, meta_learner)

    # Test forward pass
    x = torch.randn(batch_size, seq_len, input_size)

    outputs = ensemble(x)

    print("\nEnsemble Outputs:")
    print(f"  Regime logits: {outputs['regime_logits'].shape}")
    print(f"  Regime: {outputs['regime'].shape}")
    print(f"  Volatility: {outputs['volatility'].shape}")
    print(f"  Direction logits: {outputs['direction_logits'].shape}")
    print(f"  Magnitude: {outputs['magnitude'].shape}")
    print(f"  Confidence: {outputs['confidence'].shape}")
    print(f"  Attention weights: {outputs['attention_weights'].shape}")

    print(f"\nSample predictions:")
    print(f"  Regime: {outputs['regime'][:5]}")
    print(f"  Direction: {torch.argmax(outputs['direction_logits'], dim=1)[:5]}")
    print(f"  Magnitude: {outputs['magnitude'][:5, 0]}")
    print(f"  Confidence: {outputs['confidence'][:5, 0]}")
    print(f"  Attention weights (LSTM/GRU/CNN-LSTM): {outputs['attention_weights'][:3]}")

    # Count total parameters
    total_params = sum(p.numel() for p in ensemble.parameters())
    print(f"\nTotal ensemble parameters: {total_params:,}")
