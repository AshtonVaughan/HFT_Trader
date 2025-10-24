"""
Tests for model architectures.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.regime_detector.transformer_detector import RegimeDetector
from models.predictors.specialized_models import LSTMPredictor, GRUPredictor, CNNLSTMPredictor
from models.predictors.transformer_xl import TransformerXLPredictor
from models.meta_learner.attention_meta_learner import AttentionMetaLearner, EnsemblePredictor


class TestRegimeDetector:
    """Test regime detector model."""

    def test_initialization(self):
        """Test model initialization."""
        model = RegimeDetector(input_size=50, d_model=128, nhead=4, num_layers=2)
        assert model is not None

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0

    def test_forward_pass(self):
        """Test forward pass."""
        model = RegimeDetector(input_size=50, d_model=128, nhead=4, num_layers=2)
        model.eval()

        # Create dummy input
        x = torch.randn(2, 100, 50)  # (batch, seq_len, features)

        with torch.no_grad():
            regime_logits, volatility = model(x)

        # Check output shapes
        assert regime_logits.shape == (2, 4)  # 4 regime classes
        assert volatility.shape == (2, 1)

    def test_training_mode(self):
        """Test training mode."""
        model = RegimeDetector(input_size=50, d_model=128, nhead=4, num_layers=2)
        model.train()

        x = torch.randn(2, 100, 50)
        regime_logits, volatility = model(x)

        # Test backward pass
        loss = regime_logits.sum() + volatility.sum()
        loss.backward()

        # Check gradients
        for param in model.parameters():
            assert param.grad is not None


class TestSpecializedModels:
    """Test specialized predictor models."""

    @pytest.mark.parametrize("ModelClass", [LSTMPredictor, GRUPredictor, CNNLSTMPredictor])
    def test_model_forward(self, ModelClass):
        """Test forward pass for all specialized models."""
        if ModelClass == CNNLSTMPredictor:
            model = ModelClass(input_size=50, cnn_channels=[32, 64], lstm_hidden_size=128)
        else:
            model = ModelClass(input_size=50, hidden_size=128, num_layers=2)

        model.eval()

        x = torch.randn(2, 100, 50)

        with torch.no_grad():
            outputs = model(x)

        # Check outputs
        assert 'direction_logits' in outputs
        assert 'magnitude' in outputs
        assert 'confidence' in outputs

        assert outputs['direction_logits'].shape == (2, 2)
        assert outputs['magnitude'].shape == (2, 1)
        assert outputs['confidence'].shape == (2, 1)

    def test_lstm_training(self):
        """Test LSTM training."""
        model = LSTMPredictor(input_size=50, hidden_size=128, num_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()
        x = torch.randn(4, 50, 50)
        outputs = model(x)

        # Dummy loss (use all outputs to ensure all parameters get gradients)
        loss = outputs['direction_logits'].sum() + outputs['magnitude'].sum() + outputs['confidence'].sum()
        loss.backward()
        optimizer.step()

        # Check that gradients were computed
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestTransformerXL:
    """Test Transformer-XL model."""

    def test_initialization(self):
        """Test initialization."""
        model = TransformerXLPredictor(
            input_size=50,
            d_model=128,
            nhead=4,
            num_layers=2,
            segment_len=100,
            mem_len=100
        )
        assert model is not None

    def test_forward_with_memory(self):
        """Test forward pass with memory."""
        model = TransformerXLPredictor(
            input_size=50,
            d_model=128,
            nhead=4,
            num_layers=2,
            segment_len=100,
            mem_len=100
        )
        model.eval()

        # Process multiple segments
        model.reset_memory()

        segment1 = torch.randn(1, 100, 50)
        segment2 = torch.randn(1, 100, 50)

        with torch.no_grad():
            out1 = model(segment1)
            out2 = model(segment2)  # Should use memory from segment1

        assert 'direction_logits' in out1
        assert 'direction_logits' in out2


class TestMetaLearner:
    """Test meta-learner and ensemble."""

    def test_attention_meta_learner(self):
        """Test attention meta-learner."""
        meta_learner = AttentionMetaLearner(num_models=3, embedding_dim=128)
        meta_learner.eval()

        # Dummy predictions from 3 models
        predictions = [
            {
                'direction_logits': torch.randn(2, 2),
                'magnitude': torch.randn(2, 1),
                'confidence': torch.sigmoid(torch.randn(2, 1))
            }
            for _ in range(3)
        ]

        regime = torch.tensor([0, 1])  # Dummy regime

        with torch.no_grad():
            output = meta_learner(predictions, regime)

        assert 'direction_logits' in output
        assert 'magnitude' in output
        assert 'confidence' in output

    def test_ensemble_predictor(self):
        """Test full ensemble."""
        regime_detector = RegimeDetector(input_size=50, d_model=128, nhead=4, num_layers=2)

        specialized_models = [
            LSTMPredictor(input_size=50, hidden_size=128, num_layers=2),
            GRUPredictor(input_size=50, hidden_size=128, num_layers=2),
            CNNLSTMPredictor(input_size=50, cnn_channels=[32, 64], lstm_hidden_size=128)
        ]

        meta_learner = AttentionMetaLearner(num_models=3, embedding_dim=128)

        ensemble = EnsemblePredictor(regime_detector, specialized_models, meta_learner)
        ensemble.eval()

        x = torch.randn(2, 100, 50)

        with torch.no_grad():
            outputs = ensemble(x)

        assert 'direction_logits' in outputs
        assert 'magnitude' in outputs
        assert 'confidence' in outputs
        assert 'regime_logits' in outputs


class TestModelDevices:
    """Test model device placement."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_placement(self):
        """Test CUDA placement."""
        model = LSTMPredictor(input_size=50, hidden_size=128, num_layers=2).cuda()

        x = torch.randn(2, 100, 50).cuda()

        with torch.no_grad():
            outputs = model(x)

        assert outputs['direction_logits'].is_cuda

    def test_cpu_placement(self):
        """Test CPU placement."""
        model = LSTMPredictor(input_size=50, hidden_size=128, num_layers=2).cpu()

        x = torch.randn(2, 100, 50).cpu()

        with torch.no_grad():
            outputs = model(x)

        assert not outputs['direction_logits'].is_cuda


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
