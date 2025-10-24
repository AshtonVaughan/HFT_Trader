"""
Main Training Pipeline for HFT_Trader

Trains the full ensemble:
1. Regime Detector
2. Specialized Models (LSTM, GRU, CNN-LSTM)
3. Meta-Learner
4. Backtesting evaluation
"""

import yaml
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np

from data.loaders.dataset import create_dataloaders
from models.regime_detector.transformer_detector import RegimeDetector
from models.predictors.specialized_models import LSTMPredictor, GRUPredictor, CNNLSTMPredictor
from models.meta_learner.attention_meta_learner import AttentionMetaLearner, EnsemblePredictor
from backtesting.backtest_engine import BacktestEngine
from utils.gpu_utils import get_device, setup_mixed_precision, print_gpu_memory
from utils.logger import setup_logger, logger


class Trainer:
    """
    Main training orchestrator.
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = get_device()
        self.use_mixed_precision = setup_mixed_precision() and config.get('hardware', {}).get('mixed_precision', True)

        logger.info(f"Trainer initialized on device: {self.device}")
        if self.use_mixed_precision:
            logger.info("Using mixed precision (FP16) training")

    def train(self):
        """Run full training pipeline."""
        logger.info("="*80)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("="*80)

        # 1. Load data
        logger.info("\n1. Loading preprocessed data...")
        train_df, val_df, test_df = self._load_data()

        # 2. Create DataLoaders
        logger.info("\n2. Creating DataLoaders...")
        train_loader, val_loader, test_loader = create_dataloaders(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            batch_size=self.config.get('dataloader', {}).get('batch_size', 256),
            sequence_length=self.config.get('dataloader', {}).get('sequence_length', 1000),
            num_workers=self.config.get('hardware', {}).get('num_workers', 4),
            pin_memory=self.config.get('hardware', {}).get('pin_memory', True)
        )

        feature_dim = train_loader.dataset.get_feature_dim()
        logger.info(f"   Feature dimension: {feature_dim}")

        # 3. Create models
        logger.info("\n3. Building models...")
        regime_detector, specialized_models, meta_learner, ensemble = self._build_models(feature_dim)

        # 4. Train regime detector
        logger.info("\n4. Training Regime Detector...")
        self._train_regime_detector(regime_detector, train_loader, val_loader, epochs=10)

        # 5. Train specialized models
        logger.info("\n5. Training Specialized Models...")
        for i, model in enumerate(specialized_models):
            model_name = ['LSTM', 'GRU', 'CNN-LSTM'][i]
            logger.info(f"\n   Training {model_name}...")
            self._train_predictor(model, train_loader, val_loader, model_name, epochs=10)

        # 6. Train meta-learner
        logger.info("\n6. Training Meta-Learner...")
        self._train_meta_learner(ensemble, train_loader, val_loader, epochs=5)

        # 7. Evaluate on test set
        logger.info("\n7. Evaluating on test set...")
        test_metrics = self._evaluate(ensemble, test_loader)

        # 8. Generate trading signals and backtest
        logger.info("\n8. Generating signals and backtesting...")
        backtest_results = self._backtest(ensemble, test_df, test_loader)

        # 9. Save models
        logger.info("\n9. Saving models...")
        self._save_models(regime_detector, specialized_models, meta_learner)

        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*80)
        logger.info(f"\nTest Metrics:")
        for key, value in test_metrics.items():
            logger.info(f"  {key}: {value:.4f}")

        logger.info(f"\nBacktest Results:")
        for key, value in backtest_results['metrics'].items():
            if isinstance(value, float):
                if 'rate' in key or 'return' in key or 'drawdown' in key:
                    logger.info(f"  {key}: {value:.2%}")
                else:
                    logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")

    def _load_data(self):
        """Load preprocessed datasets."""
        data_dir = Path(self.config.get('data_collection', {}).get('processed_data_dir', 'processed_data'))

        train_df = pd.read_parquet(data_dir / 'train.parquet')
        val_df = pd.read_parquet(data_dir / 'val.parquet')
        test_df = pd.read_parquet(data_dir / 'test.parquet')

        logger.info(f"   Train: {len(train_df):,} samples")
        logger.info(f"   Val: {len(val_df):,} samples")
        logger.info(f"   Test: {len(test_df):,} samples")

        return train_df, val_df, test_df

    def _build_models(self, feature_dim: int):
        """Build all models."""
        # Regime detector
        regime_detector = RegimeDetector(
            input_size=feature_dim,
            d_model=256,
            nhead=8,
            num_layers=4,
            dropout=0.2
        ).to(self.device)

        # Specialized models
        lstm = LSTMPredictor(input_size=feature_dim, hidden_size=256, num_layers=3).to(self.device)
        gru = GRUPredictor(input_size=feature_dim, hidden_size=256, num_layers=3).to(self.device)
        cnn_lstm = CNNLSTMPredictor(input_size=feature_dim, cnn_channels=[64, 128, 256], lstm_hidden_size=256).to(self.device)

        specialized_models = [lstm, gru, cnn_lstm]

        # Meta-learner
        meta_learner = AttentionMetaLearner(num_models=3, embedding_dim=128).to(self.device)

        # Ensemble
        ensemble = EnsemblePredictor(regime_detector, specialized_models, meta_learner).to(self.device)

        # Print parameter counts
        total_params = sum(p.numel() for p in ensemble.parameters())
        logger.info(f"   Total parameters: {total_params:,}")

        return regime_detector, specialized_models, meta_learner, ensemble

    def _train_regime_detector(self, model, train_loader, val_loader, epochs=10):
        """Train regime detector."""
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion_regime = nn.CrossEntropyLoss()
        criterion_volatility = nn.MSELoss()

        for epoch in range(epochs):
            model.train()
            total_loss = 0

            for features, targets, regimes, target_signs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                features = features.to(self.device)
                regimes = regimes.to(self.device)

                optimizer.zero_grad()

                regime_logits, volatility = model(features)

                loss = criterion_regime(regime_logits, regimes)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            logger.info(f"   Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    def _train_predictor(self, model, train_loader, val_loader, model_name, epochs=10):
        """Train a specialized predictor."""
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion_direction = nn.CrossEntropyLoss()
        criterion_magnitude = nn.MSELoss()

        for epoch in range(epochs):
            model.train()
            total_loss = 0

            for features, targets, regimes, target_signs in tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{epochs}"):
                features = features.to(self.device)
                targets = targets.to(self.device)
                target_signs = target_signs.to(self.device)

                optimizer.zero_grad()

                direction_logits, magnitude, confidence = model(features)

                loss_dir = criterion_direction(direction_logits, target_signs)
                loss_mag = criterion_magnitude(magnitude.squeeze(), targets)
                loss = loss_dir + 0.5 * loss_mag

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            logger.info(f"      Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    def _train_meta_learner(self, ensemble, train_loader, val_loader, epochs=5):
        """Train meta-learner (fine-tune ensemble)."""
        optimizer = torch.optim.AdamW(ensemble.parameters(), lr=5e-5)
        criterion_direction = nn.CrossEntropyLoss()
        criterion_magnitude = nn.MSELoss()

        for epoch in range(epochs):
            ensemble.train()
            total_loss = 0

            for features, targets, regimes, target_signs in tqdm(train_loader, desc=f"Meta Epoch {epoch+1}/{epochs}"):
                features = features.to(self.device)
                targets = targets.to(self.device)
                target_signs = target_signs.to(self.device)
                regimes = regimes.to(self.device)

                optimizer.zero_grad()

                outputs = ensemble(features, regime=regimes)

                loss_dir = criterion_direction(outputs['direction_logits'], target_signs)
                loss_mag = criterion_magnitude(outputs['magnitude'].squeeze(), targets)
                loss = loss_dir + 0.5 * loss_mag

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            logger.info(f"   Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    def _evaluate(self, ensemble, test_loader):
        """Evaluate ensemble on test set."""
        ensemble.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for features, targets, regimes, target_signs in test_loader:
                features = features.to(self.device)
                target_signs = target_signs.to(self.device)

                outputs = ensemble(features)
                preds = torch.argmax(outputs['direction_logits'], dim=1)

                total_correct += (preds == target_signs).sum().item()
                total_samples += len(target_signs)

        accuracy = total_correct / total_samples

        return {'accuracy': accuracy}

    def _backtest(self, ensemble, test_df, test_loader):
        """Generate signals and backtest."""
        ensemble.eval()
        all_signals = []

        with torch.no_grad():
            for features, targets, regimes, target_signs in test_loader:
                features = features.to(self.device)

                outputs = ensemble(features)
                directions = torch.argmax(outputs['direction_logits'], dim=1)
                confidences = outputs['confidence'].squeeze()

                # Convert to signals: 1 = buy, -1 = sell, 0 = hold
                signals = torch.where(confidences > 0.65,  # Confidence threshold
                                     torch.where(directions == 1, torch.tensor(1), torch.tensor(-1)),
                                     torch.tensor(0))

                all_signals.extend(signals.cpu().numpy())

        # Create signals series
        signals_series = pd.Series(all_signals[:len(test_df)], index=test_df.index if hasattr(test_df, 'index') else range(len(test_df)))

        # Backtest
        engine = BacktestEngine(initial_capital=10000)
        results = engine.backtest(test_df, signals_series)

        return results

    def _save_models(self, regime_detector, specialized_models, meta_learner):
        """Save trained models."""
        save_dir = Path('models/checkpoints')
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(regime_detector.state_dict(), save_dir / 'regime_detector.pth')
        torch.save(specialized_models[0].state_dict(), save_dir / 'lstm.pth')
        torch.save(specialized_models[1].state_dict(), save_dir / 'gru.pth')
        torch.save(specialized_models[2].state_dict(), save_dir / 'cnn_lstm.pth')
        torch.save(meta_learner.state_dict(), save_dir / 'meta_learner.pth')

        logger.info(f"   Models saved to {save_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Train HFT_Trader models")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logger
    setup_logger(
        level=config.get('logging', {}).get('level', 'INFO'),
        log_file=config.get('logging', {}).get('log_file', 'hft_train.log')
    )

    # Train
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
