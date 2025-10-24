"""
Enhanced Training Pipeline with Advanced Features

Includes:
- Learning rate scheduling (cosine annealing, OneCycleLR)
- Gradient clipping and monitoring
- Early stopping
- Model checkpointing
- TensorBoard logging
- Mixed precision training (AMP)
- Gradient accumulation
"""

import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from typing import Dict, Optional
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from data.loaders.dataset import create_dataloaders
from models.regime_detector.transformer_detector import RegimeDetector
from models.predictors.specialized_models import LSTMPredictor, GRUPredictor, CNNLSTMPredictor
from models.predictors.transformer_xl import TransformerXLPredictor
from models.meta_learner.attention_meta_learner import AttentionMetaLearner, EnsemblePredictor
from backtesting.backtest_engine import BacktestEngine
from utils.gpu_utils import get_device, print_gpu_memory
from utils.logger import setup_logger, logger


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class ModelCheckpoint:
    """Model checkpointing handler."""

    def __init__(self, checkpoint_dir: str, mode: str = 'min'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.best_metric = float('inf') if mode == 'min' else float('-inf')

    def save(self, model: nn.Module, metric: float, epoch: int, name: str = 'model'):
        """Save checkpoint if metric improved."""
        is_best = False

        if self.mode == 'min':
            is_best = metric < self.best_metric
        else:
            is_best = metric > self.best_metric

        if is_best:
            self.best_metric = metric

            # Save best model
            checkpoint_path = self.checkpoint_dir / f'{name}_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metric': metric
            }, checkpoint_path)

            logger.info(f"✓ Saved best model: {name}_best.pth (metric: {metric:.4f})")

        # Save latest
        checkpoint_path = self.checkpoint_dir / f'{name}_latest.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metric': metric
        }, checkpoint_path)

        return is_best


class EnhancedTrainer:
    """Enhanced trainer with advanced features."""

    def __init__(self, config: dict):
        self.config = config
        self.device = get_device()

        # Mixed precision training
        self.use_amp = config.get('training', {}).get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None

        # Gradient accumulation
        self.accumulation_steps = config.get('training', {}).get('accumulation_steps', 1)

        # TensorBoard
        self.writer = SummaryWriter(log_dir='runs/training')

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('training', {}).get('early_stopping_patience', 15),
            min_delta=config.get('training', {}).get('early_stopping_delta', 1e-4)
        )

        # Checkpointing
        self.checkpoint = ModelCheckpoint(
            checkpoint_dir='models/checkpoints',
            mode='min'
        )

        logger.info(f"Enhanced Trainer initialized on {self.device}")
        logger.info(f"  Mixed Precision: {self.use_amp}")
        logger.info(f"  Gradient Accumulation Steps: {self.accumulation_steps}")

    def train(self):
        """Run enhanced training pipeline."""
        logger.info("="*80)
        logger.info("ENHANCED TRAINING PIPELINE")
        logger.info("="*80)

        # Load data
        logger.info("\n1. Loading data...")
        train_df, val_df, test_df = self._load_data()

        # Create dataloaders
        logger.info("\n2. Creating dataloaders...")
        train_loader, val_loader, test_loader = create_dataloaders(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            batch_size=self.config.get('dataloader', {}).get('batch_size', 256),
            sequence_length=self.config.get('dataloader', {}).get('sequence_length', 1000),
            num_workers=4,
            pin_memory=True
        )

        # Create smaller batch loaders for memory-intensive models (CNN-LSTM, Transformer-XL)
        train_loader_small, val_loader_small, _ = create_dataloaders(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            batch_size=64,  # Reduced batch size for CNN models
            sequence_length=self.config.get('dataloader', {}).get('sequence_length', 1000),
            num_workers=4,
            pin_memory=True
        )

        feature_dim = train_loader.dataset.get_feature_dim()

        # Build models
        logger.info("\n3. Building models...")
        models_dict = self._build_models(feature_dim)

        # Train each component
        logger.info("\n4. Training regime detector...")
        self._train_with_enhancements(
            model=models_dict['regime_detector'],
            train_loader=train_loader,
            val_loader=val_loader,
            model_name='regime_detector',
            epochs=10,
            lr=1e-4
        )

        logger.info("\n5. Training specialized models...")
        for name, model in zip(['lstm', 'gru', 'cnn_lstm', 'transformer_xl'], models_dict['specialized_models']):
            # Use smaller batch size for memory-intensive models
            if name in ['cnn_lstm', 'transformer_xl']:
                logger.info(f"   Using reduced batch size (64) for {name}")
                self._train_with_enhancements(
                    model=model,
                    train_loader=train_loader_small,
                    val_loader=val_loader_small,
                    model_name=name,
                    epochs=10,
                    lr=1e-4
                )
            else:
                self._train_with_enhancements(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    model_name=name,
                    epochs=10,
                    lr=1e-4
                )

        logger.info("\n6. Training meta-learner...")
        ensemble = models_dict['ensemble']
        self._train_with_enhancements(
            model=ensemble,
            train_loader=train_loader,
            val_loader=val_loader,
            model_name='ensemble',
            epochs=5,
            lr=5e-5
        )

        # Evaluate
        logger.info("\n7. Final evaluation...")
        test_metrics = self._evaluate(ensemble, test_loader)

        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*80)

        for key, value in test_metrics.items():
            logger.info(f"  {key}: {value:.4f}")

        self.writer.close()

    def _train_with_enhancements(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        model_name: str,
        epochs: int,
        lr: float
    ):
        """Train model with all enhancements."""
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=self.config.get('training', {}).get('weight_decay', 0.01)
        )

        # Learning rate scheduler (OneCycleLR)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr * 10,
            steps_per_epoch=len(train_loader) // self.accumulation_steps,
            epochs=epochs,
            pct_start=0.3,
            anneal_strategy='cos'
        )

        # Loss functions
        criterion_direction = nn.CrossEntropyLoss()
        criterion_magnitude = nn.MSELoss()

        # Gradient clipping value
        max_grad_norm = self.config.get('training', {}).get('max_grad_norm', 1.0)

        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            grad_norms = []

            optimizer.zero_grad()

            progress_bar = tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{epochs}")

            for batch_idx, (features, targets, regimes, target_signs) in enumerate(progress_bar):
                features = features.to(self.device)
                targets = targets.to(self.device)
                target_signs = target_signs.to(self.device)
                regimes = regimes.to(self.device)

                # Mixed precision forward pass
                with autocast(enabled=self.use_amp):
                    outputs = model(features)

                    # Handle different output formats
                    if isinstance(outputs, tuple):
                        # RegimeDetector returns (regime_logits, volatility)
                        regime_logits, volatility = outputs
                        loss_regime = criterion_direction(regime_logits, regimes)
                        loss_vol = criterion_magnitude(volatility.squeeze(), targets.abs())
                        loss = (loss_regime + 0.3 * loss_vol) / self.accumulation_steps
                    else:
                        # Predictors return dict with direction_logits and magnitude
                        loss_dir = criterion_direction(outputs['direction_logits'], target_signs)
                        loss_mag = criterion_magnitude(outputs['magnitude'].squeeze(), targets)
                        loss = (loss_dir + 0.5 * loss_mag) / self.accumulation_steps

                # Backward pass with gradient scaling
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    # Clip gradients
                    if self.use_amp:
                        self.scaler.unscale_(optimizer)

                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    grad_norms.append(grad_norm.item())

                    # Optimizer step
                    if self.use_amp:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()

                    optimizer.zero_grad()

                    # Learning rate scheduler step
                    scheduler.step()

                train_loss += loss.item() * self.accumulation_steps

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * self.accumulation_steps:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            val_loss, val_accuracy = self._validate(model, val_loader, criterion_direction, criterion_magnitude)

            # Logging
            logger.info(f"  Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
                       f"Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.2%}, "
                       f"LR={scheduler.get_last_lr()[0]:.2e}, "
                       f"Avg Grad Norm={np.mean(grad_norms):.2f}")

            # TensorBoard
            self.writer.add_scalar(f'{model_name}/train_loss', avg_train_loss, epoch)
            self.writer.add_scalar(f'{model_name}/val_loss', val_loss, epoch)
            self.writer.add_scalar(f'{model_name}/val_accuracy', val_accuracy, epoch)
            self.writer.add_scalar(f'{model_name}/learning_rate', scheduler.get_last_lr()[0], epoch)
            self.writer.add_scalar(f'{model_name}/grad_norm', np.mean(grad_norms), epoch)

            # Checkpointing
            is_best = self.checkpoint.save(model, val_loss, epoch, model_name)

            # Early stopping
            if self.early_stopping(val_loss):
                logger.info(f"  Early stopping triggered for {model_name}")
                break

    def _validate(self, model: nn.Module, val_loader, criterion_direction, criterion_magnitude):
        """Validate model."""
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for features, targets, regimes, target_signs in val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                target_signs = target_signs.to(self.device)
                regimes = regimes.to(self.device)

                outputs = model(features)

                # Handle different output formats
                if isinstance(outputs, tuple):
                    # RegimeDetector returns (regime_logits, volatility)
                    regime_logits, volatility = outputs
                    loss_regime = criterion_direction(regime_logits, regimes)
                    loss_vol = criterion_magnitude(volatility.squeeze(), targets.abs())
                    loss = loss_regime + 0.3 * loss_vol

                    # Accuracy for regime classification
                    preds = torch.argmax(regime_logits, dim=1)
                    correct += (preds == regimes).sum().item()
                else:
                    # Predictors return dict with direction_logits and magnitude
                    loss_dir = criterion_direction(outputs['direction_logits'], target_signs)
                    loss_mag = criterion_magnitude(outputs['magnitude'].squeeze(), targets)
                    loss = loss_dir + 0.5 * loss_mag

                    # Accuracy for direction classification
                    preds = torch.argmax(outputs['direction_logits'], dim=1)
                    correct += (preds == target_signs).sum().item()

                val_loss += loss.item()
                total += target_signs.size(0)

        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total

        return avg_val_loss, accuracy

    def _build_models(self, feature_dim: int) -> Dict:
        """Build all models."""
        regime_detector = RegimeDetector(
            input_size=feature_dim,
            d_model=256,
            nhead=8,
            num_layers=4,
            dropout=0.2
        ).to(self.device)

        lstm = LSTMPredictor(input_size=feature_dim, hidden_size=256, num_layers=3).to(self.device)
        gru = GRUPredictor(input_size=feature_dim, hidden_size=256, num_layers=3).to(self.device)
        cnn_lstm = CNNLSTMPredictor(input_size=feature_dim, cnn_channels=[64, 128, 256], lstm_hidden_size=256).to(self.device)
        transformer_xl = TransformerXLPredictor(input_size=feature_dim, d_model=256, nhead=8, num_layers=4).to(self.device)

        specialized_models = [lstm, gru, cnn_lstm, transformer_xl]

        meta_learner = AttentionMetaLearner(num_models=4, embedding_dim=128).to(self.device)

        ensemble = EnsemblePredictor(regime_detector, specialized_models, meta_learner).to(self.device)

        total_params = sum(p.numel() for p in ensemble.parameters())
        logger.info(f"  Total parameters: {total_params:,}")

        return {
            'regime_detector': regime_detector,
            'specialized_models': specialized_models,
            'meta_learner': meta_learner,
            'ensemble': ensemble
        }

    def _load_data(self):
        """Load preprocessed data."""
        data_dir = Path(self.config.get('data_collection', {}).get('processed_data_dir', 'processed_data'))

        train_df = pd.read_parquet(data_dir / 'train.parquet')
        val_df = pd.read_parquet(data_dir / 'val.parquet')
        test_df = pd.read_parquet(data_dir / 'test.parquet')

        logger.info(f"  Train: {len(train_df):,} samples")
        logger.info(f"  Val: {len(val_df):,} samples")
        logger.info(f"  Test: {len(test_df):,} samples")

        return train_df, val_df, test_df

    def _evaluate(self, ensemble, test_loader):
        """Evaluate ensemble."""
        ensemble.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for features, targets, regimes, target_signs in test_loader:
                features = features.to(self.device)
                target_signs = target_signs.to(self.device)

                outputs = ensemble(features)
                preds = torch.argmax(outputs['direction_logits'], dim=1)

                correct += (preds == target_signs).sum().item()
                total += target_signs.size(0)

        accuracy = correct / total

        return {'accuracy': accuracy}


def main():
    parser = argparse.ArgumentParser(description="Enhanced training")
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Add training config
    config['training'] = {
        'use_amp': True,
        'accumulation_steps': 4,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        'early_stopping_patience': 15,
        'early_stopping_delta': 1e-4
    }

    setup_logger(level='INFO', log_file='train_enhanced.log')

    trainer = EnhancedTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
