"""
Hyperparameter Optimization using Optuna

Optimizes hyperparameters for all models using Bayesian optimization.
"""

import optuna
import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict

from data.loaders.dataset import create_dataloaders
from models.regime_detector.transformer_detector import RegimeDetector
from models.predictors.specialized_models import LSTMPredictor, GRUPredictor, CNNLSTMPredictor
from models.meta_learner.attention_meta_learner import AttentionMetaLearner, EnsemblePredictor
from utils.gpu_utils import get_device
from utils.logger import setup_logger, logger


class HyperparameterOptimizer:
    """
    Optimize hyperparameters using Optuna.
    """

    def __init__(self, config: dict, n_trials: int = 50):
        """
        Args:
            config: Base configuration
            n_trials: Number of optimization trials
        """
        self.config = config
        self.n_trials = n_trials
        self.device = get_device()

        # Load data once
        logger.info("Loading data for optimization...")
        data_dir = Path(config.get('data_collection', {}).get('processed_data_dir', 'processed_data'))
        self.train_df = pd.read_parquet(data_dir / 'train.parquet')
        self.val_df = pd.read_parquet(data_dir / 'val.parquet')

        logger.info(f"Train: {len(self.train_df):,} samples")
        logger.info(f"Val: {len(self.val_df):,} samples")

    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function.

        Args:
            trial: Optuna trial

        Returns:
            Validation loss (to minimize)
        """
        # Suggest hyperparameters
        config = self._suggest_hyperparameters(trial)

        # Create dataloaders
        train_loader, val_loader, _ = create_dataloaders(
            train_df=self.train_df,
            val_df=self.val_df,
            test_df=self.val_df,  # Dummy for now
            batch_size=config['batch_size'],
            sequence_length=config['sequence_length'],
            num_workers=4,
            pin_memory=True
        )

        feature_dim = train_loader.dataset.get_feature_dim()

        # Build models
        regime_detector = RegimeDetector(
            input_size=feature_dim,
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_transformer_layers'],
            dropout=config['dropout']
        ).to(self.device)

        lstm = LSTMPredictor(
            input_size=feature_dim,
            hidden_size=config['lstm_hidden'],
            num_layers=config['lstm_layers']
        ).to(self.device)

        gru = GRUPredictor(
            input_size=feature_dim,
            hidden_size=config['gru_hidden'],
            num_layers=config['gru_layers']
        ).to(self.device)

        cnn_lstm = CNNLSTMPredictor(
            input_size=feature_dim,
            cnn_channels=config['cnn_channels'],
            lstm_hidden_size=config['cnn_lstm_hidden']
        ).to(self.device)

        specialized_models = [lstm, gru, cnn_lstm]

        meta_learner = AttentionMetaLearner(
            num_models=3,
            embedding_dim=config['meta_embedding_dim']
        ).to(self.device)

        ensemble = EnsemblePredictor(regime_detector, specialized_models, meta_learner).to(self.device)

        # Train briefly
        val_loss = self._train_and_evaluate(
            ensemble,
            train_loader,
            val_loader,
            config,
            trial
        )

        return val_loss

    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict:
        """Suggest hyperparameters for trial."""
        config = {}

        # Training hyperparameters
        config['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        config['batch_size'] = trial.suggest_categorical('batch_size', [128, 256, 512])
        config['sequence_length'] = trial.suggest_categorical('sequence_length', [500, 1000, 1500])
        config['dropout'] = trial.suggest_float('dropout', 0.1, 0.5)

        # Regime detector
        config['d_model'] = trial.suggest_categorical('d_model', [128, 256, 512])
        config['nhead'] = trial.suggest_categorical('nhead', [4, 8, 16])
        config['num_transformer_layers'] = trial.suggest_int('num_transformer_layers', 2, 6)

        # LSTM
        config['lstm_hidden'] = trial.suggest_categorical('lstm_hidden', [128, 256, 512])
        config['lstm_layers'] = trial.suggest_int('lstm_layers', 2, 4)

        # GRU
        config['gru_hidden'] = trial.suggest_categorical('gru_hidden', [128, 256, 512])
        config['gru_layers'] = trial.suggest_int('gru_layers', 2, 4)

        # CNN-LSTM
        config['cnn_channels'] = trial.suggest_categorical('cnn_channels',
                                                          [[32, 64, 128], [64, 128, 256], [128, 256, 512]])
        config['cnn_lstm_hidden'] = trial.suggest_categorical('cnn_lstm_hidden', [128, 256, 512])

        # Meta-learner
        config['meta_embedding_dim'] = trial.suggest_categorical('meta_embedding_dim', [64, 128, 256])

        return config

    def _train_and_evaluate(
        self,
        ensemble: EnsemblePredictor,
        train_loader,
        val_loader,
        config: Dict,
        trial: optuna.Trial
    ) -> float:
        """Train and evaluate model."""
        optimizer = torch.optim.AdamW(ensemble.parameters(), lr=config['learning_rate'])
        criterion_direction = nn.CrossEntropyLoss()
        criterion_magnitude = nn.MSELoss()

        # Train for limited epochs (fast evaluation)
        max_epochs = 3

        for epoch in range(max_epochs):
            # Train
            ensemble.train()
            for batch_idx, (features, targets, regimes, target_signs) in enumerate(train_loader):
                if batch_idx >= 20:  # Limit batches per epoch for speed
                    break

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

            # Validate
            ensemble.eval()
            val_losses = []

            with torch.no_grad():
                for batch_idx, (features, targets, regimes, target_signs) in enumerate(val_loader):
                    if batch_idx >= 10:  # Limit validation batches
                        break

                    features = features.to(self.device)
                    targets = targets.to(self.device)
                    target_signs = target_signs.to(self.device)

                    outputs = ensemble(features)

                    loss_dir = criterion_direction(outputs['direction_logits'], target_signs)
                    loss_mag = criterion_magnitude(outputs['magnitude'].squeeze(), targets)
                    loss = loss_dir + 0.5 * loss_mag

                    val_losses.append(loss.item())

            avg_val_loss = np.mean(val_losses)

            # Report intermediate value for pruning
            trial.report(avg_val_loss, epoch)

            # Prune unpromising trials
            if trial.should_prune():
                raise optuna.TrialPruned()

        return avg_val_loss

    def optimize(self) -> Dict:
        """Run optimization."""
        logger.info(f"Starting hyperparameter optimization ({self.n_trials} trials)...")

        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
        )

        study.optimize(self.objective, n_trials=self.n_trials, timeout=None)

        # Best parameters
        best_params = study.best_params
        best_value = study.best_value

        logger.info(f"\nOptimization complete!")
        logger.info(f"Best validation loss: {best_value:.4f}")
        logger.info(f"\nBest hyperparameters:")
        for key, value in best_params.items():
            logger.info(f"  {key}: {value}")

        # Save best config
        optimized_config = self.config.copy()

        # Update config with best params
        optimized_config['dataloader'] = optimized_config.get('dataloader', {})
        optimized_config['dataloader']['batch_size'] = best_params['batch_size']
        optimized_config['dataloader']['sequence_length'] = best_params['sequence_length']

        optimized_config['model'] = {
            'learning_rate': best_params['learning_rate'],
            'dropout': best_params['dropout'],
            'regime_detector': {
                'd_model': best_params['d_model'],
                'nhead': best_params['nhead'],
                'num_layers': best_params['num_transformer_layers']
            },
            'lstm': {
                'hidden_size': best_params['lstm_hidden'],
                'num_layers': best_params['lstm_layers']
            },
            'gru': {
                'hidden_size': best_params['gru_hidden'],
                'num_layers': best_params['gru_layers']
            },
            'cnn_lstm': {
                'cnn_channels': best_params['cnn_channels'],
                'lstm_hidden_size': best_params['cnn_lstm_hidden']
            },
            'meta_learner': {
                'embedding_dim': best_params['meta_embedding_dim']
            }
        }

        # Save to file
        output_file = Path('config/optimized_config.yaml')
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            yaml.dump(optimized_config, f, default_flow_style=False)

        logger.info(f"\nOptimized config saved to {output_file}")

        # Optimization history
        logger.info(f"\nOptimization history:")
        logger.info(f"  Total trials: {len(study.trials)}")
        logger.info(f"  Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        logger.info(f"  Complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")

        return optimized_config


def main():
    parser = argparse.ArgumentParser(description="Optimize HFT_Trader hyperparameters")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Base config file')
    parser.add_argument('--trials', type=int, default=50, help='Number of trials')
    args = parser.parse_args()

    # Load base config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logger
    setup_logger(
        level='INFO',
        log_file='hft_optimize.log'
    )

    # Run optimization
    optimizer = HyperparameterOptimizer(config, n_trials=args.trials)
    optimized_config = optimizer.optimize()


if __name__ == '__main__':
    main()
