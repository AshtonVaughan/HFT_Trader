"""
Walk-Forward Optimization and Backtesting

Implements walk-forward analysis to prevent overfitting:
- Splits data into rolling train/test windows
- Trains on in-sample data
- Tests on out-of-sample data
- Aggregates results for realistic performance estimation
"""

import yaml
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from data.loaders.dataset import create_dataloaders
from models.regime_detector.transformer_detector import RegimeDetector
from models.predictors.specialized_models import LSTMPredictor, GRUPredictor, CNNLSTMPredictor
from models.meta_learner.attention_meta_learner import AttentionMetaLearner, EnsemblePredictor
from backtesting.backtest_engine import BacktestEngine
from utils.gpu_utils import get_device
from utils.logger import setup_logger, logger


class WalkForwardAnalyzer:
    """
    Walk-forward optimization and analysis.
    """

    def __init__(
        self,
        config: dict,
        train_months: int = 6,
        test_months: int = 1,
        step_months: int = 1
    ):
        """
        Args:
            config: Configuration dict
            train_months: Months of data for training
            test_months: Months of data for testing
            step_months: Step size for rolling window
        """
        self.config = config
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months
        self.device = get_device()

        logger.info(f"Walk-Forward Analyzer initialized")
        logger.info(f"  Train window: {train_months} months")
        logger.info(f"  Test window: {test_months} months")
        logger.info(f"  Step size: {step_months} months")

    def run_walk_forward(self, data: pd.DataFrame) -> Dict:
        """
        Run walk-forward optimization.

        Args:
            data: Full dataset with timestamps

        Returns:
            Dictionary with results
        """
        logger.info("="*80)
        logger.info("WALK-FORWARD OPTIMIZATION")
        logger.info("="*80)

        # Create windows
        windows = self._create_windows(data)
        logger.info(f"\nCreated {len(windows)} walk-forward windows")

        # Results for each window
        all_results = []

        for i, (train_df, test_df) in enumerate(windows):
            logger.info(f"\n{'='*80}")
            logger.info(f"Window {i+1}/{len(windows)}")
            logger.info(f"{'='*80}")
            logger.info(f"Train: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
            logger.info(f"Test: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

            # Train model on in-sample data
            ensemble = self._train_on_window(train_df)

            # Test on out-of-sample data
            results = self._test_on_window(ensemble, test_df)

            results['window'] = i
            results['train_start'] = train_df['timestamp'].min()
            results['train_end'] = train_df['timestamp'].max()
            results['test_start'] = test_df['timestamp'].min()
            results['test_end'] = test_df['timestamp'].max()

            all_results.append(results)

            logger.info(f"\nWindow {i+1} Results:")
            logger.info(f"  Return: {results['total_return']:.2%}")
            logger.info(f"  Win Rate: {results['win_rate']:.2%}")
            logger.info(f"  Sharpe: {results['sharpe_ratio']:.2f}")
            logger.info(f"  Max DD: {results['max_drawdown']:.2%}")

        # Aggregate results
        aggregate_results = self._aggregate_results(all_results)

        logger.info(f"\n{'='*80}")
        logger.info("WALK-FORWARD RESULTS SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"\nAggregate Performance:")
        logger.info(f"  Total Return: {aggregate_results['total_return']:.2%}")
        logger.info(f"  Avg Return per Window: {aggregate_results['avg_return']:.2%}")
        logger.info(f"  Std Return: {aggregate_results['std_return']:.2%}")
        logger.info(f"  Win Rate: {aggregate_results['avg_win_rate']:.2%}")
        logger.info(f"  Avg Sharpe: {aggregate_results['avg_sharpe']:.2f}")
        logger.info(f"  Max Drawdown: {aggregate_results['max_drawdown']:.2%}")
        logger.info(f"  Total Trades: {aggregate_results['total_trades']}")

        # Save results
        self._save_results(all_results, aggregate_results)

        # Plot results
        self._plot_results(all_results)

        return aggregate_results

    def _create_windows(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create rolling train/test windows."""
        if 'timestamp' not in data.columns:
            raise ValueError("Data must have 'timestamp' column")

        data = data.sort_values('timestamp').reset_index(drop=True)

        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])

        windows = []

        start_date = data['timestamp'].min()
        end_date = data['timestamp'].max()

        current_date = start_date

        while True:
            # Train window
            train_start = current_date
            train_end = train_start + pd.DateOffset(months=self.train_months)

            # Test window
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=self.test_months)

            # Check if we have enough data
            if test_end > end_date:
                break

            # Get data for windows
            train_df = data[(data['timestamp'] >= train_start) & (data['timestamp'] < train_end)].copy()
            test_df = data[(data['timestamp'] >= test_start) & (data['timestamp'] < test_end)].copy()

            if len(train_df) > 0 and len(test_df) > 0:
                windows.append((train_df, test_df))

            # Move to next window
            current_date += pd.DateOffset(months=self.step_months)

        return windows

    def _train_on_window(self, train_df: pd.DataFrame) -> EnsemblePredictor:
        """Train model on a single window."""
        logger.info(f"\nTraining on {len(train_df):,} samples...")

        # Split train into train/val
        split_idx = int(len(train_df) * 0.85)
        train_split = train_df.iloc[:split_idx]
        val_split = train_df.iloc[split_idx:]

        # Create dataloaders
        train_loader, val_loader, _ = create_dataloaders(
            train_df=train_split,
            val_df=val_split,
            test_df=val_split,  # Dummy
            batch_size=self.config.get('dataloader', {}).get('batch_size', 256),
            sequence_length=self.config.get('dataloader', {}).get('sequence_length', 1000),
            num_workers=4,
            pin_memory=True
        )

        feature_dim = train_loader.dataset.get_feature_dim()

        # Build models (using config parameters)
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

        specialized_models = [lstm, gru, cnn_lstm]

        meta_learner = AttentionMetaLearner(num_models=3, embedding_dim=128).to(self.device)

        ensemble = EnsemblePredictor(regime_detector, specialized_models, meta_learner).to(self.device)

        # Train (reduced epochs for speed)
        optimizer = torch.optim.AdamW(ensemble.parameters(), lr=1e-4)
        criterion_direction = torch.nn.CrossEntropyLoss()
        criterion_magnitude = torch.nn.MSELoss()

        max_epochs = 5  # Fewer epochs for walk-forward

        for epoch in range(max_epochs):
            ensemble.train()
            total_loss = 0

            for features, targets, regimes, target_signs in train_loader:
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
            logger.info(f"  Epoch {epoch+1}/{max_epochs}: Loss = {avg_loss:.4f}")

        return ensemble

    def _test_on_window(self, ensemble: EnsemblePredictor, test_df: pd.DataFrame) -> Dict:
        """Test model on a single window."""
        logger.info(f"\nTesting on {len(test_df):,} samples...")

        # Generate signals
        ensemble.eval()
        all_signals = []

        # Create dataloader for testing
        _, _, test_loader = create_dataloaders(
            train_df=test_df,  # Dummy
            val_df=test_df,    # Dummy
            test_df=test_df,
            batch_size=256,
            sequence_length=1000,
            num_workers=4,
            pin_memory=True
        )

        with torch.no_grad():
            for features, targets, regimes, target_signs in test_loader:
                features = features.to(self.device)

                outputs = ensemble(features)
                directions = torch.argmax(outputs['direction_logits'], dim=1)
                confidences = outputs['confidence'].squeeze()

                # Convert to signals
                signals = torch.where(confidences > 0.65,
                                    torch.where(directions == 1, torch.tensor(1), torch.tensor(-1)),
                                    torch.tensor(0))

                all_signals.extend(signals.cpu().numpy())

        # Create signals series
        signals_series = pd.Series(all_signals[:len(test_df)], index=test_df.index if hasattr(test_df, 'index') else range(len(test_df)))

        # Backtest
        engine = BacktestEngine(initial_capital=10000)
        results = engine.backtest(test_df, signals_series)

        return results['metrics']

    def _aggregate_results(self, all_results: List[Dict]) -> Dict:
        """Aggregate results across all windows."""
        # Extract metrics
        returns = [r['total_return'] for r in all_results]
        win_rates = [r['win_rate'] for r in all_results]
        sharpes = [r['sharpe_ratio'] for r in all_results]
        drawdowns = [r['max_drawdown'] for r in all_results]
        trades = [r['total_trades'] for r in all_results]

        # Compound returns
        total_return = np.prod([1 + r for r in returns]) - 1

        aggregate = {
            'total_return': total_return,
            'avg_return': np.mean(returns),
            'std_return': np.std(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'avg_win_rate': np.mean(win_rates),
            'avg_sharpe': np.mean(sharpes),
            'max_drawdown': np.max(drawdowns),
            'total_trades': sum(trades),
            'num_windows': len(all_results),
            'profitable_windows': sum([1 for r in returns if r > 0]),
            'win_pct': sum([1 for r in returns if r > 0]) / len(returns)
        }

        return aggregate

    def _save_results(self, all_results: List[Dict], aggregate_results: Dict):
        """Save results to files."""
        output_dir = Path('results/walk_forward')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(output_dir / 'walk_forward_detailed.csv', index=False)

        # Save aggregate results
        with open(output_dir / 'walk_forward_aggregate.yaml', 'w') as f:
            yaml.dump(aggregate_results, f)

        logger.info(f"\nResults saved to {output_dir}/")

    def _plot_results(self, all_results: List[Dict]):
        """Plot walk-forward results."""
        output_dir = Path('results/walk_forward')
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Returns per window
        returns = [r['total_return'] for r in all_results]
        axes[0, 0].bar(range(len(returns)), returns)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_title('Returns per Window')
        axes[0, 0].set_xlabel('Window')
        axes[0, 0].set_ylabel('Return')

        # Cumulative returns
        cumulative = np.cumprod([1 + r for r in returns])
        axes[0, 1].plot(cumulative)
        axes[0, 1].set_title('Cumulative Returns')
        axes[0, 1].set_xlabel('Window')
        axes[0, 1].set_ylabel('Cumulative Return')

        # Win rates
        win_rates = [r['win_rate'] for r in all_results]
        axes[1, 0].plot(win_rates)
        axes[1, 0].axhline(y=0.5, color='r', linestyle='--')
        axes[1, 0].set_title('Win Rate per Window')
        axes[1, 0].set_xlabel('Window')
        axes[1, 0].set_ylabel('Win Rate')

        # Sharpe ratios
        sharpes = [r['sharpe_ratio'] for r in all_results]
        axes[1, 1].plot(sharpes)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_title('Sharpe Ratio per Window')
        axes[1, 1].set_xlabel('Window')
        axes[1, 1].set_ylabel('Sharpe Ratio')

        plt.tight_layout()
        plt.savefig(output_dir / 'walk_forward_analysis.png', dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output_dir}/walk_forward_analysis.png")


def main():
    parser = argparse.ArgumentParser(description="Walk-forward optimization")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file')
    parser.add_argument('--train-months', type=int, default=6, help='Training window (months)')
    parser.add_argument('--test-months', type=int, default=1, help='Testing window (months)')
    parser.add_argument('--step-months', type=int, default=1, help='Step size (months)')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logger
    setup_logger(level='INFO', log_file='walk_forward.log')

    # Load full dataset
    data_dir = Path(config.get('data_collection', {}).get('processed_data_dir', 'processed_data'))
    full_data = pd.read_parquet(data_dir / 'merged_features.parquet')

    # Run walk-forward
    analyzer = WalkForwardAnalyzer(
        config,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months
    )

    results = analyzer.run_walk_forward(full_data)


if __name__ == '__main__':
    main()
