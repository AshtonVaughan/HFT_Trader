"""
PyTorch Dataset for Sequence Data

Creates sliding window sequences for time-series prediction.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, List

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logger import logger


class ForexSequenceDataset(Dataset):
    """
    PyTorch Dataset for forex time-series data with sliding windows.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        sequence_length: int = 1000,
        forecast_horizon: int = 1,
        feature_cols: List[str] = None,
        target_col: str = 'return_1',
        regime_col: str = 'regime_id'
    ):
        """
        Args:
            data: DataFrame with features and targets
            sequence_length: Length of input sequence (lookback window)
            forecast_horizon: How many steps ahead to predict
            feature_cols: List of feature column names (if None, auto-detect)
            target_col: Target column for prediction
            regime_col: Regime label column
        """
        self.data = data
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.target_col = target_col
        self.regime_col = regime_col

        # Auto-detect feature columns
        if feature_cols is None:
            exclude_cols = ['timestamp', 'regime', 'regime_id', 'open', 'high', 'low', 'close', 'volume']
            exclude_cols += [target_col]  # Exclude target from features
            self.feature_cols = [col for col in data.columns if col not in exclude_cols and data[col].dtype in [np.float32, np.float64, np.int64]]
        else:
            self.feature_cols = feature_cols

        logger.info(f"ForexSequenceDataset: {len(self.feature_cols)} features, {len(data)} samples")

        # Convert to numpy for faster indexing
        self.features = data[self.feature_cols].values.astype(np.float32)

        # Target (next return)
        if target_col in data.columns:
            self.targets = data[target_col].values.astype(np.float32)
        else:
            # Create target from close prices
            logger.warning(f"Target column '{target_col}' not found, creating from close prices")
            self.targets = data['close'].pct_change(periods=forecast_horizon).shift(-forecast_horizon).values.astype(np.float32)

        # Regime labels
        if regime_col in data.columns:
            self.regimes = data[regime_col].values.astype(np.int64)
        else:
            logger.warning(f"Regime column '{regime_col}' not found, using default regime 0")
            self.regimes = np.zeros(len(data), dtype=np.int64)

        # Valid indices (where we have full sequence + target)
        self.valid_indices = list(range(sequence_length, len(data) - forecast_horizon))

        logger.info(f"   Valid sequences: {len(self.valid_indices):,} (from {len(data):,} total)")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sequence sample.

        Returns:
            features: (sequence_length, num_features) tensor
            target: scalar tensor (next return)
            regime: scalar tensor (regime ID)
            target_sign: scalar tensor (direction: 0=down, 1=up)
        """
        actual_idx = self.valid_indices[idx]

        # Get sequence
        start_idx = actual_idx - self.sequence_length
        end_idx = actual_idx

        features = self.features[start_idx:end_idx]  # (seq_len, num_features)

        # Get target
        target_idx = actual_idx + self.forecast_horizon - 1
        target = self.targets[target_idx]  # scalar

        # Get regime (current regime)
        regime = self.regimes[actual_idx]

        # Target sign (0=negative, 1=positive) for classification
        target_sign = 1 if target > 0 else 0

        # Convert to tensors
        features = torch.from_numpy(features)
        target = torch.tensor(target, dtype=torch.float32)
        regime = torch.tensor(regime, dtype=torch.long)
        target_sign = torch.tensor(target_sign, dtype=torch.long)

        return features, target, regime, target_sign

    def get_feature_dim(self) -> int:
        """Get number of features."""
        return len(self.feature_cols)


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 256,
    sequence_length: int = 1000,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, test DataLoaders.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        batch_size: Batch size
        sequence_length: Sequence length
        num_workers: Number of worker processes
        pin_memory: Pin memory for GPU

    Returns:
        train_loader, val_loader, test_loader
    """
    logger.info("Creating DataLoaders...")

    # Create datasets
    train_dataset = ForexSequenceDataset(train_df, sequence_length=sequence_length)
    val_dataset = ForexSequenceDataset(val_df, sequence_length=sequence_length)
    test_dataset = ForexSequenceDataset(test_df, sequence_length=sequence_length)

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batches
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Fixed: Drop incomplete batches for Transformer-XL memory consistency
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Fixed: Drop incomplete batches for Transformer-XL memory consistency
    )

    logger.info(f"   Train batches: {len(train_loader)}")
    logger.info(f"   Val batches: {len(val_loader)}")
    logger.info(f"   Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test dataset
    dates = pd.date_range('2024-01-01', periods=5000, freq='1T')

    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(5000).cumsum() + 1.1000,
        'high': np.random.randn(5000).cumsum() + 1.1010,
        'low': np.random.randn(5000).cumsum() + 1.0990,
        'close': np.random.randn(5000).cumsum() + 1.1000,
        'volume': np.random.randint(100, 1000, 5000),
        'feature_1': np.random.randn(5000),
        'feature_2': np.random.randn(5000) * 100,
        'feature_3': np.random.randn(5000) * 0.01,
        'return_1': np.random.randn(5000) * 0.001,
        'regime_id': np.random.randint(0, 4, 5000)
    })

    dataset = ForexSequenceDataset(df, sequence_length=100, forecast_horizon=1)

    print(f"\nDataset length: {len(dataset)}")
    print(f"Feature dim: {dataset.get_feature_dim()}")

    # Get a sample
    features, target, regime, target_sign = dataset[0]
    print(f"\nSample shapes:")
    print(f"  Features: {features.shape}")
    print(f"  Target: {target.shape}")
    print(f"  Regime: {regime.shape}")
    print(f"  Target sign: {target_sign.shape}")

    # Test dataloader
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    print(f"\nBatch shapes:")
    print(f"  Features: {batch[0].shape}")  # (batch, seq_len, features)
    print(f"  Targets: {batch[1].shape}")  # (batch,)
    print(f"  Regimes: {batch[2].shape}")  # (batch,)
    print(f"  Target signs: {batch[3].shape}")  # (batch,)
