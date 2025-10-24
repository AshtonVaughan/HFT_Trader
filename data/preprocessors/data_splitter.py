"""
Data Splitter with Regime Labeling

Splits data into train/val/test and labels market regimes for supervised learning.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from sklearn.preprocessing import RobustScaler
import joblib

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logger import logger


class DataSplitter:
    """
    Split data and label regimes.
    """

    REGIME_CLASSES = ['trending_up', 'trending_down', 'ranging', 'volatile']

    def __init__(self, train_pct=0.70, val_pct=0.15, test_pct=0.15):
        """
        Args:
            train_pct: Fraction for training
            val_pct: Fraction for validation
            test_pct: Fraction for testing
        """
        assert abs(train_pct + val_pct + test_pct - 1.0) < 0.01, "Splits must sum to 1.0"

        self.train_pct = train_pct
        self.val_pct = val_pct
        self.test_pct = test_pct

        self.scaler = None

        logger.info(f"DataSplitter initialized: {train_pct:.0%} train / {val_pct:.0%} val / {test_pct:.0%} test")

    def split_and_label(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split data chronologically and add regime labels.

        Args:
            df: Full feature DataFrame

        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        logger.info("="*80)
        logger.info("SPLITTING AND LABELING DATA")
        logger.info("="*80)

        df = df.copy()

        # 1. Label regimes
        logger.info("\n1. Labeling market regimes...")
        df = self._label_regimes(df)

        # 2. Split chronologically
        logger.info("\n2. Splitting data chronologically...")
        splits = self._split_data(df)

        # 3. Scale features
        logger.info("\n3. Scaling features...")
        splits = self._scale_features(splits)

        # 4. Print stats
        self._print_split_stats(splits)

        logger.info("\n" + "="*80)
        logger.info("DATA SPLITTING COMPLETE")
        logger.info("="*80)

        return splits

    def _label_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label each bar with market regime.

        Regimes:
        - trending_up: ADX > 25 and price > EMA200
        - trending_down: ADX > 25 and price < EMA200
        - ranging: ADX < 20
        - volatile: ATR in top 20th percentile
        """
        df = df.copy()

        # Ensure required features exist
        if 'adx' not in df.columns or 'ema_200' not in df.columns or 'atr' not in df.columns:
            logger.warning("Missing required features for regime labeling. Using default regime.")
            df['regime'] = 'ranging'
            df['regime_id'] = 2
            return df

        # Calculate ATR percentile threshold
        atr_threshold = df['atr'].quantile(0.80)

        def classify_regime(row):
            # Volatile takes priority
            if row['atr'] >= atr_threshold:
                return 'volatile', 3

            # Trending
            if row['adx'] > 25:
                if row['close'] > row['ema_200']:
                    return 'trending_up', 0
                else:
                    return 'trending_down', 1

            # Ranging (default)
            return 'ranging', 2

        df[['regime', 'regime_id']] = df.apply(
            lambda row: pd.Series(classify_regime(row)),
            axis=1
        )

        # Print regime distribution
        regime_counts = df['regime'].value_counts()
        logger.info(f"   Regime distribution:")
        for regime, count in regime_counts.items():
            pct = count / len(df) * 100
            logger.info(f"     {regime}: {count} ({pct:.1f}%)")

        return df

    def _split_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split data chronologically."""
        n = len(df)
        train_end = int(n * self.train_pct)
        val_end = int(n * (self.train_pct + self.val_pct))

        splits = {
            'train': df.iloc[:train_end].copy(),
            'val': df.iloc[train_end:val_end].copy(),
            'test': df.iloc[val_end:].copy()
        }

        for split_name, split_df in splits.items():
            logger.info(f"   {split_name}: {len(split_df):,} samples ({len(split_df)/n:.1%})")
            logger.info(f"      Date range: {split_df['timestamp'].min()} to {split_df['timestamp'].max()}")

        return splits

    def _scale_features(self, splits: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Scale features using RobustScaler (handles outliers better).

        Fit on train set only, transform all sets.
        """
        # Identify feature columns (exclude timestamp, regime labels, OHLCV)
        exclude_cols = ['timestamp', 'regime', 'regime_id', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in splits['train'].columns if col not in exclude_cols]

        if len(feature_cols) == 0:
            logger.warning("No features to scale!")
            return splits

        logger.info(f"   Scaling {len(feature_cols)} features with RobustScaler")

        # Fit scaler on train set
        self.scaler = RobustScaler()
        self.scaler.fit(splits['train'][feature_cols])

        # Transform all sets
        for split_name in ['train', 'val', 'test']:
            splits[split_name][feature_cols] = self.scaler.transform(splits[split_name][feature_cols])

        return splits

    def _print_split_stats(self, splits: Dict[str, pd.DataFrame]):
        """Print statistics for each split."""
        logger.info("\n4. Split Statistics:")

        for split_name, split_df in splits.items():
            logger.info(f"\n   {split_name.upper()}:")
            logger.info(f"     Samples: {len(split_df):,}")

            # Regime distribution
            if 'regime' in split_df.columns:
                regime_counts = split_df['regime'].value_counts()
                logger.info(f"     Regime distribution:")
                for regime, count in regime_counts.items():
                    pct = count / len(split_df) * 100
                    logger.info(f"       {regime}: {count} ({pct:.1f}%)")

    def save_splits(self, splits: Dict[str, pd.DataFrame], output_dir: str):
        """
        Save splits to Parquet files.

        Args:
            splits: Dictionary of split DataFrames
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for split_name, split_df in splits.items():
            output_file = output_path / f"{split_name}.parquet"
            split_df.to_parquet(output_file, compression='snappy', index=False)
            logger.info(f"   Saved {split_name} to {output_file}")

        # Save scaler
        if self.scaler:
            scaler_file = output_path / "scaler.pkl"
            joblib.dump(self.scaler, scaler_file)
            logger.info(f"   Saved scaler to {scaler_file}")


if __name__ == '__main__':
    # Test splitter
    dates = pd.date_range('2024-01-01', periods=1000, freq='1H')

    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(1000).cumsum() + 1.1000,
        'high': np.random.randn(1000).cumsum() + 1.1010,
        'low': np.random.randn(1000).cumsum() + 1.0990,
        'close': np.random.randn(1000).cumsum() + 1.1000,
        'volume': np.random.randint(100, 1000, 1000),
        'adx': np.random.uniform(10, 40, 1000),
        'ema_200': np.random.randn(1000).cumsum() + 1.1000,
        'atr': np.random.uniform(0.0001, 0.001, 1000),
        'feature_1': np.random.randn(1000),
        'feature_2': np.random.randn(1000) * 100,
    })

    splitter = DataSplitter(train_pct=0.7, val_pct=0.15, test_pct=0.15)
    splits = splitter.split_and_label(df)

    print(f"\nTrain shape: {splits['train'].shape}")
    print(f"Val shape: {splits['val'].shape}")
    print(f"Test shape: {splits['test'].shape}")
    print(f"\nTrain regime distribution:")
    print(splits['train']['regime'].value_counts())
